/* Upload → /predict → wireframe overlay synced to the video, with three tabs:
   frame analysis (live readouts), stroke analysis (scores), session summary. */

/*
  HOW THIS FILE IS ORGANIZED
  --------------------------
  This is all the behavior of the page. index.html is just the boxes and
  buttons; this file makes them do things. Reading order, top to bottom:

    1. Grab every element from the page we'll need later.
    2. Score → color math (used by the bars, the skeleton, and the badge).
    3. Lookup tables saying which tile shows which number, and which body
       points get connected by a line.
    4. The wireframe: drawing it and keeping it lined up with the video.
    5. Tabs.
    6. The upload flow (pick a file → POST it → get JSON back).
    7. Filling in the results: the chart and the summary stats.

  The one idea that makes the whole file work: the server sends back several
  arrays that are all the SAME LENGTH and in the SAME ORDER. Row 5 of the
  body-point list, row 5 of the hip-angle list, and pose.t[5] all describe the
  same instant of the video. So "what should be on screen right now?" is
  always the same question: find the right row number, then read row 5 of
  everything. That row number is what frameAt() below computes.
*/

/* ---------- every element from index.html that we need to touch ---------- */

const form = document.getElementById("form");
const statusEl = document.getElementById("status");
const fileInput = document.getElementById("video");
const fileLabel = document.getElementById("file-label");
const submitBtn = document.getElementById("submit");
const poseWrap = document.getElementById("pose-wrap");
const poseVideo = document.getElementById("pose-video");
const poseCanvas = document.getElementById("pose-canvas");
const poseNote = document.getElementById("pose-note");
const strokeBadge = document.getElementById("stroke-badge");
const badgeStroke = document.getElementById("badge-stroke");
const badgeScore = document.getElementById("badge-score");
const chartEmpty = document.getElementById("chart-empty");
const tabsNav = document.getElementById("tabs");
const avgValue = document.getElementById("avg-value");

// Reads one of the color variables defined at the top of static/style.css,
// e.g. cssVar("--c-hip") gives back the pink used for the hip.
// Doing it this way means the colors live in ONE place: if I change the hip
// color in the CSS, both the dot next to the "Hip angle" tile and the ring
// drawn on the hip in the video change together automatically.
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------- grade → color (0 red … 110 green) ---------- */

// Absolute hue on the fixed 0–110 scale.
//
// In HSL color, "hue" is an angle: 0 is red, 60 is yellow, 120 is green.
// So this maps a score onto that range — score 0 → hue 0 (red), score 110 →
// hue 120 (green), and everything in between lands proportionally.
// The Math.min/Math.max clamps the score into 0–110 first, because Ridge
// regression doesn't know scores are supposed to stop at 100 and can hand
// back 104 or -3. We clamp for COLOR only; the number shown to the user is
// never clamped.
function absHue(grade) {
  return (120 * Math.min(110, Math.max(0, grade))) / 110;
}

// The absolute scale alone makes a uniformly good video all near-identical
// greens, so per video we widen the spectrum: the best stroke keeps its true
// color and the worst is pulled redder until the hues span at least MIN_HUE_SPREAD.
const MIN_HUE_SPREAD = 50;
// The current stretch, recalculated for each video. gMin/gMax are the worst
// and best scores; hueLow/hueHigh are the colors those two get mapped to.
let scale = { gMin: 0, gMax: 110, hueLow: 0, hueHigh: 120 };

// Works out the color stretch for one video. Called once per analysis, before
// anything gets drawn.
//
// The problem it solves: if every stroke in a video scores 70–80, the absolute
// scale gives them all nearly the same green and you can't tell them apart.
// So the best stroke keeps its true color (hueHigh), and the worst stroke gets
// pushed toward red until the two are at least 50 hue-degrees apart.
//
// The Math.min in hueLow is what makes it a "stretch" and never a "squash":
// it takes whichever is redder, the worst stroke's real color or the pulled-
// down one. So a video that already has a big spread is left alone, and only
// a bunched-up video gets exaggerated.
function setGradeScale(gradeList) {
  const gMin = Math.min(...gradeList);
  const gMax = Math.max(...gradeList);
  const hueHigh = absHue(gMax);
  const hueLow = Math.min(absHue(gMin), Math.max(0, hueHigh - MIN_HUE_SPREAD));
  scale = { gMin, gMax, hueLow, hueHigh };
}

// Turns one score into an actual CSS color string, using the stretch that
// setGradeScale worked out. THIS is the function that colors everything —
// the chart bars, the skeleton, the badge, the summary numbers — so they can
// never disagree with each other.
//
// `light` is the HSL lightness percentage. Higher = paler. Different parts of
// the UI pass different values so text stays readable on the dark background
// (bars use 50, the badge 65) while still being the same hue.
function gradeColor(grade, light = 55) {
  let hue;
  // Guard for a video with only one stroke, or where every stroke scored
  // exactly the same: there's no range to stretch across, and dividing by
  // (gMax - gMin) would be division by zero. Fall back to absolute colors.
  if (scale.gMax === scale.gMin) {
    hue = absHue(grade);
  } else {
    // t is "how far between worst and best is this stroke", 0 to 1.
    // Then use t to slide between hueLow and hueHigh. This is linear
    // interpolation — the same idea as a weighted average.
    const t = Math.min(1, Math.max(0, (grade - scale.gMin) / (scale.gMax - scale.gMin)));
    hue = scale.hueLow + t * (scale.hueHigh - scale.hueLow);
  }
  return `hsl(${Math.round(hue)} 75% ${light}%)`;
}

/* ---------- per-tab config ---------- */

// Gray, used for the skeleton when there's no stroke color to show.
const NEUTRAL_STROKE = "#8b949e";
// Near-black and see-through. Drawn UNDER the real skeleton lines as a thicker
// outline, so the figure stays visible against a white shirt or a bright wall.
const UNDER_STROKE = "rgba(13, 15, 18, 0.55)";

// Tab 1 readouts: metric key in the response → tile element + formatter.
//
// One entry per tile on the Frame analysis tab. `key` is the name of the array
// inside the server's "metrics" object, `el` is the tile to write into, and
// `fmt` turns the raw number into display text with the right units and
// number of decimal places. Keeping these together in a list means
// updateForRow() can just loop instead of repeating itself six times.
const READOUTS = [
  { key: "hip_angle", el: document.getElementById("val-hip"), fmt: (v) => `${v.toFixed(0)}°` },
  { key: "knee_angle", el: document.getElementById("val-knee"), fmt: (v) => `${v.toFixed(0)}°` },
  { key: "elbow_angle", el: document.getElementById("val-elbow"), fmt: (v) => `${v.toFixed(0)}°` },
  { key: "wrist_x", el: document.getElementById("val-wrist"), fmt: (v) => v.toFixed(2) },
  { key: "wrist_v", el: document.getElementById("val-vel"), fmt: (v) => `${v.toFixed(2)}/s` },
  { key: "wrist_a", el: document.getElementById("val-acc"), fmt: (v) => `${v.toFixed(1)}/s²` },
];

// Tab 2 radio overlays: feature key → right-axis title.
//
// The six things the model measures. The key on the left is the name Python
// uses (see feature_extraction.py); the text on the right is the human label
// with units, used to title the extra y-axis when you overlay that feature on
// the chart. Units differ wildly between them — degrees vs. a 0-to-1 fraction
// — which is exactly why the overlay needs its own axis instead of sharing
// the score axis.
const FEATURE_AXIS = {
  min_hip_angle: "Min hip angle (deg)",
  fastest_hip_velocity_timing: "Hip velocity timing (fraction of stroke)",
  knee_min_accel_timing: "Knee accel timing (fraction of stroke)",
  body_angle_at_catch: "Body angle at catch (deg)",
  leg_back_lag: "Leg–back lag (fraction, + = legs first)",
  elbow_angle_range: "Elbow angle range (deg)",
};

/* ---------- wireframe overlay ---------- */

// Joint pairs to connect, by MediaPipe landmark name. The 10 face landmarks
// are skipped; the nose is drawn as a ring for the head.
//
// Each pair is one line segment of the stick figure: shoulders across, hips
// across, the two sides of the torso, then each arm (shoulder→elbow→wrist)
// and each leg (hip→knee→ankle→heel→toe). 16 lines total.
//
// These are NAMES, not numbers, on purpose. The server sends its own list of
// landmark names with every response, and resolvePoseIndices() matches these
// names against that list to get the numbers. So if the joint list ever
// changes on the Python side, this keeps working instead of silently drawing
// lines between the wrong body parts.
const POSE_EDGES = [
  ["left_shoulder", "right_shoulder"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["left_ankle", "left_heel"],
  ["left_heel", "left_foot_index"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["right_ankle", "right_heel"],
  ["right_heel", "right_foot_index"],
];

// Skip drawing rather than freeze on a stale pose when playback sits in a
// stretch where no pose was detected.
//
// Remember that extraction.py throws away frames where MediaPipe found no
// body, so there can be gaps in time. Without this check, the skeleton would
// freeze in its last known position and look like a bug. A quarter of a
// second is about 7 sampled frames — long enough not to trigger on normal
// gaps, short enough that a real dropout gets caught.
const POSE_MAX_GAP_S = 0.25;

/* ---- Everything below is state: what the page currently knows. All of it
        gets reset by resetAnalysis() when a new video is chosen. ---- */

let data = null;          // full /predict response
let pose = null;          // data.pose: { t, xy, width, height, landmarks }
let poseEdgeIdx = [];     // POSE_EDGES resolved to landmark indices
let poseNoseIdx = -1;
let ringJoints = null;    // near-side landmark indices for Tab 1 rings
let strokeOfRow = null;   // row index → stroke number (or -1)
let grades = [];
let poseCursor = 0;       // monotonic cursor into pose.t
let poseRaf = 0;
let objectUrl = null;
let currentTab = "strokes";
let chart = null;
let lastRow = -1;
// Set only in the no-video fallback, where nothing drives video.currentTime.
let standaloneTime = null;

// What time is it in the clip right now?
//
// Normally the <video> element's own currentTime is the clock, and everything
// follows it. But in the codec-fallback mode (see useSkeletonOnly) there IS no
// playing video, so a made-up clock in standaloneTime takes over instead.
// Every drawing function asks this rather than reading currentTime directly,
// so neither mode needs to know which one is active.
function currentPoseTime() {
  return standaloneTime !== null ? standaloneTime : (poseVideo.currentTime || 0);
}

// Translates body-part NAMES into position numbers, once per analysis.
//
// The server sends coordinates as one long flat list per row — 66 numbers,
// x and y for each of the 33 body points, in a fixed order. To draw the left
// elbow we need to know it's point number 13, so its x is at position 26 and
// its y at position 27. This function builds that name→number map from the
// landmark list the server sent, then uses it to convert POSE_EDGES from
// names into index pairs, so drawing later is fast and does no lookups.
//
// It also picks out the four joints that get colored rings on the Frame tab,
// using near_side from the response — the side facing the camera. That
// matters because MediaPipe is basically guessing where the far arm and leg
// are (they're hidden behind the body), so we only ever highlight the side we
// can actually see.
function resolvePoseIndices(names) {
  const byName = new Map(names.map((n, i) => [n, i]));
  poseEdgeIdx = POSE_EDGES
    .map(([a, b]) => [byName.get(a), byName.get(b)])
    // Drop any edge naming a joint the server didn't send, so a mismatch
    // means a missing line rather than a crash mid-draw.
    .filter(([a, b]) => a !== undefined && b !== undefined);
  poseNoseIdx = byName.has("nose") ? byName.get("nose") : -1;
  const side = data.near_side === "left" ? "left" : "right";
  ringJoints = {
    hip: byName.get(`${side}_hip`),
    knee: byName.get(`${side}_knee`),
    elbow: byName.get(`${side}_elbow`),
    wrist: byName.get(`${side}_wrist`),
  };
}

// Matches the canvas size to the video box, then redraws.
//
// A canvas has two different sizes: how big it LOOKS on the page (set by CSS)
// and how many actual pixels it stores (width/height set here). On a retina
// screen those differ — devicePixelRatio is 2 or 3 — and if you ignore that,
// every line comes out blurry. So we make the pixel grid dpr times bigger
// than the on-screen box, and drawPoseAtCurrentTime scales its drawing to
// match.
//
// Called whenever the box changes size: on video load, and from the
// ResizeObserver further down that watches for window resizes and layout
// shifts.
function sizePoseCanvas() {
  const rect = poseWrap.getBoundingClientRect();
  // A hidden or not-yet-laid-out element measures 0×0; bail rather than
  // create a zero-size canvas.
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  poseCanvas.width = Math.round(rect.width * dpr);
  poseCanvas.height = Math.round(rect.height * dpr);
  drawPoseAtCurrentTime();
}

// Find the sampled frame nearest `target` seconds. pose.t is sorted, and
// playback moves in small steps, so walk the cursor instead of re-searching.
//
// This is the "which row?" function the header comment mentioned, and it runs
// up to 60 times a second, so it's written to be cheap.
//
// The naive version would search the whole pose.t list every time. But the
// video only moves forward a tiny amount between frames, so the answer is
// almost always the same row as last time or the next one. poseCursor
// remembers where we were and just steps from there — usually zero or one
// steps of work.
//
// The two while loops handle both directions: the first walks forward during
// normal playback, the second walks backward if you drag the scrubber back.
// The `if` at the end handles the fact that the loops leave the cursor on the
// row just BEFORE the target time — if the following row is actually closer,
// use that one instead, so the skeleton doesn't lag half a frame behind.
function frameAt(target) {
  const t = pose.t;
  while (poseCursor < t.length - 1 && t[poseCursor + 1] <= target) poseCursor++;
  while (poseCursor > 0 && t[poseCursor] > target) poseCursor--;
  const next = poseCursor + 1;
  if (next < t.length && Math.abs(t[next] - target) < Math.abs(t[poseCursor] - target)) {
    poseCursor = next;
  }
  return poseCursor;
}

// What color should the stick figure be for this row?
//
// The skeleton is drawn identically on all three tabs — only the color
// changes, and that's decided here:
//   summary → the fixed lime accent color
//   strokes → the color of whichever stroke is playing right now, so the
//             figure matches its bar in the chart
//   frames  → plain gray, because on that tab the colored RINGS are the
//             thing you're meant to look at
//
// strokeOfRow[row] is -1 for frames that fall outside any stroke (before the
// first catch or after the last), and those draw gray too.
function skeletonColor(row) {
  if (currentTab === "summary") return cssVar("--accent");
  if (currentTab === "strokes") {
    const k = strokeOfRow[row];
    return k >= 0 ? gradeColor(grades[k]) : NEUTRAL_STROKE;
  }
  return NEUTRAL_STROKE;
}

// Draws one colored circle around one joint (used on the Frame analysis tab).
//
// `row` is the flat list of 66 coordinates for the current moment, so a joint
// numbered `joint` has its x at position joint*2 and its y at joint*2+1. Those
// coordinates arrive as fractions from 0 to 1, so multiplying by the canvas
// width and height turns them into actual pixel positions.
//
// ctx.w and ctx.h aren't standard canvas properties — they're stashed onto the
// context by drawPoseAtCurrentTime just before calling this, as a shortcut for
// passing the size in.
//
// The early return covers a joint the server didn't send, where byName.get()
// in resolvePoseIndices gave back undefined.
function drawRing(ctx, row, joint, color, radius, width) {
  if (joint === undefined) return;
  ctx.beginPath();
  ctx.arc(row[joint * 2] * ctx.w, row[joint * 2 + 1] * ctx.h, radius, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

// Draws the whole stick figure for whatever moment the video is at.
//
// This is the heart of the overlay and gets called constantly — every screen
// refresh while playing, plus once on any seek, resize, or tab switch. The
// steps are: work out the size, clear whatever was there, find the right row,
// bail if there's no pose for this moment, then draw lines → joints → head →
// rings, and finally hand the row number to updateForRow so the tiles and
// badge match what's on screen.
function drawPoseAtCurrentTime() {
  const ctx = poseCanvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  // Convert the canvas's pixel size back into on-screen size, so all the
  // drawing below can be written in normal screen units.
  const w = poseCanvas.width / dpr;
  const h = poseCanvas.height / dpr;

  // Scale every drawing operation by dpr so it stays sharp on retina screens,
  // then wipe the previous frame — otherwise the figures would smear together.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  // No results yet (page just loaded, or a new file was picked): a cleared
  // canvas is the correct end state, so stop here.
  if (!pose || !pose.t.length) return;

  const now = currentPoseTime();
  const i = frameAt(now);
  // The nearest pose is too far away in time — MediaPipe lost the body here.
  // Draw nothing and blank the tiles, rather than freezing on old data.
  if (Math.abs(pose.t[i] - now) > POSE_MAX_GAP_S) {
    updateForRow(-1);
    return;
  }
  const row = pose.xy[i];

  // Scale strokes with the panel so the figure reads the same at any width.
  // 640 is just a reference width: on a 640px-wide panel unit is 1 and the
  // numbers below are plain pixels. On a bigger panel everything scales up
  // proportionally, so the figure doesn't look spindly on a large screen.
  const unit = Math.max(w, h) / 640;
  // Round off the line ends and corners so the joints look like joints
  // instead of sharp mitred corners.
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Dark under-stroke keeps the figure legible over light clothing.
  //
  // Two passes over the SAME set of lines: first a fat dark translucent one,
  // then the real colored one on top and thinner. The dark one peeks out at
  // the edges and acts as an outline. Drawing order matters — the second pass
  // covers the middle of the first.
  const mainColor = skeletonColor(i);
  for (const pass of [
    { color: UNDER_STROKE, width: 7.5 * unit },
    { color: mainColor, width: 4 * unit },
  ]) {
    ctx.strokeStyle = pass.color;
    ctx.lineWidth = pass.width;
    // One beginPath + one stroke for all 16 lines, rather than stroking each
    // separately — same picture, much less work for the browser.
    ctx.beginPath();
    for (const [a, b] of poseEdgeIdx) {
      ctx.moveTo(row[a * 2] * w, row[a * 2 + 1] * h);
      ctx.lineTo(row[b * 2] * w, row[b * 2 + 1] * h);
    }
    ctx.stroke();
  }

  // Joints.
  //
  // Small off-white dots at each joint. poseEdgeIdx.flat() flattens the pairs
  // into one long list of joint numbers, and the Set removes duplicates —
  // most joints appear in two or three edges (a knee is in both the thigh and
  // the shin line), and without the Set we'd draw the same dot repeatedly.
  ctx.fillStyle = "#f5f7f4";
  const seen = new Set(poseEdgeIdx.flat());
  for (const j of seen) {
    ctx.beginPath();
    ctx.arc(row[j * 2] * w, row[j * 2 + 1] * h, 3.2 * unit, 0, Math.PI * 2);
    ctx.fill();
  }

  // The head: one bigger circle at the nose. The other nine face points
  // (eyes, ears, mouth) are skipped entirely — a face made of dots looks bad
  // and tells you nothing about rowing form.
  if (poseNoseIdx >= 0) {
    ctx.beginPath();
    ctx.arc(row[poseNoseIdx * 2] * w, row[poseNoseIdx * 2 + 1] * h, 9 * unit, 0, Math.PI * 2);
    ctx.strokeStyle = mainColor;
    ctx.lineWidth = 3 * unit;
    ctx.stroke();
  }

  // Tab 1: color-coded rings marking the joints the readout tiles describe.
  //
  // Each ring uses the same CSS color variable as its tile, so you can look
  // from the amber number to the amber circle and know they're the same
  // thing. Only drawn on the Frame tab; the other tabs would just be cluttered.
  if (currentTab === "frames" && ringJoints) {
    // Stash the canvas size on the context so drawRing can reach it.
    ctx.w = w;
    ctx.h = h;
    drawRing(ctx, row, ringJoints.hip, cssVar("--c-hip"), 8 * unit, 2.5 * unit);
    drawRing(ctx, row, ringJoints.knee, cssVar("--c-knee"), 8 * unit, 2.5 * unit);
    drawRing(ctx, row, ringJoints.elbow, cssVar("--c-elbow"), 8 * unit, 2.5 * unit);
    // Three concentric rings: wrist position, velocity, acceleration.
    // The wrist is one point but has three tiles describing it, so it gets
    // three rings at increasing radius rather than three overlapping circles.
    drawRing(ctx, row, ringJoints.wrist, cssVar("--c-wrist"), 6 * unit, 2.5 * unit);
    drawRing(ctx, row, ringJoints.wrist, cssVar("--c-vel"), 10 * unit, 2.5 * unit);
    drawRing(ctx, row, ringJoints.wrist, cssVar("--c-acc"), 14 * unit, 2.5 * unit);
  }

  updateForRow(i);
}

// Update the tiles and the stroke badge for the pose row under the playhead.
// row = -1 means "no pose here" (gap or before analysis).
//
// The first two lines are the important optimization in this file. The draw
// loop runs at the screen's refresh rate — 60 times a second — but the pose
// data was only sampled around 15–30 times a second, so the row number often
// doesn't change between draws. Writing the same text into the same tiles 60
// times a second is wasted effort and makes the page sluggish, so if the row
// hasn't changed we leave immediately.
//
// (Elsewhere the code sets lastRow = -2, an impossible row number, to
// deliberately defeat this check and force a refresh — after a tab switch,
// for instance, where the tiles need rewriting even though time didn't move.)
function updateForRow(row) {
  if (row === lastRow) return;
  lastRow = row;

  // The six live numbers on the Frame tab. Only bothered with when that tab
  // is actually visible. A missing value shows an em dash instead of "NaN".
  if (currentTab === "frames" && data) {
    for (const { key, el, fmt } of READOUTS) {
      const v = row >= 0 ? data.metrics[key][row] : null;
      el.textContent = v === null || v === undefined ? "—" : fmt(v);
    }
  }

  // The little "Stroke 3 / 77.1" badge floating over the video. Shown only on
  // the Stroke tab, and only when the current moment is inside a stroke —
  // hidden during the lead-in before the first catch. k is a 0-based stroke
  // number so it gets +1 for display.
  const k = row >= 0 && strokeOfRow ? strokeOfRow[row] : -1;
  if (currentTab === "strokes" && k >= 0) {
    badgeStroke.textContent = `Stroke ${k + 1}`;
    badgeScore.textContent = grades[k].toFixed(1);
    strokeBadge.style.color = gradeColor(grades[k], 65);
    strokeBadge.hidden = false;
  } else {
    strokeBadge.hidden = true;
  }
}

// The animation loop: draw, then ask the browser to call us again before the
// next screen refresh. requestAnimationFrame is the right tool because it
// syncs to the display and automatically pauses in a background tab, unlike
// a plain timer.
function poseLoop() {
  drawPoseAtCurrentTime();
  poseRaf = requestAnimationFrame(poseLoop);
}

// Start the loop, but only if it isn't already running. poseRaf holds the
// pending request's id and doubles as the "is it running?" flag — without
// this guard, two play events would start two loops that both keep going.
function startPoseLoop() {
  if (!poseRaf) poseRaf = requestAnimationFrame(poseLoop);
}

// Stop the loop when the video pauses or ends, so we're not redrawing an
// unchanging picture 60 times a second forever. The final draw makes sure the
// last frame is correct rather than one frame stale.
function stopPoseLoop() {
  if (poseRaf) cancelAnimationFrame(poseRaf);
  poseRaf = 0;
  drawPoseAtCurrentTime();
}

// Wire the loop to the video's own events. Between them these cover every way
// the picture can need updating: play/pause/ended start and stop the loop,
// "seeked" redraws once after scrubbing while paused (no loop is running
// then, so without this the skeleton wouldn't follow the scrubber), and
// "loadeddata" sizes the canvas once the video's real dimensions are known.
poseVideo.addEventListener("play", startPoseLoop);
poseVideo.addEventListener("pause", stopPoseLoop);
poseVideo.addEventListener("ended", stopPoseLoop);
poseVideo.addEventListener("seeked", drawPoseAtCurrentTime);
poseVideo.addEventListener("loadeddata", sizePoseCanvas);

// iPhone .MOV clips are HEVC: Safari and most Chrome desktop builds decode
// them, Firefox and Chrome-on-Linux often can't. Fall back to the skeleton
// on its own rather than showing an empty box.
poseVideo.addEventListener("error", useSkeletonOnly);

// Switches the panel into "no video, skeleton only" mode.
//
// The analysis itself is completely fine in this situation — the server
// decoded the video with OpenCV, which handles far more formats than a
// browser does. It's only the on-screen playback that failed. So rather than
// showing an error, hide the video element and animate the wireframe by
// itself against the dark background.
//
// The .no-video CSS class does the hiding, and also makes the box take the
// clip's aspect ratio (from the --pose-aspect variable set in renderResults),
// since without a video there's nothing to give the box a shape.
//
// The guard at the top matters because the error event can fire more than
// once for the same file, and re-entering would restart the animation.
function useSkeletonOnly() {
  if (poseWrap.classList.contains("no-video")) return;
  poseWrap.classList.add("no-video");
  poseNote.textContent =
    "Your browser can't play this video format, so the wireframe is shown on its own.";
  sizePoseCanvas();
  // Nothing is driving currentTime now, so animate the skeleton directly.
  if (pose && pose.t.length) playSkeletonStandalone();
}

// Runs the skeleton on its own clock, looping forever, for the no-video case.
//
// performance.now() is a millisecond stopwatch. Subtracting the start time
// gives elapsed real time, /1000 converts to seconds, and the % span wraps it
// back to the beginning at the end of the clip — so the stroke replays on a
// loop. Adding pose.t[0] shifts it to line up with the real timestamps, which
// don't necessarily start at zero.
//
// Setting standaloneTime is what makes currentPoseTime() return this made-up
// clock instead of the dead video element, so every drawing function keeps
// working with no changes.
function playSkeletonStandalone() {
  const span = pose.t[pose.t.length - 1] - pose.t[0];
  // A single-frame clip has no duration to loop over; leave the static draw.
  if (span <= 0) return;
  const start = performance.now();
  const tick = () => {
    standaloneTime = pose.t[0] + ((performance.now() - start) / 1000) % span;
    drawPoseAtCurrentTime();
    poseRaf = requestAnimationFrame(tick);
  };
  // Cancel any loop already running, so analyzing a second video doesn't
  // leave two animations fighting over the canvas.
  if (poseRaf) cancelAnimationFrame(poseRaf);
  standaloneTime = pose.t[0];
  poseRaf = requestAnimationFrame(tick);
}

// Redraw whenever the video box changes size for ANY reason — window resize,
// phone rotation, a panel opening and pushing the layout around. Without
// this the canvas would keep its old pixel size and the skeleton would sit
// offset from the body.
new ResizeObserver(sizePoseCanvas).observe(poseWrap);

/* ---------- tabs ---------- */

// Handles clicking one of the three tab buttons.
//
// This is one listener on the container rather than three on the buttons —
// closest() walks up from whatever was actually clicked to find the button.
// Then: remember the new tab, restyle the buttons, show the matching panel
// and hide the other two, and redraw so the skeleton picks up the new tab's
// coloring immediately instead of waiting for the next played frame.
tabsNav.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-tab]");
  // Ignore clicks on the gap between buttons, and on the already-active tab.
  if (!btn || btn.dataset.tab === currentTab) return;
  currentTab = btn.dataset.tab;
  for (const b of tabsNav.querySelectorAll("button")) {
    const active = b === btn;
    b.classList.toggle("active", active);
    // aria-selected tells screen readers which tab is current.
    b.setAttribute("aria-selected", String(active));
  }
  document.getElementById("panel-frames").hidden = currentTab !== "frames";
  document.getElementById("panel-strokes").hidden = currentTab !== "strokes";
  document.getElementById("panel-summary").hidden = currentTab !== "summary";
  // Chart.js can't size a canvas inside a hidden panel, so build it on first view.
  // A hidden element measures 0×0, so a chart built then would be invisible.
  // If it already exists, resize() re-measures now that the panel is visible.
  if (currentTab === "strokes" && data) {
    if (!chart) buildChart();
    else chart.resize();
  }
  lastRow = -2; // force a badge/tile refresh
  drawPoseAtCurrentTime();
});

/* ---------- upload flow ---------- */

// Blank out every readout so a newly chosen video starts from a clean slate.
//
// Without this, picking a second video would leave the first one's scores and
// stats on screen until the new analysis came back — which takes a minute and
// would look like the results were already in. Clears the stored response,
// the chart, the six live tiles, the average, and all six summary tiles
// (including their little sub-labels and their colors).
function resetAnalysis() {
  data = null;
  pose = null;
  poseCursor = 0;
  lastRow = -1;
  standaloneTime = null;
  strokeBadge.hidden = true;
  // destroy() is Chart.js's own cleanup — just dropping the reference would
  // leave its event listeners attached to the canvas.
  if (chart) { chart.destroy(); chart = null; }
  chartEmpty.hidden = false;
  for (const { el } of READOUTS) el.textContent = "—";
  avgValue.textContent = "—";
  avgValue.style.color = "";
  for (const id of ["sum-strokes", "sum-rate", "sum-avg", "sum-best", "sum-worst", "sum-consistency"]) {
    const el = document.getElementById(id);
    el.textContent = "—";
    el.style.color = "";
  }
  for (const id of ["sum-strokes-sub", "sum-best-sub", "sum-worst-sub", "sum-consistency-sub"]) {
    document.getElementById(id).textContent = "";
  }
}

// Fires when a file is picked. Note this happens BEFORE any uploading — the
// video starts playing immediately, and analysis only begins when Analyze is
// pressed.
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  // The ?. means "only read .name if file exists", covering the case where
  // the user opens the picker and cancels.
  fileLabel.textContent = file?.name || "Choose a video…";
  if (!file) return;

  // Play the clip straight from the local File — nothing is re-downloaded
  // from the server, and the preview is available before scoring finishes.
  //
  // createObjectURL makes a temporary address pointing at the file already on
  // your computer. revokeObjectURL releases the previous one first, because
  // those addresses hold onto the file's memory until explicitly freed.
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  resetAnalysis();
  // Drop both state classes: "empty" hides the placeholder text, and
  // "no-video" might be left over from a previous unplayable clip.
  poseWrap.classList.remove("no-video", "empty");
  poseNote.textContent = "";
  poseVideo.src = objectUrl;
});

// Shows a message in red. The .error class is what makes it red; the normal
// status messages use the same element without it.
function showError(msg) {
  statusEl.textContent = msg;
  statusEl.classList.add("error");
}

// The Analyze button: sends the video to the server and waits for scores.
//
// preventDefault stops the browser's default form behavior, which would be to
// navigate away to a new page and lose everything. FormData packages the file
// as a multipart upload under the field name "video", which is exactly what
// app.py's predict() looks for.
//
// async/await means the code reads top-to-bottom even though the upload takes
// a minute: await pauses this function without freezing the page.
//
// Two different failure modes are handled. The catch block is for the request
// never completing — server down, connection dropped. The `if (!res.ok ...)`
// is for the server answering normally with an error message inside, like
// "No rowing strokes detected." The finally block re-enables the button
// either way, so a failure can be retried.
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  // Clear any previous error styling and disable the button so an impatient
  // double-click doesn't upload the same video twice.
  statusEl.classList.remove("error");
  statusEl.textContent = "Uploading and analyzing… this can take a minute.";
  submitBtn.disabled = true;

  const body = new FormData();
  body.append("video", file);

  try {
    const res = await fetch("/predict", { method: "POST", body });
    const payload = await res.json();
    if (!res.ok || payload.error) {
      showError(payload.error || "Analysis failed.");
      return;
    }
    statusEl.textContent = "";
    renderResults(payload);
  } catch (err) {
    showError("Request failed. Is the server running?");
    console.error(err);
  } finally {
    submitBtn.disabled = false;
  }
});

/* ---------- results ---------- */

// Takes the server's JSON and sets up everything that depends on it.
//
// Runs once per successful analysis, in order: store the response, pull out
// the scores, work out the color stretch for this video, convert landmark
// names to numbers, build the row→stroke lookup, then fill in the average and
// the summary tab and get the canvas drawing.
function renderResults(payload) {
  data = payload;
  pose = payload.pose;
  // Reset the row-finding cursor, since we're starting a different clip.
  poseCursor = 0;
  // -2 is an impossible row number, which forces the next updateForRow to
  // actually redraw instead of thinking nothing changed.
  lastRow = -2;
  grades = payload.strokes.map((s) => s.predicted_grade);
  setGradeScale(grades);
  resolvePoseIndices(pose.landmarks);

  // Row index → stroke, for O(1) lookup while drawing. Ranges are inclusive
  // and consecutive strokes share a boundary row.
  //
  // Rather than searching the stroke ranges on every draw, precompute the
  // answer for every row once: a list as long as the video where each entry
  // says which stroke that moment belongs to. Filled with -1 first so that
  // rows outside every stroke (before the first catch, after the last) have a
  // sensible "no stroke" value. Int16Array is a compact numbers-only list;
  // e + 1 because fill() stops one short and the ranges include their end row.
  strokeOfRow = new Int16Array(pose.t.length).fill(-1);
  payload.stroke_ranges.forEach(([s, e], k) => strokeOfRow.fill(k, s, e + 1));

  // Hand the clip's shape to the CSS, so the box can hold the right aspect
  // ratio in the no-video fallback where there's no video to give it one.
  if (pose.width && pose.height) {
    poseWrap.style.setProperty("--pose-aspect", `${pose.width} / ${pose.height}`);
  }

  avgValue.textContent = Number(payload.predicted_grade_mean).toFixed(1);
  avgValue.style.color = gradeColor(payload.predicted_grade_mean, 60);
  renderSummary(payload);

  // Throw away any chart from a previous video, then rebuild — but only if
  // the Stroke tab is showing. If it isn't, the tab-switch handler builds it
  // later, once the panel is visible and can actually be measured.
  if (chart) { chart.destroy(); chart = null; }
  if (currentTab === "strokes") buildChart();

  sizePoseCanvas();
  // If this clip already failed to play, the standalone animation needs
  // restarting now that there's finally pose data to animate.
  if (poseWrap.classList.contains("no-video")) playSkeletonStandalone();
}

/* ---------- Tab 2: stroke chart + feature overlay ---------- */

// Builds the bar chart: one bar per stroke, colored by its score.
//
// Called on the first visit to the Stroke tab (or right away if that tab is
// already open). Most of what follows is Chart.js configuration — colors for
// the dark theme, axis titles, and so on.
//
// Two settings are worth understanding. suggestedMin/Max pin the score axis
// to roughly 0–110 so bars are judged against the full scale rather than
// being auto-zoomed, which would make a two-point difference look enormous.
// And the y1 axis is defined here but starts hidden — it's the right-hand
// axis that appears only when a feature is overlaid.
function buildChart() {
  chartEmpty.hidden = true;
  const ctx = document.getElementById("stroke-chart").getContext("2d");
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      // Bar labels along the bottom: "1", "2", "3"…
      labels: data.strokes.map((s) => String(s.stroke)),
      datasets: [{
        label: "Stroke score",
        data: grades,
        // Same gradeColor as the skeleton and the badge, so a bar and the
        // wireframe during that stroke are always the same color.
        backgroundColor: grades.map((g) => gradeColor(g, 50)),
        borderRadius: 4,
        // Stops bars becoming absurdly wide on a video with only 3 strokes.
        maxBarThickness: 42,
        yAxisID: "y",
        order: 2, // higher order draws first, so the feature line lands on top
      }],
    },
    options: {
      // These two together let the chart fill its container's height, which
      // the CSS controls, instead of forcing its own shape.
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        // No legend while a single dataset is shown — the axis already says
        // "Score". applyFeatureOverlay switches it on when a second one
        // appears and the two need telling apart.
        legend: { display: false, labels: { color: "rgba(245, 247, 244, 0.8)" } },
        tooltip: {
          // Hovering a bar says "Stroke 3" rather than just "3".
          callbacks: { title: (items) => `Stroke ${items[0].label}` },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Stroke", color: "rgba(245, 247, 244, 0.7)" },
          ticks: { color: "rgba(245, 247, 244, 0.7)" },
          grid: { display: false },
        },
        y: {
          title: { display: true, text: "Score", color: "rgba(245, 247, 244, 0.7)" },
          suggestedMin: 0,
          suggestedMax: 110,
          ticks: { color: "rgba(245, 247, 244, 0.7)" },
          grid: { color: "rgba(255, 255, 255, 0.06)" },
        },
        // The overlay axis on the right. Hidden until a feature is picked,
        // and its title gets rewritten each time to name that feature's units.
        y1: {
          display: false,
          position: "right",
          title: { display: true, text: "", color: "rgba(245, 247, 244, 0.7)" },
          ticks: { color: "rgba(245, 247, 244, 0.7)" },
          // Don't draw a second set of gridlines over the first — it'd be a mess.
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
  applyFeatureOverlay();
}

// Adds or removes the dashed feature line, based on which radio is selected.
//
// This is how you check whether a measurement actually tracks the score —
// overlay "min hip angle" and see whether it rises and falls with the bars.
//
// Setting datasets.length = 1 truncates the list back to just the bars,
// removing any previous overlay. That's why switching between features works
// without them piling up. An empty key is the "None" option, which just
// leaves the bars alone.
//
// The feature goes on its own right-hand axis because its units have nothing
// to do with the 0–100 score: on a shared axis, a value like 0.31 would be a
// flat line pinned to the bottom.
function applyFeatureOverlay() {
  if (!chart) return;
  const key = document.querySelector('input[name="feature"]:checked').value;
  chart.data.datasets.length = 1; // drop any previous overlay
  if (key) {
    chart.data.datasets.push({
      type: "line",
      label: FEATURE_AXIS[key],
      data: data.strokes.map((s) => s.features[key]),
      // Dashed and off-white on purpose: it must not look like it belongs to
      // the red-to-green score scale.
      borderColor: "#e8eaed",
      borderDash: [6, 4],
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: "#e8eaed",
      yAxisID: "y1",
      order: 1,
    });
  }
  // Show the right-hand axis and the legend only when there's an overlay.
  // Boolean("") is false and Boolean("min_hip_angle") is true, so this reads
  // as "is a feature selected?"
  chart.options.scales.y1.display = Boolean(key);
  chart.options.scales.y1.title.text = key ? FEATURE_AXIS[key] : "";
  chart.options.plugins.legend.display = Boolean(key);
  // Redraw with the changes. Nothing appears without this call.
  chart.update();
}

// One listener on the group of radio buttons, rather than one per button.
document.getElementById("feature-radios").addEventListener("change", applyFeatureOverlay);

/* ---------- Tab 3: session summary ---------- */

// Small helper: write text into a summary tile, optionally coloring it.
// Saves repeating getElementById three times per stat below.
function setStat(id, text, color) {
  const el = document.getElementById(id);
  el.textContent = text;
  if (color) el.style.color = color;
}

// Fills in the six Session summary tiles. All of this is worked out in the
// browser from the numbers already sent — the server does no summary math.
function renderSummary(payload) {
  const n = payload.strokes_detected;
  const ranges = payload.stroke_ranges;
  const t = pose.t;

  // Stroke count, plus which side faced the camera. That sub-label is worth
  // showing because every measurement was taken from that side only.
  setStat("sum-strokes", String(n));
  document.getElementById("sum-strokes-sub").textContent =
    `${payload.near_side} side toward camera`;

  // Stroke rate in strokes per minute, the number an erg monitor shows.
  //
  // span is the time from the start of the first stroke to the end of the
  // last, in seconds — found by using the stroke ranges' row numbers to look
  // up actual timestamps in pose.t. Then strokes ÷ seconds × 60 = per minute.
  // Guarded against a zero span, which would divide by zero and show Infinity.
  const span = t[ranges[ranges.length - 1][1]] - t[ranges[0][0]];
  setStat("sum-rate", span > 0 ? (60 * n / span).toFixed(1) : "—");

  setStat("sum-avg", payload.predicted_grade_mean.toFixed(1),
    gradeColor(payload.predicted_grade_mean, 60));

  // Best and worst stroke: one pass keeping track of the highest and lowest
  // scoring positions. Both start at 0 (the first stroke) and get replaced by
  // anything better or worse. best/worst hold the POSITION, not the score, so
  // the stroke number can be reported too — +1 since positions start at 0.
  let best = 0;
  let worst = 0;
  grades.forEach((g, i) => {
    if (g > grades[best]) best = i;
    if (g < grades[worst]) worst = i;
  });
  setStat("sum-best", grades[best].toFixed(1), gradeColor(grades[best], 60));
  document.getElementById("sum-best-sub").textContent = `stroke ${best + 1}`;
  setStat("sum-worst", grades[worst].toFixed(1), gradeColor(grades[worst], 60));
  document.getElementById("sum-worst-sub").textContent = `stroke ${worst + 1}`;

  // Consistency, from the standard deviation of the scores.
  //
  // Standard deviation measures spread: average the scores, then for each
  // score take how far it is from that average, square it (so being under and
  // over both count as "off"), average those, and square-root back to the
  // original units. Small = every stroke came out about the same, which is
  // what you actually want on an erg piece.
  //
  // Turning it into a percentage is my own rough scale, not a standard
  // formula: a spread of 0 points shows 100%, a spread of 12 points shows
  // 88%, clamped so it can't go below 0. The real ± value is printed
  // underneath, since that's the honest number.
  const mean = grades.reduce((a, b) => a + b, 0) / grades.length;
  const std = Math.sqrt(grades.reduce((a, g) => a + (g - mean) ** 2, 0) / grades.length);
  setStat("sum-consistency", `${Math.max(0, Math.min(100, 100 - std)).toFixed(0)}%`);
  document.getElementById("sum-consistency-sub").textContent =
    `scores vary by ±${std.toFixed(1)} pts`;
}
