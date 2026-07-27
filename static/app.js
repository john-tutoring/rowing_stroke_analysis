/* Upload → /predict → wireframe overlay synced to the video, with three tabs:
   frame analysis (live readouts), stroke analysis (scores), session summary. */

const form = document.getElementById("form");
const statusEl = document.getElementById("status");
const fileInput = document.getElementById("video");
const fileLabel = document.getElementById("file-label");
const submitBtn = document.getElementById("submit");
const videoPanel = document.getElementById("video-panel");
const poseWrap = document.getElementById("pose-wrap");
const poseVideo = document.getElementById("pose-video");
const poseCanvas = document.getElementById("pose-canvas");
const poseNote = document.getElementById("pose-note");
const strokeBadge = document.getElementById("stroke-badge");
const analysis = document.getElementById("analysis");
const tabsNav = document.getElementById("tabs");
const avgValue = document.getElementById("avg-value");

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------- grade → color (0 red … 110 green, clamped) ---------- */

function gradeColor(grade, light = 55) {
  const g = Math.min(110, Math.max(0, grade));
  return `hsl(${Math.round((120 * g) / 110)} 75% ${light}%)`;
}

/* ---------- per-tab config ---------- */

const NEUTRAL_STROKE = "#8b949e";
const UNDER_STROKE = "rgba(13, 15, 18, 0.55)";

// Tab 1 readouts: metric key in the response → tile element + formatter.
const READOUTS = [
  { key: "hip_angle", el: document.getElementById("val-hip"), fmt: (v) => `${v.toFixed(0)}°` },
  { key: "knee_angle", el: document.getElementById("val-knee"), fmt: (v) => `${v.toFixed(0)}°` },
  { key: "elbow_angle", el: document.getElementById("val-elbow"), fmt: (v) => `${v.toFixed(0)}°` },
  { key: "wrist_x", el: document.getElementById("val-wrist"), fmt: (v) => v.toFixed(2) },
  { key: "wrist_v", el: document.getElementById("val-vel"), fmt: (v) => `${v.toFixed(2)}/s` },
  { key: "wrist_a", el: document.getElementById("val-acc"), fmt: (v) => `${v.toFixed(1)}/s²` },
];

// Tab 2 radio overlays: feature key → right-axis title.
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
const POSE_MAX_GAP_S = 0.25;

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
let currentTab = "frames";
let chart = null;
let lastRow = -1;
// Set only in the no-video fallback, where nothing drives video.currentTime.
let standaloneTime = null;

function currentPoseTime() {
  return standaloneTime !== null ? standaloneTime : (poseVideo.currentTime || 0);
}

function resolvePoseIndices(names) {
  const byName = new Map(names.map((n, i) => [n, i]));
  poseEdgeIdx = POSE_EDGES
    .map(([a, b]) => [byName.get(a), byName.get(b)])
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

function sizePoseCanvas() {
  const rect = poseWrap.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  poseCanvas.width = Math.round(rect.width * dpr);
  poseCanvas.height = Math.round(rect.height * dpr);
  drawPoseAtCurrentTime();
}

// Find the sampled frame nearest `target` seconds. pose.t is sorted, and
// playback moves in small steps, so walk the cursor instead of re-searching.
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

function skeletonColor(row) {
  if (currentTab === "summary") return cssVar("--accent");
  if (currentTab === "strokes") {
    const k = strokeOfRow[row];
    return k >= 0 ? gradeColor(grades[k]) : NEUTRAL_STROKE;
  }
  return NEUTRAL_STROKE;
}

function drawRing(ctx, row, joint, color, radius, width) {
  if (joint === undefined) return;
  ctx.beginPath();
  ctx.arc(row[joint * 2] * ctx.w, row[joint * 2 + 1] * ctx.h, radius, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawPoseAtCurrentTime() {
  const ctx = poseCanvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = poseCanvas.width / dpr;
  const h = poseCanvas.height / dpr;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!pose || !pose.t.length) return;

  const now = currentPoseTime();
  const i = frameAt(now);
  if (Math.abs(pose.t[i] - now) > POSE_MAX_GAP_S) {
    updateForRow(-1);
    return;
  }
  const row = pose.xy[i];

  // Scale strokes with the panel so the figure reads the same at any width.
  const unit = Math.max(w, h) / 640;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Dark under-stroke keeps the figure legible over light clothing.
  const mainColor = skeletonColor(i);
  for (const pass of [
    { color: UNDER_STROKE, width: 7.5 * unit },
    { color: mainColor, width: 4 * unit },
  ]) {
    ctx.strokeStyle = pass.color;
    ctx.lineWidth = pass.width;
    ctx.beginPath();
    for (const [a, b] of poseEdgeIdx) {
      ctx.moveTo(row[a * 2] * w, row[a * 2 + 1] * h);
      ctx.lineTo(row[b * 2] * w, row[b * 2 + 1] * h);
    }
    ctx.stroke();
  }

  // Joints.
  ctx.fillStyle = "#f5f7f4";
  const seen = new Set(poseEdgeIdx.flat());
  for (const j of seen) {
    ctx.beginPath();
    ctx.arc(row[j * 2] * w, row[j * 2 + 1] * h, 3.2 * unit, 0, Math.PI * 2);
    ctx.fill();
  }

  if (poseNoseIdx >= 0) {
    ctx.beginPath();
    ctx.arc(row[poseNoseIdx * 2] * w, row[poseNoseIdx * 2 + 1] * h, 9 * unit, 0, Math.PI * 2);
    ctx.strokeStyle = mainColor;
    ctx.lineWidth = 3 * unit;
    ctx.stroke();
  }

  // Tab 1: color-coded rings marking the joints the readout tiles describe.
  if (currentTab === "frames" && ringJoints) {
    ctx.w = w;
    ctx.h = h;
    drawRing(ctx, row, ringJoints.hip, cssVar("--c-hip"), 8 * unit, 2.5 * unit);
    drawRing(ctx, row, ringJoints.knee, cssVar("--c-knee"), 8 * unit, 2.5 * unit);
    drawRing(ctx, row, ringJoints.elbow, cssVar("--c-elbow"), 8 * unit, 2.5 * unit);
    // Three concentric rings: wrist position, velocity, acceleration.
    drawRing(ctx, row, ringJoints.wrist, cssVar("--c-wrist"), 6 * unit, 2.5 * unit);
    drawRing(ctx, row, ringJoints.wrist, cssVar("--c-vel"), 10 * unit, 2.5 * unit);
    drawRing(ctx, row, ringJoints.wrist, cssVar("--c-acc"), 14 * unit, 2.5 * unit);
  }

  updateForRow(i);
}

// Update the tiles and the stroke badge for the pose row under the playhead.
// row = -1 means "no pose here" (gap or before analysis).
function updateForRow(row) {
  if (row === lastRow) return;
  lastRow = row;

  if (currentTab === "frames" && data) {
    for (const { key, el, fmt } of READOUTS) {
      const v = row >= 0 ? data.metrics[key][row] : null;
      el.textContent = v === null || v === undefined ? "—" : fmt(v);
    }
  }

  const k = row >= 0 && strokeOfRow ? strokeOfRow[row] : -1;
  if (currentTab === "strokes" && k >= 0) {
    strokeBadge.textContent = `Stroke ${k + 1} · ${grades[k].toFixed(1)}`;
    strokeBadge.style.color = gradeColor(grades[k], 65);
    strokeBadge.hidden = false;
  } else {
    strokeBadge.hidden = true;
  }
}

function poseLoop() {
  drawPoseAtCurrentTime();
  poseRaf = requestAnimationFrame(poseLoop);
}

function startPoseLoop() {
  if (!poseRaf) poseRaf = requestAnimationFrame(poseLoop);
}

function stopPoseLoop() {
  if (poseRaf) cancelAnimationFrame(poseRaf);
  poseRaf = 0;
  drawPoseAtCurrentTime();
}

poseVideo.addEventListener("play", startPoseLoop);
poseVideo.addEventListener("pause", stopPoseLoop);
poseVideo.addEventListener("ended", stopPoseLoop);
poseVideo.addEventListener("seeked", drawPoseAtCurrentTime);
poseVideo.addEventListener("loadeddata", sizePoseCanvas);

// iPhone .MOV clips are HEVC: Safari and most Chrome desktop builds decode
// them, Firefox and Chrome-on-Linux often can't. Fall back to the skeleton
// on its own rather than showing an empty box.
poseVideo.addEventListener("error", useSkeletonOnly);

function useSkeletonOnly() {
  if (poseWrap.classList.contains("no-video")) return;
  poseWrap.classList.add("no-video");
  poseNote.textContent =
    "Your browser can't play this video format, so the wireframe is shown on its own.";
  sizePoseCanvas();
  // Nothing is driving currentTime now, so animate the skeleton directly.
  if (pose && pose.t.length) playSkeletonStandalone();
}

function playSkeletonStandalone() {
  const span = pose.t[pose.t.length - 1] - pose.t[0];
  if (span <= 0) return;
  const start = performance.now();
  const tick = () => {
    standaloneTime = pose.t[0] + ((performance.now() - start) / 1000) % span;
    drawPoseAtCurrentTime();
    poseRaf = requestAnimationFrame(tick);
  };
  if (poseRaf) cancelAnimationFrame(poseRaf);
  standaloneTime = pose.t[0];
  poseRaf = requestAnimationFrame(tick);
}

new ResizeObserver(sizePoseCanvas).observe(poseWrap);

/* ---------- tabs ---------- */

tabsNav.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-tab]");
  if (!btn || btn.dataset.tab === currentTab) return;
  currentTab = btn.dataset.tab;
  for (const b of tabsNav.querySelectorAll("button")) {
    const active = b === btn;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  }
  document.getElementById("panel-frames").hidden = currentTab !== "frames";
  document.getElementById("panel-strokes").hidden = currentTab !== "strokes";
  document.getElementById("panel-summary").hidden = currentTab !== "summary";
  // Chart.js can't size a canvas inside a hidden panel, so build it on first view.
  if (currentTab === "strokes" && data) {
    if (!chart) buildChart();
    else chart.resize();
  }
  lastRow = -2; // force a badge/tile refresh
  drawPoseAtCurrentTime();
});

/* ---------- upload flow ---------- */

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileLabel.textContent = file?.name || "Choose a video…";
  if (!file) return;

  // Play the clip straight from the local File — nothing is re-downloaded
  // from the server, and the preview is available before scoring finishes.
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  data = null;
  pose = null;
  poseCursor = 0;
  lastRow = -1;
  standaloneTime = null;
  strokeBadge.hidden = true;
  analysis.hidden = true;
  if (chart) { chart.destroy(); chart = null; }
  poseWrap.classList.remove("no-video");
  poseNote.textContent = "";
  poseVideo.src = objectUrl;
  videoPanel.classList.add("visible");
});

function showError(msg) {
  statusEl.textContent = msg;
  statusEl.classList.add("error");
  analysis.hidden = true;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

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

function renderResults(payload) {
  data = payload;
  pose = payload.pose;
  poseCursor = 0;
  lastRow = -2;
  grades = payload.strokes.map((s) => s.predicted_grade);
  resolvePoseIndices(pose.landmarks);

  // Row index → stroke, for O(1) lookup while drawing. Ranges are inclusive
  // and consecutive strokes share a boundary row.
  strokeOfRow = new Int16Array(pose.t.length).fill(-1);
  payload.stroke_ranges.forEach(([s, e], k) => strokeOfRow.fill(k, s, e + 1));

  if (pose.width && pose.height) {
    poseWrap.style.setProperty("--pose-aspect", `${pose.width} / ${pose.height}`);
  }
  videoPanel.classList.add("visible");
  analysis.hidden = false;

  avgValue.textContent = Number(payload.predicted_grade_mean).toFixed(1);
  avgValue.style.color = gradeColor(payload.predicted_grade_mean, 60);
  renderSummary(payload);

  if (chart) { chart.destroy(); chart = null; }
  if (currentTab === "strokes") buildChart();

  sizePoseCanvas();
  if (poseWrap.classList.contains("no-video")) playSkeletonStandalone();
}

/* ---------- Tab 2: stroke chart + feature overlay ---------- */

function buildChart() {
  const ctx = document.getElementById("stroke-chart").getContext("2d");
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.strokes.map((s) => String(s.stroke)),
      datasets: [{
        label: "Stroke score",
        data: grades,
        backgroundColor: grades.map((g) => gradeColor(g, 50)),
        borderRadius: 4,
        maxBarThickness: 42,
        yAxisID: "y",
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false, labels: { color: "rgba(245, 247, 244, 0.8)" } },
        tooltip: {
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
        y1: {
          display: false,
          position: "right",
          title: { display: true, text: "", color: "rgba(245, 247, 244, 0.7)" },
          ticks: { color: "rgba(245, 247, 244, 0.7)" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
  applyFeatureOverlay();
}

function applyFeatureOverlay() {
  if (!chart) return;
  const key = document.querySelector('input[name="feature"]:checked').value;
  chart.data.datasets.length = 1; // drop any previous overlay
  if (key) {
    chart.data.datasets.push({
      type: "line",
      label: FEATURE_AXIS[key],
      data: data.strokes.map((s) => s.features[key]),
      borderColor: "#e8eaed",
      borderDash: [6, 4],
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: "#e8eaed",
      yAxisID: "y1",
    });
  }
  chart.options.scales.y1.display = Boolean(key);
  chart.options.scales.y1.title.text = key ? FEATURE_AXIS[key] : "";
  chart.options.plugins.legend.display = Boolean(key);
  chart.update();
}

document.getElementById("feature-radios").addEventListener("change", applyFeatureOverlay);

/* ---------- Tab 3: session summary ---------- */

function setStat(id, text, color) {
  const el = document.getElementById(id);
  el.textContent = text;
  if (color) el.style.color = color;
}

function renderSummary(payload) {
  const n = payload.strokes_detected;
  const ranges = payload.stroke_ranges;
  const t = pose.t;

  setStat("sum-strokes", String(n));
  document.getElementById("sum-strokes-sub").textContent =
    `${payload.near_side} side toward camera`;

  const span = t[ranges[ranges.length - 1][1]] - t[ranges[0][0]];
  setStat("sum-rate", span > 0 ? (60 * n / span).toFixed(1) : "—");

  setStat("sum-avg", payload.predicted_grade_mean.toFixed(1),
    gradeColor(payload.predicted_grade_mean, 60));

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

  const mean = grades.reduce((a, b) => a + b, 0) / grades.length;
  const std = Math.sqrt(grades.reduce((a, g) => a + (g - mean) ** 2, 0) / grades.length);
  setStat("sum-consistency", `${Math.max(0, Math.min(100, 100 - std)).toFixed(0)}%`);
  document.getElementById("sum-consistency-sub").textContent =
    `scores vary by ±${std.toFixed(1)} pts`;
}
