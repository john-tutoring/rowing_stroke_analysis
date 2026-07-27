# StrokeScore — internal technical details

Everything in one place: how the pipeline works end to end, which parts of the app
live in which files, what `/predict` sends back, and exactly how the wireframe is
drawn and kept in sync with the video.

---

## 1. End-to-end flow

```
browser                              server (Flask, app.py)
-------                              ----------------------
choose file ──► video plays locally
click Analyze ──► POST /predict ───► save upload to uploads/
                                     extraction.py: MediaPipe pose per frame
                                     split_strokes.py: split into stroke cycles
                                     feature_extraction.py: 6 features per stroke
                                     model.joblib: predicted grade per stroke
                                     delete upload
             ◄── JSON ◄───────────── scores + pose coordinates + per-frame metrics
draw wireframe on <canvas> over the local video, synced by timestamp
```

Key design point: the server never keeps or serves the video. The browser plays the
user's local file via `URL.createObjectURL(file)`; the server returns only numbers.
That keeps the response small (~200 KB for an 11 s clip) and avoids storing user
media.

## 2. Which parts live in which files

### Backend (shipped in the Docker image)

| File | Role |
|------|------|
| `app.py` | Flask app. `GET /` serves `index.html`, `GET /static/*` the assets, `GET /health` a liveness check. `POST /predict` runs the whole pipeline (`video_to_predictions`) and builds the response, including `pose_payload()` (joint coordinates for the overlay) and `frame_metrics()` (per-frame angles/velocity for Tab 1). |
| `extraction.py` | Video → landmarks. `joints_from_video(..., return_timestamps=True)` returns `(frames, t, video_info)`: smoothed landmark rows, one capture-time per row (seconds), and `{fps, width, height}`. |
| `joint_info.py` | The 33 MediaPipe landmark names in canonical order, CSV header layout, and `COL_LOOKUP` (column name → index). Everything downstream addresses columns through this. |
| `split_strokes.py` | Stroke segmentation. The web app calls `split_cycles_with_ranges(arr)` which returns the per-stroke arrays *and* their `[start, end]` row indices; training uses `split_cycles`/`split_cycles_with_videos` (identical peak logic). Also `right_side_closer()` (which side faces the camera). |
| `feature_extraction.py` | Per-stroke features for the model, plus the reusable helpers the app also uses per-frame: `get_angle_vector` (three-joint angle per frame), `angle_abc`, `slide_axis`, `catch_index`. |
| `artifacts/model.joblib` | Fitted sklearn `Pipeline(MinMaxScaler → Ridge(alpha=1.0))`, loaded once at startup. |

### Front end

| File | Role |
|------|------|
| `index.html` | Markup only: upload form, the video panel (`<video>` + `<canvas>` + score badge), the three tab buttons, and the three panels (readout tiles / chart + radios + average / stat tiles). Loads Chart.js 4 from CDN. |
| `static/style.css` | Dark theme (CSS custom properties at the top), responsive grid for the tiles, the `.pose-wrap` layering that makes the overlay math work (see §6), tab pills, badge sizing. |
| `static/app.js` | All behavior: upload/fetch flow, the wireframe renderer and video sync, tab switching, per-tab wireframe coloring, live readout tiles, the stroke chart with feature overlays, and the summary stats. Plain functions, no framework. |

### Training-only scripts (not in the image)

`model_compare.py`, `hyperparam_tune.py`, `export_model.py`, `plot_cycle_signals.py`.

### Reference-only files (not part of the app, never deployed)

`index_reference.html`, `app_reference.py` — an earlier overlay prototype kept for
reference. `Example_UI.png` — visual inspiration for the theme.

## 3. Pose extraction details (`extraction.py`)

- MediaPipe Tasks `PoseLandmarker` (`pose_landmarker_lite.task`), `RunningMode.VIDEO`,
  one pose per frame.
- **Sampling:** pose runs on every 2nd frame (`FRAME_MODULUS = 2`). Frames where no
  pose is detected are **dropped entirely**. The row index is therefore *not* a
  uniform time grid — this is why every row carries an explicit timestamp.
- **Timestamps:** `timestamp_ms = frame_index / fps * 1000` is fed to MediaPipe and,
  for kept rows only, appended to `frame_times` (seconds).
- **Smoothing:** a 5-wide moving average (`SMOOTH_WINDOW = 5`, `np.convolve`
  mode `"valid"`) over every column, so the output has 4 fewer rows than detections.
  `smooth_timestamps()` applies the *same* window to the times, keeping `t[i]`
  aligned with row `i`.
- **Row layout** (from `joint_info.py`): `[video_index, row_grade]` + 33 joints ×
  `(X, Y, Z)` = 101 columns. The web app passes `rowing_grade=0` as a placeholder.

### The coordinate system (important)

MediaPipe returns x normalized by frame **width** and y normalized by frame
**height**. On a non-square frame that distorts every angle, so extraction stores
`y * (height / width)` — y in x's units — which makes angle math correct.
Consequence: stored y is in `[0, height/width]`, **not** `[0, 1]`.

- All angle/feature math uses the stored (angle-true) coordinates.
- The browser overlay needs plain normalized coordinates, so `pose_payload()` in
  `app.py` multiplies y back by `width / height` before sending. If the skeleton
  ever renders vertically squashed, this conversion is the first thing to check.
- Z is depth relative to the hip midpoint (more negative = closer to camera). It is
  used only for near-side detection, never for drawing or angles.

## 4. Stroke splitting (`split_strokes.py`)

- Split signal: near-side `wrist_X − ankle_X` per frame — effectively handle travel
  along the slide, one clean oscillation per stroke.
- `scipy.signal.find_peaks(signal, distance=PEAK_DISTANCE=10,
  prominence=PEAK_PROMINENCE=0.3)`. A stroke is the rows between two consecutive
  peaks, endpoints inclusive (consecutive strokes share their boundary row).
  N peaks → N−1 strokes; frames before the first and after the last peak belong to
  no stroke.
- `split_cycles_with_ranges()` returns `(cycles, ranges)` where
  `ranges[k] = [start_row, end_row]` indexes the **same row space** as the smoothed
  frames, the timestamps `t`, the pose payload, and the metric arrays. This shared
  row space is what lets the front end map "stroke k" to a video time interval:
  `t[start] … t[end]`.
- Near side: `right_side_closer()` compares mean Z of shoulder/elbow/wrist/hip/knee/
  ankle between sides across all frames. Features and the split signal always read
  the near (visible) side; the far side's landmarks are inferred by MediaPipe and
  noisy.

## 5. Features and model

Six features per stroke (`DEFAULT_FEATURE_EXTRACTORS`, order matters — it is the
model's column order):

| Feature | Definition | Units |
|---|---|---|
| `min_hip_angle` | min of knee–hip–shoulder angle | degrees |
| `fastest_hip_velocity_timing` | argmax of hip-angle velocity (`np.gradient`) / stroke length | 0–1 |
| `knee_min_accel_timing` | argmin of \|knee angular acceleration\| / stroke length | 0–1 |
| `body_angle_at_catch` | torso vs. seat-rail axis (SVD of hip path) at the catch frame | degrees |
| `leg_back_lag` | (argmax knee velocity − argmax hip velocity) / stroke length | signed 0–1 |
| `elbow_angle_range` | max − min of wrist–elbow–shoulder angle | degrees |

Angles are computed in the image plane from (X, Y) only. The model is
`MinMaxScaler → Ridge(alpha=1.0)`, trained on strokes whose label is the coach's
grade for the whole video (parsed from the filename). Ridge output is **unclamped**:
individual strokes can score below 0 or above 100; the UI clamps only for coloring,
never for display.

## 6. The `/predict` response — everything the front end consumes

Multipart POST, field name `video`. Success response (all arrays row-aligned):

```jsonc
{
  "strokes_detected": 4,
  "near_side": "right",                  // which side faces the camera
  "predicted_grade_mean": 76.35,
  "strokes": [                            // one entry per stroke
    {
      "stroke": 1,                        // 1-based
      "predicted_grade": 77.1,
      "features": {                       // the 6 model features, 4 dp
        "min_hip_angle": 41.2, "fastest_hip_velocity_timing": 0.3125,
        "knee_min_accel_timing": 0.5417, "body_angle_at_catch": 63.99,
        "leg_back_lag": -0.0833, "elbow_angle_range": 71.44
      }
    }
  ],
  "stroke_ranges": [[26, 86], [86, 171], ...],  // inclusive row indices per stroke

  "pose": {                               // drives the wireframe
    "fps": 60.0, "width": 1920, "height": 1080,
    "landmarks": ["nose", ..., "right_foot_index"],   // 33 names, fixed order
    "t":  [0.10, 0.13, ...],              // seconds per row, 3 dp
    "xy": [[x0, y0, x1, y1, ..., x32, y32], ...]      // 66 floats per row, 4 dp,
  },                                      // both axes normalized to [0, 1]

  "metrics": {                            // drives the Tab 1 readout tiles
    "hip_angle":   [...],                 // knee–hip–shoulder, degrees, 1 dp
    "knee_angle":  [...],                 // ankle–knee–hip
    "elbow_angle": [...],                 // wrist–elbow–shoulder
    "wrist_x":     [...],                 // wrist_X − ankle_X (handle travel), 4 dp
    "wrist_v":     [...],                 // np.gradient(wrist_x, t) — widths/s
    "wrist_a":     [...]                  // np.gradient(wrist_v, t) — widths/s²
  }
}
```

Alignment invariant: `len(pose.t) == len(pose.xy) == len(metrics.*)`, and every
index in `stroke_ranges` is a valid row. Row *i* of everything describes the same
instant `t[i]`. Velocity/acceleration use `np.gradient(_, t)` so the uneven time
spacing (dropped frames) is handled correctly.

Errors come back as `{"error": "..."}` with status 400 (no pose / no strokes /
bad upload) or 500 (unexpected exception). Upload cap is 200 MB
(`MAX_UPLOAD_BYTES` in `app.py`).

## 7. How the wireframe is drawn (static/app.js)

### Layering and coordinate mapping

`.pose-wrap` is `position: relative` with `<video>` at `width: 100%; height: auto`
— the element box therefore exactly equals the video content box (no `object-fit`
letterboxing). The `<canvas>` is absolutely positioned over it at 100%/100% with
`pointer-events: none` (clicks fall through to the video controls). Because both
boxes coincide and pose coordinates are normalized to `[0,1]`, mapping is a plain
multiply: `px = x * canvasWidth`, `py = y * canvasHeight`.

The canvas backing store is sized at `cssSize × devicePixelRatio` with
`ctx.setTransform(dpr, 0, 0, dpr, 0, 0)` so lines are crisp on retina screens.
A `ResizeObserver` on `.pose-wrap` re-sizes and redraws on any layout change.

### What gets drawn

`POSE_EDGES` lists 16 joint pairs *by landmark name* (shoulders, hips, torso
sides, both arms shoulder→elbow→wrist, both legs hip→knee→ankle→heel→toe). The 10
face landmarks are skipped; the nose is drawn as a ring for the head.
`resolvePoseIndices()` converts names to indices via `pose.landmarks` once per
response, so a server-side change to the joint list flows through automatically.

Each frame is drawn in two passes: a dark translucent under-stroke (keeps the
figure legible over light clothing), then the colored stroke. Small white dots mark
the joints. Line widths scale with `unit = max(w, h) / 640` so the figure reads the
same at any panel size.

### Video sync

The video's `currentTime` is the single clock. `frameAt(target)` finds the pose row
nearest the current time by walking a monotonic cursor over the sorted `pose.t`
array (playback advances in small steps, so this is O(1) per frame — no binary
search needed). While the video plays, a `requestAnimationFrame` loop redraws every
display frame; on `pause`/`ended` the loop stops; `seeked` and tab switches trigger
single redraws, so scrubbing while paused stays live.

If the nearest pose row is more than `POSE_MAX_GAP_S = 0.25` s away from the
playhead (a stretch where MediaPipe found no pose), the skeleton is *not* drawn and
the tiles show "—" rather than freezing on stale data.

**Codec fallback:** iPhone `.MOV` files are HEVC, which Firefox and Chrome-on-Linux
can't decode. The `<video>` `error` event switches the panel to a skeleton-only
mode: the video is hidden, the box takes the clip's aspect ratio
(`--pose-aspect: width / height`), and a `performance.now()`-based clock loops
`standaloneTime` over `pose.t` so the wireframe (and tiles/badge) animate without a
playing video.

### Per-tab wireframe differences

The same skeleton draws on every tab; only the coloring changes:

- **Frame analysis:** neutral gray skeleton, plus color-coded rings on the
  *near-side* joints (side from `near_side`): pink hip, blue knee, amber elbow, and
  three concentric rings on the wrist — teal (position), violet (velocity), orange
  (acceleration). Ring colors are the same CSS variables the tiles use, so tile ↔
  ring correspondence is automatic.
- **Stroke analysis:** the skeleton takes the grade color of the stroke currently
  under the playhead. `strokeOfRow` is a precomputed `Int16Array` mapping every row
  to its stroke index (−1 outside all strokes), filled once from `stroke_ranges`,
  so the per-frame lookup is O(1). The on-video badge ("Stroke N" over the score)
  updates from the same lookup. Rows outside any stroke draw neutral.
- **Session summary:** static lime accent.

### Tab 1 tiles

`updateForRow(i)` writes the six tile values from `metrics.*[i]` — but only when
the row index actually changes, so the rAF loop does no DOM work at 60 fps for a
30 fps-sampled pose. Formats: angles `°` 0 dp, wrist position 2 dp, velocity
`/s` 2 dp, acceleration `/s²` 1 dp (units are normalized frame-widths).

## 8. Grade → color mapping (Tab 2 bars, wireframe, badge, summary tiles)

One function (`gradeColor` in `app.js`) colors everything, so bars always match the
wireframe. Base scale: hue = `120 × clamp(grade, 0, 110) / 110` in HSL — 0 is red,
110 is green.

Because a uniformly decent video (all strokes 70–80) would render as
indistinguishable near-greens, the scale is **stretched per video**
(`setGradeScale`): the best stroke keeps its true absolute hue, and the worst
stroke's hue is lowered (pulled toward red) until the video's hues span at least
50°. Grades in between interpolate linearly. A video whose natural spread already
exceeds 50° is shown on the absolute scale unchanged, and a single-stroke video
falls back to absolute colors.

## 9. Tab 2 chart and Tab 3 stats

- Chart.js bar chart, one bar per stroke, `backgroundColor` from `gradeColor`.
  Left axis: score (suggested 0–110). The feature radios add a dashed neutral-white
  line dataset on a second right-hand axis (`y1`) titled with the feature's name and
  unit; dataset `order` is set so the line draws **on top of** the bars (Chart.js
  draws higher `order` first). Only one feature shows at a time; "None" removes it.
  The chart is built lazily on first visit to the tab because Chart.js cannot size
  a canvas inside a `hidden` panel.
- Tab 3 formulas (all client-side from the payload):
  - Stroke rate: `60 × n_strokes / (t[last_range_end] − t[first_range_start])`.
  - Best/worst: argmax/argmin of the per-stroke grades, value colored by grade.
  - Consistency: `clamp(100 − std(grades), 0, 100)` shown as a percent, with the
    raw `±std` in the caption.

## 10. Deployment

- `Dockerfile`: `python:3.12-slim-bookworm` + OpenCV/MediaPipe system libs; copies
  the five backend modules, `index.html`, `static/`, the pose model, and
  `artifacts/`; runs `gunicorn -b 0.0.0.0:$PORT -t 300 --workers 1 app:app`.
  The 300 s timeout exists because a 200 MB upload can take several minutes
  through MediaPipe; analysis time scales with video length, not resolution.
- Any new front-end file must be added to the Dockerfile `COPY` lines or it will
  silently not exist in the image.
- `hidden`-attribute gotcha (already handled, worth knowing): elements styled with
  an explicit `display: flex` need a `[hidden] { display: none }` override, or the
  attribute is ignored.
- Local dev runs on port 5000 (`python app.py`); Docker/Railway on 8080.
