# StrokeScore — rowing form analysis from a single video

Upload a side-view video of yourself on a rowing machine and get back a frame-by-frame
form breakdown: an animated wireframe drawn over your video, live joint angles, a
score for every stroke, and a whole-session summary.

Under the hood: [MediaPipe](https://developers.google.com/mediapipe) pose landmarks →
stroke segmentation → biomechanical features → a small scikit-learn model trained on
coach-graded videos. Designed for small datasets (hundreds to ~1000 strokes).

An internal deep-dive on how everything works lives in
[`details_technical.txt`](details_technical.txt).

## The web app

Three tabs, all synced to the uploaded video with a pose wireframe drawn on top:

- **Frame analysis** — color-coded rings on the hip, knee, elbow, and wrist, with live
  readout tiles for hip/knee/elbow angle, wrist position, velocity, and acceleration
  that update as the video plays.
- **Stroke analysis** — a bar per stroke, colored red→green by its predicted score
  (the spectrum stretches per video so similar strokes are still distinguishable);
  the wireframe and an on-video badge recolor as playback crosses stroke boundaries;
  radio buttons overlay any model feature on the score chart; big average score below.
- **Session summary** — stroke count, stroke rate, average score, best/worst stroke,
  and a consistency percentage.

The video plays from the user's local file — the upload is analyzed server-side and
deleted; only pose data comes back. Uploads up to 200 MB (larger files take
proportionally longer to analyze). If the browser can't decode the video codec
(e.g. iPhone HEVC `.MOV` on Linux), the wireframe animates on its own.

## Quick start (run the app)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python export_model.py      # once, after training data exists (see pipeline below)
python app.py               # open http://127.0.0.1:5000
```

The server loads `artifacts/model.joblib` at startup — it never retrains online.

### Docker / Railway

```bash
docker build -t strokescore .
docker run --rm -p 8080:8080 strokescore   # open http://localhost:8080
```

On Railway: connect the repo, make sure `artifacts/model.joblib` is in the build
context, and use the Dockerfile as the start method. The image runs gunicorn on
`$PORT` (default 8080) with a 300 s request timeout to accommodate large uploads.

## Training pipeline

Training videos go in `SampleVideos/`, with the coach's grade as digits before the
extension (`athlete_85.mp4` → grade 85). Pose extraction uses MediaPipe Tasks with
`pose_landmarker_lite.task` in the project root.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Pose extraction | `extraction.py` | `SampleVideos/*` | `all_videos_all_joints.csv` |
| 2. Stroke splitting | `split_strokes.py` | `all_videos_all_joints.csv` | `cycle_data.csv` |
| 3. Feature engineering | `feature_extraction.py` | `cycle_data.csv` | feature matrix (used by ML scripts) |
| 4. Model comparison | `model_compare.py` | `cycle_data.csv` | CV metrics leaderboard |
| 5. Hyperparameter sweeps | `hyperparam_tune.py` | `cycle_data.csv` | interactive CV sweeps |
| 6. Export production model | `export_model.py` | `cycle_data.csv` | `artifacts/model.joblib` |

```bash
python extraction.py
python split_strokes.py
python model_compare.py
python export_model.py
```

Optional diagnostics — plot the cycle-split signal for one video:

```bash
python plot_cycle_signals.py path/to/video.mp4
```

## Features

Per-stroke features (see `feature_extraction.py`):

- `min_hip_angle` — compression at the catch
- `fastest_hip_velocity_timing` — when the hip opens fastest
- `knee_min_accel_timing` — when knee angular acceleration crosses ~zero
- `body_angle_at_catch` — torso angle at the catch, i.e. forward reach
- `leg_back_lag` — sequencing: how far the legs peak ahead of the back swing
- `elbow_angle_range` — how far the arms draw

Timing features are normalized to stroke length (0–1); angle features stay in
degrees. `body_angle_at_catch` is measured against the erg's seat rail (principal
axis of hip motion), so it is unaffected by camera roll.

## Model selection

`model_compare.py` prints two leaderboards: shuffled 5-fold CV over strokes
(optimistic — the model has seen the same rower's other strokes) and
**leave-one-video-out** (honest — a rower the model has never seen, which is what
the app actually faces). Expect a large gap and a different ranking; trust the
second table. `video_mae` there compares the mean of a video's stroke predictions to
its grade — the same quantity the app reports.

`hyperparam_tune.py` is an interactive terminal tool for one-hyperparameter CV
sweeps over the same model set.

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Flask API: serves the UI, `/predict` runs the full pipeline |
| `index.html` | Page markup (tabs, tiles, chart, video panel) |
| `static/style.css` | Dark theme and responsive layout |
| `static/app.js` | Upload flow, wireframe overlay, tab logic, charts |
| `joint_info.py` | Joint names and CSV column lookup |
| `extraction.py` | Video → per-frame landmarks (+ timestamps for the overlay) |
| `split_strokes.py` | Peaks on wrist–ankle signal → stroke cycles |
| `feature_extraction.py` | Per-cycle angles and timing features |
| `model_compare.py` | Cross-validated model leaderboard |
| `hyperparam_tune.py` | Interactive one-hyperparameter CV sweeps |
| `export_model.py` | Fit + save production `Pipeline` to joblib |
| `plot_cycle_signals.py` | Debug plots of cycle-split peaks |
| `artifacts/` | `model.joblib`, `model_meta.json` |
| `Dockerfile` | Production image (gunicorn) |
| `pose_landmarker_lite.task` | MediaPipe pose model |
| `SampleVideos/` | Local training/demo videos (not shipped in the image) |
| `details_technical.txt` | Internal technical deep-dive |

## Notes

- Pose runs on every 2nd frame (`FRAME_MODULUS` in `extraction.py`); frames with no
  detected pose are dropped, so the app carries explicit per-row timestamps.
- Peak-finding parameters for cycle splits live in `split_strokes.py`
  (`PEAK_DISTANCE`, `PEAK_PROMINENCE`).
- `cycle_data.csv` is gitignored; regenerate it with the extraction + split steps.
- Keep train-time and deploy-time `scikit-learn` versions aligned when loading
  joblib artifacts.

## License

Add a license file if you publish this repository (e.g. MIT).
