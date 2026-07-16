# Rowing pose analysis

Extract pose landmarks from rowing videos with [MediaPipe](https://developers.google.com/mediapipe), segment individual stroke cycles, engineer biomechanical features, and train simple sklearn models to predict **row grade** (parsed from each video filename).

Designed for small datasets (on the order of hundreds to ~1000 strokes). Includes a local/Flask web UI and a Docker setup suitable for Railway-style PaaS deploys.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place rowing videos in `SampleVideos/`. Filenames should include the grade/score as digits before the extension (e.g. `athlete_85.mp4` → grade `85`), matching the regex in `extraction.py`.

Pose extraction uses MediaPipe Tasks with `pose_landmarker_lite.task` in the project root.

## Pipeline

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Pose extraction | `extraction.py` | `SampleVideos/*` | `all_videos_all_joints.csv` |
| 2. Stroke splitting | `split_strokes.py` | `all_videos_all_joints.csv` | `cycle_data.csv` |
| 3. Feature engineering | `feature_extraction.py` | `cycle_data.csv` | (feature matrix; used by ML scripts) |
| 4. Model comparison | `model_compare.py` | `cycle_data.csv` | CV metrics leaderboard |
| 5. Hyperparameter sweeps | `hyperparam_tune.py` | `cycle_data.csv` | interactive CV sweeps |
| 6. Export production model | `export_model.py` | `cycle_data.csv` | `artifacts/model.joblib` |
| 7. Web app | `app.py` | video upload + joblib | JSON stroke grades |

```bash
python extraction.py
python split_strokes.py
python model_compare.py
```

Optional diagnostics — plot the cycle-split signal for one video:

```bash
python plot_cycle_signals.py path/to/video.mp4
```

## Features

Per-stroke features (see `feature_extraction.py`) currently include:

- `max_hip_angle`, `min_hip_angle`
- `fastest_hip_accel_timing`, `fastest_hip_velocity_timing`
- `knee_min_accel_timing`

Timing features are normalized to stroke length (0–1). Angle features stay in degrees. Model pipelines scale columns with `MinMaxScaler` inside each CV fold / fit.

## Model comparison

`model_compare.py` loads `cycle_data.csv` and prints shuffled 5-fold CV scores (MAE, RMSE, R²) for:

| Family | Models |
|--------|--------|
| Baseline | `dummy_mean` |
| Linear | `ridge`, `lasso`, `elasticnet`, `bayesian_ridge`, `huber` |
| Kernel / neighbors | `svr_linear`, `svr_rbf`, `knn_3`, `knn_5` |
| Trees | `rf_shallow`, `extra_trees`, `gbr_shallow` |

```bash
python model_compare.py
```

## Hyperparameter tuning

Interactive terminal tool — pick a model, pick one hyperparameter, sweep ~10 values, print CV metrics:

```bash
python hyperparam_tune.py
```

Menus and defaults mirror the models in `model_compare.py` (plain-language blurbs included).

## Export a trained model

After you have `cycle_data.csv` and a preferred model definition:

```bash
python export_model.py
```

Writes:

- `artifacts/model.joblib` — fitted sklearn `Pipeline` (scaler + regressor)
- `artifacts/model_meta.json` — model name, feature list, sample counts

Edit `build_production_pipeline()` in `export_model.py` when you switch the production model (defaults to Ridge + `MinMaxScaler`). Redeploy after replacing the joblib artifact.

## Local web app

```bash
python export_model.py   # once
python app.py            # or: gunicorn -b 0.0.0.0:8080 -t 120 app:app
```

Open http://127.0.0.1:5000 (or your gunicorn port). Upload a video via `index.html`; the app returns JSON with stroke count, per-stroke predicted grades, and features. Uploads land in `uploads/`.

The server **loads** `artifacts/model.joblib` at startup — it does not retrain online.

## Docker / Railway

`Dockerfile` builds a slim Python 3.12 image with OpenCV/MediaPipe system libs, copies the app modules + pose landmarker + `artifacts/model.joblib`, and runs gunicorn on `$PORT` (default 8080).

```bash
docker build -t rowing-pose .
docker run -p 8080:8080 rowing-pose
```

On Railway: connect the repo, ensure `artifacts/model.joblib` is present in the build context, and use the Dockerfile as the start method.

## Project layout

| Path | Purpose |
|------|---------|
| `joint_info.py` | Joint names and CSV column lookup |
| `extraction.py` | Video → per-frame landmarks |
| `split_strokes.py` | Peaks on wrist–ankle signal → stroke cycles |
| `feature_extraction.py` | Per-cycle angles and timing features |
| `model_compare.py` | Cross-validated model leaderboard |
| `hyperparam_tune.py` | Interactive one-hyperparameter CV sweeps |
| `export_model.py` | Fit + save production `Pipeline` to joblib |
| `plot_cycle_signals.py` | Debug plots of cycle-split peaks |
| `app.py` | Flask API + static UI host |
| `index.html` | Upload UI for local/deployed app |
| `artifacts/` | `model.joblib`, `model_meta.json` |
| `Dockerfile` | Production image (gunicorn) |
| `requirements.txt` | Python dependencies |
| `pose_landmarker_lite.task` | MediaPipe pose model (project root) |
| `SampleVideos/` | Local training/demo videos (not required in Docker image) |

## Notes

- Frame sampling uses every 2nd frame (`FRAME_MODULUS` in `extraction.py`).
- Peak-finding parameters for cycle splits live in `split_strokes.py` / `plot_cycle_signals.py` (`distance`, `prominence`).
- `cycle_data.csv` is gitignored; regenerate it with the extraction + split steps.
- Keep train-time and deploy-time `scikit-learn` versions aligned when loading joblib artifacts.

## License

Add a license file if you publish this repository (e.g. MIT).
