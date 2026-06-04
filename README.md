# Rowing pose analysis

Extract pose landmarks from rowing videos with [MediaPipe](https://developers.google.com/mediapipe), segment individual stroke cycles, engineer biomechanical features, and compare simple machine-learning models to predict **row grade** (parsed from each video filename).

Designed for small datasets (on the order of hundreds to ~1000 strokes).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place rowing videos in `SampleVideos/`. Filenames should include the grade/score as digits before the extension (e.g. `athlete_85.mp4` → grade `85`), matching the regex in `extraction.py`.

MediaPipe may download model assets on first run. For pose extraction you may also need `pose_landmarker_lite.task` in the project root if you switch to the Tasks API path in `extraction.py`.

## Pipeline

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Pose extraction | `extraction.py` | `SampleVideos/*` | `all_videos_all_joints.csv` |
| 2. Stroke splitting | `split_strokes.py` | `all_videos_all_joints.csv` | `cycle_data.csv` |
| 3. Feature engineering | `feature_extraction.py` | `cycle_data.csv` | (prints feature matrix; optional) |
| 4. Model comparison | `model_compare.py` | joint or cycle CSV | CV metrics table |

```bash
python extraction.py
python split_strokes.py
python model_compare.py
```

## Model comparison

`model_compare.py` loads `cycle_data.csv` and prints cross-validated scores for several sklearn models.

```bash
python model_compare.py
```

Models included: dummy baseline, Ridge, linear/RBF SVR (or SVC), k-NN, shallow random forest. Features are scaled inside each CV fold (no global whitening leakage).

## Project layout

| File | Purpose |
|------|---------|
| `joint_info.py` | Joint names and CSV column lookup |
| `extraction.py` | Video → per-frame landmarks |
| `split_strokes.py` | Peaks on wrist–ankle signal → stroke cycles |
| `feature_extraction.py` | Per-cycle angles and timing features |
| `model_compare.py` | Cross-validated model leaderboard |

## Notes

- Frame sampling uses every 2nd frame (`FRAME_MODULUS` in `extraction.py`).
- Peak-finding parameters for cycle splits live in `split_strokes.py` (`distance`, `prominence`).
## License

Add a license file if you publish this repository (e.g. MIT).
