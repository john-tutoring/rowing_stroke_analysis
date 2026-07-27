"""
Local / Railway web app: upload a rowing video → stroke scores as JSON.

Loads a pre-fitted sklearn Pipeline from artifacts/model.joblib (no online training).

Local:
  python export_model.py   # once, after cycle_data.csv exists
  python app.py            # or: gunicorn -b 0.0.0.0:8080 -t 120 app:app

Docker / Railway: gunicorn is the process (see Dockerfile).
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.pipeline import Pipeline

import feature_extraction as fe
import joint_info
from extraction import TimingStats, VideoInfo, joints_from_video
from split_strokes import split_cycles

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", APP_DIR / "artifacts" / "model.joblib"))
# ~100 MB — plenty for a typical 2-minute phone video
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Run `python export_model.py` locally "
            "and ensure artifacts/model.joblib is in the image."
        )
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline in {path}, got {type(model)}")
    return model


# Loaded at import time so gunicorn workers have the model ready
model: Pipeline = load_model()


def pose_payload(
    frames: np.ndarray, frame_times: np.ndarray, video_info: VideoInfo
) -> dict:
    """Per-frame 2D landmarks for the browser to draw over the played-back video."""
    # Columns are [video_index, row_grade, x, y, z, ...]; keep x and y per joint.
    coords = frames[:, 2:].reshape(len(frames), joint_info.NUM_JOINTS, 3)[:, :, :2]
    return {
        "fps": round(video_info["fps"], 3),
        "width": video_info["width"],
        "height": video_info["height"],
        "landmarks": joint_info.joints,
        "t": np.round(frame_times, 3).tolist(),
        "xy": np.round(coords.reshape(len(frames), -1), 4).tolist(),
    }


def video_to_predictions(video_path: str) -> dict:
    """Pose → strokes → features → predicted grade per stroke."""
    timing: TimingStats = {"cv2": 0.0, "mediapipe": 0.0}
    frames, frame_times, video_info = joints_from_video(
        video_path, 0, timing, rowing_grade=0, return_timestamps=True
    )
    if frames.size == 0:
        return {"error": "No pose detected in video."}

    # Drop video index column; keep placeholder grade + joints (split code expects this layout)
    per_video = [frames[:, 1:]]
    cycles = split_cycles(per_video)
    if not cycles:
        return {"error": "No rowing strokes detected."}

    x = np.array(fe.features_from_cycles(cycles, fe.DEFAULT_FEATURE_EXTRACTORS), dtype=np.float64)
    if x.shape[1] != len(fe.FEATURE_NAMES):
        return {
            "error": (
                f"Feature count mismatch: got {x.shape[1]}, expected {len(fe.FEATURE_NAMES)}."
            ),
        }

    preds = model.predict(x)

    strokes = []
    for i, (pred, feat_row) in enumerate(zip(preds, x)):
        strokes.append({
            "stroke": i + 1,
            "predicted_grade": round(float(pred), 2),
            "features": {
                name: round(float(v), 4) for name, v in zip(fe.FEATURE_NAMES, feat_row)
            },
        })

    return {
        "strokes_detected": len(cycles),
        "predicted_grade_mean": round(float(np.mean(preds)), 2),
        "strokes": strokes,
        "pose": pose_payload(frames, frame_times, video_info),
    }


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded (field name must be 'video')."}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    UPLOAD_DIR.mkdir(exist_ok=True)
    safe_name = Path(f.filename).name
    dest = UPLOAD_DIR / safe_name
    f.save(dest)

    try:
        result = video_to_predictions(str(dest))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"Model: {MODEL_PATH}")
    print(f"Ready. Open http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
