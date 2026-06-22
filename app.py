+"""
Tiny local server: upload a rowing video, run the pipeline, return Ridge scores as JSON.

Run:  python app.py
Open: http://127.0.0.1:5000
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import feature_extraction as fe
from extraction import TimingStats, joints_from_video
from model_compare import build_xy, load_cycles
from split_strokes import split_cycles

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
# ~100 MB — plenty for a typical 2-minute phone video
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

FEATURE_NAMES = [
    "max_hip_angle",
    "min_hip_angle",
    "fastest_hip_accel_timing",
    "fastest_hip_velocity_timing",
    "fastest_elbow_accel_timing",
    "knee_min_accel_timing",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

ridge_model: Pipeline | None = None


def train_ridge() -> Pipeline:
    """Fit Ridge on cycle_data.csv (same setup as model_compare)."""
    cycles = load_cycles(str(APP_DIR / "cycle_data.csv"))
    if not cycles:
        raise RuntimeError("Need cycle_data.csv — run extraction.py and split_strokes.py first.")
    x, y = build_xy(cycles, fe.DEFAULT_FEATURE_EXTRACTORS)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe.fit(x, y)
    return pipe


def video_to_predictions(video_path: str) -> dict:
    """Pose → strokes → features → predicted grade per stroke."""
    timing: TimingStats = {"cv2": 0.0, "mediapipe": 0.0}
    frames = joints_from_video(video_path, 0, timing, rowing_grade=0)
    if frames.size == 0:
        return {"error": "No pose detected in video."}

    # Drop video index column; keep placeholder grade + joints (split code expects this layout)
    per_video = [frames[:, 1:]]
    cycles = split_cycles(per_video)
    if not cycles:
        return {"error": "No rowing strokes detected."}

    x = np.array(fe.features_from_cycles(cycles, fe.DEFAULT_FEATURE_EXTRACTORS), dtype=np.float64)
    preds = ridge_model.predict(x)

    strokes = []
    for i, (pred, feat_row) in enumerate(zip(preds, x)):
        strokes.append({
            "stroke": i + 1,
            "predicted_grade": round(float(pred), 2),
            "features": {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, feat_row)},
        })

    return {
        "strokes_detected": len(cycles),
        "predicted_grade_mean": round(float(np.mean(preds)), 2),
        "strokes": strokes,
    }


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


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

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


if __name__ == "__main__":
    print("Training Ridge on cycle_data.csv ...")
    ridge_model = train_ridge()
    print("Ready. Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
