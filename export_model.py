"""Fit the production Pipeline offline and save it for the web app.

Run after cycle_data.csv exists:

    python export_model.py

Writes artifacts/model.joblib (and a small meta JSON). Redeploy after replacing
the joblib file when you pick a better model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import feature_extraction as fe
from model_compare import build_xy, load_cycles

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
META_PATH = ARTIFACTS_DIR / "model_meta.json"


def build_production_pipeline() -> Pipeline:
    """Default production model: Ridge + MinMax (same as model_compare ridge)."""
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("model", Ridge(alpha=1.0)), # this is the line to change when you've found the best model
    ])


def main() -> None:
    cycles = load_cycles(str(APP_DIR / "cycle_data.csv"))
    if not cycles:
        raise SystemExit("No cycles found. Run extraction.py and split_strokes.py first.")

    x, y = build_xy(cycles, fe.DEFAULT_FEATURE_EXTRACTORS)
    if x.shape[1] != len(fe.FEATURE_NAMES):
        raise SystemExit(
            f"Feature count mismatch: X has {x.shape[1]} cols, "
            f"FEATURE_NAMES has {len(fe.FEATURE_NAMES)}."
        )

    pipe = build_production_pipeline()
    pipe.fit(x, y)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "model": "ridge",
                "alpha": 1.0,
                "scaler": "MinMaxScaler",
                "n_samples": int(len(y)),
                "n_features": int(x.shape[1]),
                "feature_names": fe.FEATURE_NAMES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
    print(f"Wrote {META_PATH}")


if __name__ == "__main__":
    main()
