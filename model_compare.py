"""Compare ML regressors on per-stroke features from cycle_data.csv."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import feature_extraction as fe
from split_strokes import faster_index_by_0_column

FEATURE_EXTRACTORS: list[fe.FeatureExtractor] = list(fe.DEFAULT_FEATURE_EXTRACTORS)

MODELS: dict[str, Pipeline] = {
    "dummy_mean": Pipeline([("model", DummyRegressor(strategy="mean"))]),
    "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "svr_linear": Pipeline([
        ("scale", StandardScaler()),
        ("model", SVR(kernel="linear", C=1.0)),
    ]),
    "svr_rbf": Pipeline([
        ("scale", StandardScaler()),
        ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ]),
    "knn_5": Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5)),
    ]),
    "rf_shallow": Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=5,
                random_state=0,
            ),
        ),
    ]),
}


def load_cycles(path: str = "cycle_data.csv") -> list[np.ndarray]:
    """Load ``cycle_data.csv`` and return one array per stroke (cycle)."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return faster_index_by_0_column(data)


def build_xy(
    cycles: list[np.ndarray],
    feature_extractors: list[fe.FeatureExtractor],
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix ``x`` and grade labels ``y`` (``row_grade`` in column 0 of each cycle)."""
    x = np.array(fe.features_from_cycles(cycles, feature_extractors), dtype=np.float64)
    y = np.array([c[0, 0] for c in cycles], dtype=np.float64)
    return x, y


def print_leaderboard(
    x: np.ndarray,
    y: np.ndarray,
    models: dict[str, Pipeline],
    n_splits: int = 5,
    random_state: int = 0,
) -> None:
    """Run shuffled K-fold CV and print MAE, RMSE, and R² for each model."""
    n_splits = min(n_splits, len(y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    print(f"samples={len(y)} features={x.shape[1]} kfold_splits={n_splits}\n")
    print(f"{'model':<16} {'mae':>8} {'±':>3} {'rmse':>8} {'±':>3} {'r2':>8} {'±':>3}")

    rows: list[tuple[float, str, float, float, float, float, float, float]] = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, x, y, cv=cv, scoring=scoring, n_jobs=-1)
        mae = float(-scores["test_mae"].mean())
        rows.append((
            mae,
            name,
            mae,
            float(scores["test_mae"].std()),
            float(-scores["test_rmse"].mean()),
            float(scores["test_rmse"].std()),
            float(scores["test_r2"].mean()),
            float(scores["test_r2"].std()),
        ))

    for row in sorted(rows):
        _, name, mae, mae_std, rmse, rmse_std, r2, r2_std = row
        print(
            f"{name:<16} {mae:8.3f} {mae_std:5.3f} "
            f"{rmse:8.3f} {rmse_std:5.3f} {r2:8.3f} {r2_std:5.3f}"
        )


if __name__ == "__main__":
    cycles = load_cycles()
    if not cycles:
        raise SystemExit("No cycles found. Run extraction.py and split_strokes.py first.")

    features, labels = build_xy(cycles, FEATURE_EXTRACTORS)
    print_leaderboard(features, labels, MODELS)
