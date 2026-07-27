"""Compare ML regressors on per-stroke features from cycle_data.csv."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import feature_extraction as fe
import joint_info
from split_strokes import faster_index_by_0_column

FEATURE_EXTRACTORS: list[fe.FeatureExtractor] = list(fe.DEFAULT_FEATURE_EXTRACTORS)

MODELS: dict[str, Pipeline] = {
    "dummy_mean": Pipeline([("model", DummyRegressor(strategy="mean"))]),
    "ridge": Pipeline([("scale", MinMaxScaler()), ("model", Ridge(alpha=1.0))]),
    "lasso": Pipeline([("scale", MinMaxScaler()), ("model", Lasso(alpha=0.1))]),
    "elasticnet": Pipeline([
        ("scale", MinMaxScaler()),
        ("model", ElasticNet(alpha=0.1, l1_ratio=0.5)),
    ]),
    "bayesian_ridge": Pipeline([
        ("scale", MinMaxScaler()),
        ("model", BayesianRidge()),
    ]),
    "huber": Pipeline([("scale", MinMaxScaler()), ("model", HuberRegressor())]),
    "svr_linear": Pipeline([
        ("scale", MinMaxScaler()),
        ("model", SVR(kernel="linear", C=1.0)),
    ]),
    "svr_rbf": Pipeline([
        ("scale", MinMaxScaler()),
        ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ]),
    "rf_shallow": Pipeline([
        ("scale", MinMaxScaler()),
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
    "extra_trees": Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=5,
                random_state=0,
            ),
        ),
    ]),
    "gbr_shallow": Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=50,
                max_depth=2,
                min_samples_leaf=5,
                random_state=0,
            ),
        ),
    ]),
}


def load_cycles_with_groups(
    path: str = "cycle_data.csv",
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Load ``cycle_data.csv`` as one array per stroke, plus each stroke's source video.

    The video index is the trailing column written by ``split_strokes.write_cycles_to_file``.
    It is stripped off here, so the returned cycles have the layout the feature extractors
    expect.
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.shape[1] != len(joint_info.headers) + 1:
        raise SystemExit(
            f"{path} has {data.shape[1]} columns; expected {len(joint_info.headers) + 1} "
            "(it predates the trailing video_index column). Re-run `python split_strokes.py`."
        )

    with_video = faster_index_by_0_column(data)
    cycles = [c[:, :-1] for c in with_video]
    groups = np.array([c[0, -1] for c in with_video], dtype=int)
    return cycles, groups


def load_cycles(path: str = "cycle_data.csv") -> list[np.ndarray]:
    """Load ``cycle_data.csv`` and return one array per stroke (cycle)."""
    return load_cycles_with_groups(path)[0]


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
    print(f"{'model':<18} {'mae':>8} {'±':>3} {'rmse':>8} {'±':>3} {'r2':>8} {'±':>3}")

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
            f"{name:<18} {mae:8.3f} {mae_std:5.3f} "
            f"{rmse:8.3f} {rmse_std:5.3f} {r2:8.3f} {r2_std:5.3f}"
        )


def print_lovo_leaderboard(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: dict[str, Pipeline],
) -> None:
    """
    Hold out one whole video at a time and print pooled MAE, RMSE, and R².

    A grade belongs to a video, not a stroke, and each video contributes many near-identical
    strokes. Shuffled K-fold therefore trains on a rower's other strokes and tests on the
    rest, which flatters any model that can recognise the rower. Holding out whole videos
    measures what the app actually does: score someone it has never seen.

    Predictions are pooled across folds before scoring rather than averaged per fold: every
    stroke in a held-out video shares one grade, so that fold's ``y`` has zero variance and a
    per-fold R² is degenerate.
    """
    n_videos = len(np.unique(groups))
    if n_videos < 2:
        print("\nleave-one-video-out: needs at least 2 videos; skipped.")
        return

    print(f"\nleave-one-video-out: each video held out entirely (videos={n_videos})")
    print("per-video = mean of that video's stroke predictions vs its grade\n")
    print(
        f"{'model':<18} {'stroke_mae':>11} {'video_mae':>11} "
        f"{'video_rmse':>11} {'video_r2':>10}"
    )

    video_grades = np.array([y[groups == g][0] for g in np.unique(groups)])
    rows: list[tuple[float, str, float, float, float]] = []
    for name, pipe in models.items():
        preds = cross_val_predict(
            pipe, x, y, cv=LeaveOneGroupOut(), groups=groups, n_jobs=-1
        )
        video_preds = np.array([preds[groups == g].mean() for g in np.unique(groups)])
        video_mae = float(mean_absolute_error(video_grades, video_preds))
        rows.append((
            video_mae,
            name,
            float(mean_absolute_error(y, preds)),
            float(root_mean_squared_error(video_grades, video_preds)),
            float(r2_score(video_grades, video_preds)),
        ))

    for video_mae, name, stroke_mae, video_rmse, video_r2 in sorted(rows):
        print(
            f"{name:<18} {stroke_mae:11.3f} {video_mae:11.3f} "
            f"{video_rmse:11.3f} {video_r2:10.3f}"
        )


if __name__ == "__main__":
    cycles, groups = load_cycles_with_groups()
    if not cycles:
        raise SystemExit("No cycles found. Run extraction.py and split_strokes.py first.")

    features, labels = build_xy(cycles, FEATURE_EXTRACTORS)
    print_leaderboard(features, labels, MODELS)
    print_lovo_leaderboard(features, labels, groups, MODELS)
