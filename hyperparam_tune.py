"""Interactive terminal tool: pick a model, tune one hyperparameter, print CV metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import feature_extraction as fe
from model_compare import build_xy, load_cycles

FEATURE_EXTRACTORS: list[fe.FeatureExtractor] = list(fe.DEFAULT_FEATURE_EXTRACTORS)

N_SWEEP = 10
N_SPLITS = 5
RANDOM_STATE = 0


@dataclass(frozen=True)
class HyperparamSpec:
    name: str
    description: str
    default: float | int
    values: Callable[[], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    menu_name: str
    blurb: str
    hyperparams: list[HyperparamSpec]
    build: Callable[[dict[str, Any]], Pipeline]


def _linspace_around(center: float, half_span: float, n: int = N_SWEEP) -> np.ndarray:
    """``n`` evenly spaced floats centered on ``center`` (endpoints = center ± half_span)."""
    return np.linspace(center - half_span, center + half_span, n)


def _int_range_include_center(center: int, low: int, high: int, n: int = N_SWEEP) -> np.ndarray:
    """``n`` distinct integers from ``low``..``high``, always including ``center``."""
    span = high - low + 1
    if span <= n:
        return np.arange(low, high + 1, dtype=int)

    # Evenly spaced ints, then force the default into the set
    vals = np.unique(np.round(np.linspace(low, high, n)).astype(int))
    if center not in vals:
        idx = int(np.argmin(np.abs(vals.astype(float) - center)))
        vals = vals.copy()
        vals[idx] = center
        vals = np.unique(vals)
    # Refill if unique/replace collapsed the length
    while len(vals) < n:
        for candidate in range(low, high + 1):
            if candidate not in vals:
                vals = np.sort(np.append(vals, candidate))
                break
        else:
            break
    return vals[:n].astype(int)


def _ridge(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("model", Ridge(alpha=float(params.get("alpha", 1.0)))),
    ])


def _lasso(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("model", Lasso(alpha=float(params.get("alpha", 0.1)))),
    ])


def _elasticnet(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            ElasticNet(
                alpha=float(params.get("alpha", 0.1)),
                l1_ratio=float(params.get("l1_ratio", 0.5)),
            ),
        ),
    ])


def _bayesian_ridge(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            BayesianRidge(
                alpha_1=float(params.get("alpha_1", 1e-6)),
                lambda_1=float(params.get("lambda_1", 1e-6)),
            ),
        ),
    ])


def _huber(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            HuberRegressor(
                epsilon=float(params.get("epsilon", 1.35)),
                alpha=float(params.get("alpha", 0.0001)),
            ),
        ),
    ])


def _svr_linear(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("model", SVR(kernel="linear", C=float(params.get("C", 1.0)))),
    ])


def _svr_rbf(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            SVR(
                kernel="rbf",
                C=float(params.get("C", 1.0)),
                epsilon=float(params.get("epsilon", 0.1)),
            ),
        ),
    ])


def _knn(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("model", KNeighborsRegressor(n_neighbors=int(params.get("n_neighbors", 5)))),
    ])


def _rf(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=int(params.get("max_depth", 4)),
                min_samples_leaf=int(params.get("min_samples_leaf", 5)),
                random_state=0,
            ),
        ),
    ])


def _extra_trees(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=int(params.get("max_depth", 4)),
                min_samples_leaf=int(params.get("min_samples_leaf", 5)),
                random_state=0,
            ),
        ),
    ])


def _gbr(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=int(params.get("n_estimators", 50)),
                max_depth=int(params.get("max_depth", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 5)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                random_state=0,
            ),
        ),
    ])


MODELS: list[ModelSpec] = [
    ModelSpec(
        key="ridge",
        menu_name="ridge",
        blurb=(
            "Ridge is like drawing a straight-line relationship between the stroke "
            "measurements and the rowing grade. It gently holds the line back so it "
            "does not twist itself into a weird shape just to fit a few odd strokes. "
            "A good starting point when you want something simple and steady."
        ),
        hyperparams=[
            HyperparamSpec(
                name="alpha",
                description=(
                    "How strongly to hold the line back. Higher = smoother / more cautious; "
                    "lower = freer to fit the training strokes closely."
                ),
                default=1.0,
                values=lambda: _linspace_around(1.0, 0.9),
            ),
        ],
        build=_ridge,
    ),
    ModelSpec(
        key="lasso",
        menu_name="lasso",
        blurb=(
            "Lasso is also a straight-line model, but it can push some feature weights "
            "all the way to zero. That means it may ignore measurements that do not help "
            "much, which can keep things simpler when only a few signals matter."
        ),
        hyperparams=[
            HyperparamSpec(
                name="alpha",
                description=(
                    "How strongly to shrink (and zero out) feature weights. Higher = "
                    "more features ignored / simpler model; lower = freer fit."
                ),
                default=0.1,
                values=lambda: _linspace_around(0.1, 0.09),
            ),
        ],
        build=_lasso,
    ),
    ModelSpec(
        key="elasticnet",
        menu_name="elasticnet",
        blurb=(
            "Elastic Net mixes Ridge and Lasso: it both shrinks weights and can drop "
            "weak features. Useful when features are related and you want a middle "
            "ground between “keep everything a little” and “drop some entirely.”"
        ),
        hyperparams=[
            HyperparamSpec(
                name="alpha",
                description=(
                    "Overall strength of the hold-back. Higher = smoother / more cautious; "
                    "lower = freer to fit the training strokes."
                ),
                default=0.1,
                values=lambda: _linspace_around(0.1, 0.09),
            ),
            HyperparamSpec(
                name="l1_ratio",
                description=(
                    "Mix between Lasso-style dropping (closer to 1) and Ridge-style "
                    "gentle shrinking (closer to 0)."
                ),
                default=0.5,
                values=lambda: _linspace_around(0.5, 0.4),
            ),
        ],
        build=_elasticnet,
    ),
    ModelSpec(
        key="bayesian_ridge",
        menu_name="bayesian_ridge",
        blurb=(
            "Bayesian Ridge is another straight-line model that chooses how hard to "
            "hold itself back automatically from the data. The two knobs below gently "
            "nudge the prior beliefs that guide that automatic choice."
        ),
        hyperparams=[
            HyperparamSpec(
                name="alpha_1",
                description=(
                    "Prior shape for noise precision. Larger tends to assume less noise "
                    "up front; smaller is more open-minded about noisy grades."
                ),
                default=1e-6,
                values=lambda: np.logspace(-8, -4, N_SWEEP),
            ),
            HyperparamSpec(
                name="lambda_1",
                description=(
                    "Prior shape for weight precision. Larger tends toward stronger "
                    "shrinkage; smaller lets weights roam more freely."
                ),
                default=1e-6,
                values=lambda: np.logspace(-8, -4, N_SWEEP),
            ),
        ],
        build=_bayesian_ridge,
    ),
    ModelSpec(
        key="huber",
        menu_name="huber",
        blurb=(
            "Huber is a straight-line model that stays calm when a few strokes are "
            "wild outliers. It fits most points carefully, but does not let one bad "
            "stroke yank the whole line off course."
        ),
        hyperparams=[
            HyperparamSpec(
                name="epsilon",
                description=(
                    "How far off a stroke can be before it is treated as an outlier. "
                    "Larger = more tolerant; smaller = fussier about large mistakes."
                ),
                default=1.35,
                # sklearn requires epsilon >= 1.0
                values=lambda: np.linspace(1.0, 2.0, N_SWEEP),
            ),
            HyperparamSpec(
                name="alpha",
                description=(
                    "How strongly to hold the line back (L2 regularization). Higher = "
                    "smoother / more cautious; lower = freer fit."
                ),
                default=0.0001,
                values=lambda: np.logspace(-6, -2, N_SWEEP),
            ),
        ],
        build=_huber,
    ),
    ModelSpec(
        key="svr_linear",
        menu_name="svr_linear",
        blurb=(
            "Linear SVR also looks for a straight-line link between measurements and grade, "
            "but it mostly cares about getting most strokes close enough and is less bothered "
            "by a few outliers. Think of it as a sturdy straight ruler that ignores the "
            "occasional wild point."
        ),
        hyperparams=[
            HyperparamSpec(
                name="C",
                description=(
                    "How hard the model tries to fit every stroke. Higher = stick closer to "
                    "the data; lower = allow more wiggle room / smoother answers."
                ),
                default=1.0,
                values=lambda: _linspace_around(1.0, 0.9),
            ),
        ],
        build=_svr_linear,
    ),
    ModelSpec(
        key="svr_rbf",
        menu_name="svr_rbf",
        blurb=(
            "RBF SVR can bend and curve instead of staying on a straight line. It learns "
            "smoother, wiggly patterns in the stroke features, which can capture more "
            "nuance — but it can also overreact if you push it too hard."
        ),
        hyperparams=[
            HyperparamSpec(
                name="C",
                description=(
                    "How hard the model tries to fit every stroke. Higher = stick closer to "
                    "the data; lower = allow more wiggle room / smoother answers."
                ),
                default=1.0,
                values=lambda: _linspace_around(1.0, 0.9),
            ),
            HyperparamSpec(
                name="epsilon",
                description=(
                    "How far off a prediction can be before the model starts caring. "
                    "Larger = more tolerant of small mistakes; smaller = fussier about accuracy."
                ),
                default=0.1,
                values=lambda: _linspace_around(0.1, 0.09),
            ),
        ],
        build=_svr_rbf,
    ),
    ModelSpec(
        key="knn_3",
        menu_name="knn_3",
        blurb=(
            "k-Nearest Neighbors grades a stroke by looking at the most similar strokes "
            "it has already seen and averaging their grades. This entry starts with a "
            "small neighborhood (3) so predictions stay local and sensitive."
        ),
        hyperparams=[
            HyperparamSpec(
                name="n_neighbors",
                description=(
                    "How many similar strokes to average. Smaller = more local / sensitive; "
                    "larger = smoother average over more neighbors."
                ),
                default=3,
                values=lambda: _int_range_include_center(3, 1, 10),
            ),
        ],
        build=_knn,
    ),
    ModelSpec(
        key="knn",
        menu_name="knn_5",
        blurb=(
            "k-Nearest Neighbors grades a stroke by looking at the most similar strokes "
            "it has already seen and averaging their grades. No fancy equation — just "
            "“what did strokes like this one usually score?” Default neighborhood is 5."
        ),
        hyperparams=[
            HyperparamSpec(
                name="n_neighbors",
                description=(
                    "How many similar strokes to average. Smaller = more local / sensitive; "
                    "larger = smoother average over more neighbors."
                ),
                default=5,
                values=lambda: _int_range_include_center(5, 1, 10),
            ),
        ],
        build=_knn,
    ),
    ModelSpec(
        key="rf_shallow",
        menu_name="rf_shallow",
        blurb=(
            "A random forest asks many small decision trees for an opinion and averages "
            "them. Each tree looks at slightly different pieces of the data, so together "
            "they are often more reliable than any single tree alone."
        ),
        hyperparams=[
            HyperparamSpec(
                name="n_estimators",
                description=(
                    "How many trees to grow. More trees usually means a steadier average, "
                    "with diminishing returns as the count gets large."
                ),
                default=100,
                # 20..200 step 20 → ten values with 100 in the middle
                values=lambda: np.arange(20, 220, 20, dtype=int),
            ),
            HyperparamSpec(
                name="max_depth",
                description=(
                    "How deep each tree may grow. Deeper = more detailed rules; "
                    "shallower = simpler, less likely to memorize quirks."
                ),
                default=4,
                values=lambda: _int_range_include_center(4, 1, 10),
            ),
            HyperparamSpec(
                name="min_samples_leaf",
                description=(
                    "Smallest group of strokes allowed at the end of a tree branch. "
                    "Larger = chunkier / safer groups; smaller = finer splits."
                ),
                default=5,
                values=lambda: _int_range_include_center(5, 1, 10),
            ),
        ],
        build=_rf,
    ),
    ModelSpec(
        key="extra_trees",
        menu_name="extra_trees",
        blurb=(
            "Extra Trees is like a random forest, but the trees pick split points more "
            "randomly. That extra randomness can reduce overfitting and sometimes gives "
            "a steadier average on small datasets."
        ),
        hyperparams=[
            HyperparamSpec(
                name="n_estimators",
                description=(
                    "How many trees to grow. More trees usually means a steadier average, "
                    "with diminishing returns as the count gets large."
                ),
                default=100,
                values=lambda: np.arange(20, 220, 20, dtype=int),
            ),
            HyperparamSpec(
                name="max_depth",
                description=(
                    "How deep each tree may grow. Deeper = more detailed rules; "
                    "shallower = simpler, less likely to memorize quirks."
                ),
                default=4,
                values=lambda: _int_range_include_center(4, 1, 10),
            ),
            HyperparamSpec(
                name="min_samples_leaf",
                description=(
                    "Smallest group of strokes allowed at the end of a tree branch. "
                    "Larger = chunkier / safer groups; smaller = finer splits."
                ),
                default=5,
                values=lambda: _int_range_include_center(5, 1, 10),
            ),
        ],
        build=_extra_trees,
    ),
    ModelSpec(
        key="gbr_shallow",
        menu_name="gbr_shallow",
        blurb=(
            "Gradient boosting builds trees one after another, each trying to fix the "
            "mistakes of the previous ones. Kept shallow here so it does not memorize "
            "every quirk in a small set of strokes."
        ),
        hyperparams=[
            HyperparamSpec(
                name="n_estimators",
                description=(
                    "How many boosting rounds (trees) to add. More can improve fit, but "
                    "too many may overfit on a small dataset."
                ),
                default=50,
                values=lambda: np.arange(10, 110, 10, dtype=int),
            ),
            HyperparamSpec(
                name="max_depth",
                description=(
                    "How deep each tree may grow. Deeper = more detailed corrections; "
                    "shallower = safer on small data."
                ),
                default=2,
                values=lambda: _int_range_include_center(2, 1, 5),
            ),
            HyperparamSpec(
                name="min_samples_leaf",
                description=(
                    "Smallest group of strokes allowed at the end of a tree branch. "
                    "Larger = chunkier / safer groups; smaller = finer splits."
                ),
                default=5,
                values=lambda: _int_range_include_center(5, 1, 10),
            ),
            HyperparamSpec(
                name="learning_rate",
                description=(
                    "How big a step each new tree takes. Smaller = slower, steadier "
                    "learning; larger = faster but easier to overshoot."
                ),
                default=0.1,
                values=lambda: _linspace_around(0.1, 0.09),
            ),
        ],
        build=_gbr,
    ),
]


def _prompt_choice(prompt: str, n_options: int) -> int:
    """Ask for an integer in 1..n_options; re-prompt until valid."""
    while True:
        raw = input(prompt).strip()
        try:
            choice = int(raw)
        except ValueError:
            print(f"  Please enter a number from 1 to {n_options}.")
            continue
        if 1 <= choice <= n_options:
            return choice
        print(f"  Please enter a number from 1 to {n_options}.")


def _default_params(model: ModelSpec) -> dict[str, Any]:
    return {hp.name: hp.default for hp in model.hyperparams}


def _evaluate(
    pipe: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    n_splits = min(n_splits, len(y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    scores = cross_validate(pipe, x, y, cv=cv, scoring=scoring, n_jobs=-1)
    mae = float(-scores["test_mae"].mean())
    rmse = float(-scores["test_rmse"].mean())
    r2 = float(scores["test_r2"].mean())
    return mae, rmse, r2


def _sweep(
    model: ModelSpec,
    hp: HyperparamSpec,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    values = hp.values()
    if len(values) != N_SWEEP:
        # Guard: int helpers may theoretically under-produce; pad/truncate to 10
        if len(values) > N_SWEEP:
            values = values[:N_SWEEP]
        else:
            raise RuntimeError(f"Expected {N_SWEEP} sweep values for {hp.name}, got {len(values)}")

    base = _default_params(model)

    print(f"\nSweeping {hp.name} over {N_SWEEP} values (default={hp.default})…\n")
    print(f"{hp.name:>14}  {'mae':>8}  {'rmse':>8}  {'r2':>8}")
    print("-" * 44)

    for v in values:
        params = dict(base)
        params[hp.name] = int(v) if isinstance(hp.default, int) else float(v)
        pipe = model.build(params)
        mae, rmse, r2 = _evaluate(pipe, x, y)
        label = f"{params[hp.name]:g}" if isinstance(params[hp.name], float) else str(params[hp.name])
        print(f"{label:>14}  {mae:8.3f}  {rmse:8.3f}  {r2:8.3f}")


def main() -> None:
    cycles = load_cycles()
    if not cycles:
        raise SystemExit("No cycles found. Run extraction.py and split_strokes.py first.")

    x, y = build_xy(cycles, FEATURE_EXTRACTORS)
    print(f"Loaded {len(y)} strokes, {x.shape[1]} features.\n")

    print("Models:")
    for i, m in enumerate(MODELS, start=1):
        print(f"  {i}. {m.menu_name}")
    print()
    model_idx = _prompt_choice("Choose a model number: ", len(MODELS))
    model = MODELS[model_idx - 1]

    print(f"\n=== {model.menu_name} ===\n")
    print(model.blurb)
    print()

    print("Common hyperparameters:")
    for i, hp in enumerate(model.hyperparams, start=1):
        print(f"  {i}. {hp.name}")
        print(f"     {hp.description}")
    print()
    hp_idx = _prompt_choice("Which hyperparameter to tune? ", len(model.hyperparams))
    hp = model.hyperparams[hp_idx - 1]

    _sweep(model, hp, x, y)


if __name__ == "__main__":
    main()
