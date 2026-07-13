"""Interactive terminal tool: pick a model, tune one hyperparameter, print CV metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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
        key="knn",
        menu_name="knn",
        blurb=(
            "k-Nearest Neighbors grades a stroke by looking at the most similar strokes "
            "it has already seen and averaging their grades. No fancy equation — just "
            "“what did strokes like this one usually score?”"
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
