"""Per-stroke biomechanical features from cycle joint trajectories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np

import joint_info
from split_strokes import faster_index_by_0_column, right_side_closer

CycleArray: TypeAlias = np.ndarray
FeatureExtractor: TypeAlias = Callable[[CycleArray], float]


def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Return the angle at ``b`` formed by segments ``b→a`` and ``b→c``, in degrees.

    Dimension-agnostic; callers pass image-plane (X, Y) points.
    """
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def get_indexes(body_part: str, right_hand: bool) -> list[int]:
    """
    Column indices into a cycle array for ``body_part`` X/Y/Z.

    Indices assume the leading index column was already removed.
    """
    handed = "right" if right_hand else "left"
    dims = ("X", "Y", "Z")
    return [
        joint_info.COL_LOOKUP[f"{handed}_{body_part}_{xyz}"] - 1
        for xyz in dims
    ]


def get_angle_vector(
    cycle_data: CycleArray,
    joint1: str,
    joint2: str,
    joint3: str,
    right_hand: bool | None = None,
) -> np.ndarray:
    """
    Per-frame angles at ``joint2`` (vertex) for joints ``joint1`` and ``joint3``.

    Measured in the image plane from (X, Y) only. The rowers are shot in profile, so the
    joints of interest lie in that plane and MediaPipe's z — its least reliable channel,
    worst on the occluded far side — only added noise.

    Returns one angle in degrees per frame. ``right_hand`` defaults to whichever side
    ``right_side_closer`` finds nearer the camera, so features always read the visible
    limb rather than the occluded one; pass True/False to force a side.
    """
    if right_hand is None:
        right_hand = right_side_closer(cycle_data)

    j1_indexes = get_indexes(joint1, right_hand)
    j2_indexes = get_indexes(joint2, right_hand)
    j3_indexes = get_indexes(joint3, right_hand)

    j1 = cycle_data[:, j1_indexes[0] : j1_indexes[1] + 1]
    j2 = cycle_data[:, j2_indexes[0] : j2_indexes[1] + 1]
    j3 = cycle_data[:, j3_indexes[0] : j3_indexes[1] + 1]

    angles = [angle_abc(j1[i], j2[i], j3[i]) for i in range(len(j2))]
    return np.array(angles, dtype=np.float64)


def get_joint_xy(
    cycle_data: CycleArray,
    body_part: str,
    right_hand: bool | None = None,
) -> np.ndarray:
    """
    Per-frame image-plane (X, Y) coordinates for ``body_part``.

    ``right_hand`` defaults to the side nearer the camera, matching ``get_angle_vector``.
    """
    if right_hand is None:
        right_hand = right_side_closer(cycle_data)

    indexes = get_indexes(body_part, right_hand)
    return cycle_data[:, indexes[0] : indexes[1] + 1]


def slide_axis(cycle_data: CycleArray) -> np.ndarray:
    """
    Unit vector along the erg's seat rail, recovered from the hip's path of motion.

    The rail is horizontal in the world, so this is a reference direction that survives
    camera roll — unlike the image axes.
    """
    hip = get_joint_xy(cycle_data, "hip")
    centered = hip - hip.mean(axis=0)
    return np.linalg.svd(centered, full_matrices=False)[2][0]


def catch_index(cycle_data: CycleArray) -> int:
    """Frame index of the catch: the most compressed point of the stroke."""
    return int(np.argmin(get_angle_vector(cycle_data, "ankle", "knee", "hip")))


def features_from_cycles(
    per_cycle_data: Sequence[CycleArray],
    feature_extractors: Sequence[FeatureExtractor],
) -> list[list[float]]:
    """Apply each extractor to every cycle; return one feature row per cycle."""
    all_features: list[list[float]] = []
    for cycle in per_cycle_data:
        features = [float(extractor(cycle)) for extractor in feature_extractors]
        all_features.append(features)
    return all_features


def max_hip_angle(one_cycle_data: CycleArray) -> float:
    """Maximum knee–hip–shoulder angle (degrees) during the stroke."""
    hip_angles = get_angle_vector(one_cycle_data, "knee", "hip", "shoulder")
    return float(max(hip_angles))


def min_hip_angle(one_cycle_data: CycleArray) -> float:
    """Minimum knee–hip–shoulder angle (degrees) during the stroke."""
    hip_angles = get_angle_vector(one_cycle_data, "knee", "hip", "shoulder")
    return float(min(hip_angles))


def fastest_hip_accel_timing(one_cycle_data: CycleArray) -> float:
    """Normalized frame index where hip-angle acceleration is largest."""
    hip_angles = get_angle_vector(one_cycle_data, "knee", "hip", "shoulder")
    accel = np.gradient(np.gradient(hip_angles))
    return float(np.argmax(accel) / len(accel))


def fastest_hip_velocity_timing(one_cycle_data: CycleArray) -> float:
    """Normalized frame index where hip-angle velocity is largest."""
    hip_angles = get_angle_vector(one_cycle_data, "knee", "hip", "shoulder")
    velocity = np.gradient(hip_angles)
    return float(np.argmax(velocity) / len(velocity))


def fastest_elbow_accel_timing(one_cycle_data: CycleArray) -> float:
    """Normalized frame index where wrist–elbow–shoulder angular acceleration peaks."""
    elbow = get_angle_vector(one_cycle_data, "wrist", "elbow", "shoulder")
    accel = np.gradient(np.gradient(elbow))
    return float(np.argmax(accel) / len(accel))


def knee_min_accel_timing(one_cycle_data: CycleArray) -> float:
    """Normalized frame index where ankle–knee–hip angular acceleration is closest to zero."""
    knee = get_angle_vector(one_cycle_data, "ankle", "knee", "hip")
    accel = np.gradient(np.gradient(knee))
    return float(np.argmin(np.abs(accel)) / len(accel))


def body_angle_at_catch(one_cycle_data: CycleArray) -> float:
    """
    Angle between the torso and the seat rail at the catch (degrees).

    90 is upright; smaller means more forward reach. Referencing the rail rather than the
    image axes keeps this stable under camera roll.
    """
    frame = catch_index(one_cycle_data)
    torso = (
        get_joint_xy(one_cycle_data, "shoulder")[frame]
        - get_joint_xy(one_cycle_data, "hip")[frame]
    )
    along = slide_axis(one_cycle_data)
    across = np.array([-along[1], along[0]])
    return float(np.degrees(np.arctan2(abs(torso @ across), abs(torso @ along))))


def leg_back_lag(one_cycle_data: CycleArray) -> float:
    """
    Normalized delay between peak leg drive and peak back swing.

    Positive means the legs peak before the back opens — the sequencing coaches look for.
    """
    knee = np.gradient(get_angle_vector(one_cycle_data, "ankle", "knee", "hip"))
    hip = np.gradient(get_angle_vector(one_cycle_data, "knee", "hip", "shoulder"))
    return float((np.argmax(knee) - np.argmax(hip)) / len(one_cycle_data))


def elbow_angle_range(one_cycle_data: CycleArray) -> float:
    """Range of the wrist–elbow–shoulder angle (degrees): how far the arms draw."""
    elbow = get_angle_vector(one_cycle_data, "wrist", "elbow", "shoulder")
    return float(elbow.max() - elbow.min())


DEFAULT_FEATURE_EXTRACTORS: list[FeatureExtractor] = [
    min_hip_angle,
    fastest_hip_velocity_timing,
    # fastest_elbow_accel_timing,
    knee_min_accel_timing,
    body_angle_at_catch,
    leg_back_lag,
    elbow_angle_range,
]

FEATURE_NAMES: list[str] = [fn.__name__ for fn in DEFAULT_FEATURE_EXTRACTORS]


if __name__ == "__main__":
    per_cycle_data = np.genfromtxt("cycle_data.csv", delimiter=",", skip_header=1)
    cycles = faster_index_by_0_column(per_cycle_data)
    features = np.array(features_from_cycles(cycles, DEFAULT_FEATURE_EXTRACTORS))
    print(features.shape)
