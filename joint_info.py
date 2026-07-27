"""MediaPipe pose landmark names and CSV column indices for joint tables."""

from typing import Final

joints: Final[list[str]] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

JOINT_INDEX_LOOKUP: Final[dict[str, int]] = {label: i for i, label in enumerate(joints)}
JOINT_LABELS: Final[dict[int, str]] = {i: label for i, label in enumerate(joints)}

NUM_JOINTS: Final[int] = 33

headers: Final[list[str]] = ["index", "row_grade"] + [
    label + dim for label in joints for dim in ("_X", "_Y", "_Z")
]
HEADER: Final[str] = ",".join(headers)

# Cycle CSVs carry the source video as a trailing column so held-out-by-video CV is possible.
# It goes last precisely so every COL_LOOKUP offset below stays valid.
CYCLE_HEADER: Final[str] = HEADER + ",video_index"

COL_LOOKUP: Final[dict[str, int]] = {name: idx for idx, name in enumerate(headers)}
