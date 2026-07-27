"""Split per-video joint frames into individual rowing stroke cycles."""

from __future__ import annotations

from typing import Final

import joint_info
import numpy as np
from scipy.signal import find_peaks

CORE_JOINTS: Final[tuple[str, ...]] = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
PEAK_DISTANCE: Final[int] = 10
PEAK_PROMINENCE: Final[float] = 0.3


def right_side_closer(frames: np.ndarray) -> bool:
    """
    True when the rower's right side is nearer the camera, False when the left side is.

    MediaPipe z is depth relative to the hip midpoint: smaller (more negative) is closer
    to the camera. Averaging the core joints over every frame separates the two sides by
    ~0.25 in normalized units on side-view footage, so a plain mean comparison needs no
    threshold — on SampleVideos every single frame agrees with its video's verdict.

    ``frames`` must omit the leading index column (row_grade in column 0), the same layout
    ``split_cycles`` and the feature extractors receive.
    """
    left = [joint_info.COL_LOOKUP[f"left_{part}_Z"] - 1 for part in CORE_JOINTS]
    right = [joint_info.COL_LOOKUP[f"right_{part}_Z"] - 1 for part in CORE_JOINTS]
    return bool(frames[:, right].mean() < frames[:, left].mean())


def x_dif_between_points(
    arr: np.ndarray,
    label1: str,
    label2: str,
    right_hand: bool | None = None,
) -> np.ndarray:
    """
    Return the per-frame difference in X between two body-part landmarks.

    ``arr`` must omit the leading index column (e.g. after ``faster_index_by_0_column``).
    ``right_hand`` defaults to whichever side ``right_side_closer`` finds nearer the
    camera; pass True/False to force a side.
    """
    if right_hand is None:
        right_hand = right_side_closer(arr)

    dims = ("X", "Y", "Z")
    handed = "right" if right_hand else "left"
    column_labels_1 = [f"{handed}_{label1}_{xyz}" for xyz in dims]
    column_labels_2 = [f"{handed}_{label2}_{xyz}" for xyz in dims]
    cols_1 = [joint_info.COL_LOOKUP[label] - 1 for label in column_labels_1]
    cols_2 = [joint_info.COL_LOOKUP[label] - 1 for label in column_labels_2]
    return arr[:, cols_1[0]] - arr[:, cols_2[0]]


def index_by_0_column(all_frames: np.ndarray) -> list[np.ndarray]:
    """
    Split ``all_frames`` into a list of arrays by changes in column 0 (video index).

    Each slice drops column 0 and keeps ``row_grade`` plus joint coordinates.
    """
    num_videos = len(set(all_frames[:, 0]))
    prev_index = 0.0
    change_points = [0]
    for cur_row in range(len(all_frames)):
        if all_frames[cur_row, 0] != prev_index:
            change_points.append(cur_row)
        prev_index = all_frames[cur_row, 0]
    change_points.append(len(all_frames))

    indexed_by_video: list[np.ndarray] = []
    for i in range(num_videos):
        start = change_points[i]
        stop = change_points[i + 1]
        indexed_by_video.append(np.copy(all_frames[start:stop, 1:]))
    return indexed_by_video


def faster_index_by_0_column(all_frames: np.ndarray) -> list[np.ndarray]:
    """
    Split rows wherever column 0 changes; return slices without column 0.

    Used for frame CSVs (video index) and cycle CSVs (cycle index).
    """
    splits = np.where(np.diff(all_frames[:, 0]) != 0)[0] + 1
    return list(np.split(all_frames[:, 1:], splits))


def split_cycles_with_videos(
    list_of_2d_arrays: list[np.ndarray],
) -> tuple[list[np.ndarray], list[int]]:
    """
    Detect stroke peaks via wrist minus ankle X, then slice cycles between peaks.

    Uses the side nearer the camera (per video), and skips videos with fewer than two peaks.
    Also returns the source video index for each cycle, so evaluation can hold out whole
    videos rather than splitting a rower's strokes across train and test.
    """
    cycles_out: list[np.ndarray] = []
    video_ids: list[int] = []
    for video_index, arr in enumerate(list_of_2d_arrays):
        arr = np.asarray(arr)
        split_vector = x_dif_between_points(arr, "wrist", "ankle")
        peaks, _ = find_peaks(split_vector, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)

        if len(peaks) < 2:
            continue

        start, end = peaks[0], peaks[-1]
        trimmed = arr[start : end + 1]
        trimmed_peaks = peaks[(peaks >= start) & (peaks <= end)] - start

        for i in range(len(trimmed_peaks) - 1):
            a = trimmed_peaks[i]
            b = trimmed_peaks[i + 1]
            cycle = trimmed[a : b + 1]
            cycles_out.append(cycle)
            video_ids.append(video_index)

    return cycles_out, video_ids


def split_cycles(list_of_2d_arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Cycles only, for callers that don't need the source video (e.g. the web app)."""
    return split_cycles_with_videos(list_of_2d_arrays)[0]


def split_cycles_with_ranges(arr: np.ndarray) -> tuple[list[np.ndarray], list[list[int]]]:
    """
    Cycles for one video plus the ``[start, end]`` row indices (inclusive) of each.

    The indices refer to rows of ``arr``, which is the same row space as the smoothed
    landmark/timestamp arrays, so the web app can map each stroke to a video time range.
    """
    split_vector = x_dif_between_points(arr, "wrist", "ankle")
    peaks, _ = find_peaks(split_vector, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    if len(peaks) < 2:
        return [], []
    cycles = [arr[peaks[i] : peaks[i + 1] + 1] for i in range(len(peaks) - 1)]
    ranges = [[int(peaks[i]), int(peaks[i + 1])] for i in range(len(peaks) - 1)]
    return cycles, ranges


def write_cycles_to_file(
    per_cycle_data: list[np.ndarray],
    video_ids: list[int],
    filename: str = "cycle_data.csv",
) -> None:
    """
    Write cycles to CSV with a leading cycle-id column and a trailing video-index column.

    Not efficient for large pipelines; convenient for inspection.
    """
    pieces: list[np.ndarray] = []
    for i, (arr, video_index) in enumerate(zip(per_cycle_data, video_ids)):
        index_col = np.full((arr.shape[0], 1), i, dtype=np.float64)
        video_col = np.full((arr.shape[0], 1), video_index, dtype=np.float64)
        pieces.append(np.hstack((index_col, arr, video_col)))

    result = np.vstack(pieces)
    np.savetxt(filename, result, delimiter=",", fmt="%.6f", header=joint_info.CYCLE_HEADER)


if __name__ == "__main__":
    all_frames = np.genfromtxt(
        "all_videos_all_joints.csv",
        delimiter=",",
        skip_header=1,
    )
    frames_by_video = faster_index_by_0_column(all_frames)
    per_cycle_data, video_ids = split_cycles_with_videos(frames_by_video)
    write_cycles_to_file(per_cycle_data, video_ids)
