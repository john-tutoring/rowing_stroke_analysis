"""Split per-video joint frames into individual rowing stroke cycles."""

from __future__ import annotations

import joint_info
import numpy as np
from scipy.signal import find_peaks


def x_dif_between_points(
    arr: np.ndarray,
    label1: str,
    label2: str,
    right_hand: bool = True,
) -> np.ndarray:
    """
    Return the per-frame difference in X between two body-part landmarks.

    ``arr`` must omit the leading index column (e.g. after ``faster_index_by_0_column``).
    """
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


def split_cycles(list_of_2d_arrays: list[np.ndarray]) -> list[np.ndarray]:
    """
    Detect stroke peaks via right wrist minus right ankle X, then slice cycles between peaks.

    Skips videos with fewer than two peaks.
    """
    cycles_out: list[np.ndarray] = []
    for arr in list_of_2d_arrays:
        arr = np.asarray(arr)
        split_vector = x_dif_between_points(arr, "wrist", "ankle")
        peaks, _ = find_peaks(split_vector, distance=10, prominence=0.3)

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

    return cycles_out


def write_cycles_to_file(
    per_cycle_data: list[np.ndarray],
    filename: str = "cycle_data.csv",
) -> None:
    """
    Write cycles to CSV with a leading cycle-id column.

    Not efficient for large pipelines; convenient for inspection.
    """
    pieces: list[np.ndarray] = []
    for i, arr in enumerate(per_cycle_data):
        index_col = np.full((arr.shape[0], 1), i, dtype=np.float64)
        pieces.append(np.hstack((index_col, arr)))

    result = np.vstack(pieces)
    np.savetxt(filename, result, delimiter=",", fmt="%.6f", header=joint_info.HEADER)


if __name__ == "__main__":
    all_frames = np.genfromtxt(
        "all_videos_all_joints.csv",
        delimiter=",",
        skip_header=1,
    )
    frames_by_video = faster_index_by_0_column(all_frames)
    per_cycle_data = split_cycles(frames_by_video)
    write_cycles_to_file(per_cycle_data)
