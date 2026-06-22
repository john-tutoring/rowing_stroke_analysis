"""Extract per-frame MediaPipe pose landmarks from rowing videos."""

from __future__ import annotations

import os
import re
from time import time
from typing import TypedDict

import cv2
import mediapipe as mp
import numpy as np

import joint_info

MODEL_PATH: str = "./pose_landmarker_lite.task"
VIDEO_DIRECTORY: str = "./SampleVideos"
FRAME_MODULUS: int = 2
RESIZE_TARGET: int = 256


class TimingStats(TypedDict):
    """Accumulated seconds spent in OpenCV vs MediaPipe."""

    cv2: float
    mediapipe: float


def get_movie_files(directory: str) -> list[str]:
    """Return paths to video files under ``directory`` with a supported extension."""
    exts = (".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm")
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(exts)
    ]


def joints_from_video(
    filename: str,
    video_index: int,
    times: TimingStats,
    rowing_grade: int | None = None,
) -> np.ndarray:
    """
    Read ``filename``, run pose on every ``FRAME_MODULUS``-th frame, return smoothed landmarks.

    Each row is ``[video_index, rowing_grade, x, y, z, ...]`` for 33 joints.
    ``rowing_grade`` comes from the argument, or digits before the extension in ``filename``.
    """
    frame_reader = cv2.VideoCapture(filename)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0)
    if rowing_grade is None:
        match = re.search(r"(\d+)(?=\.)", filename)
        if match is None:
            raise ValueError(f"No grade digits before extension in filename: {filename}")
        rowing_grade = int(match.group(1))
    all_landmarks: list[list[float]] = []
    frame_index = 0

    while frame_reader.isOpened():
        cv2_start = time()
        ret, frame = frame_reader.read()
        if not ret:
            break
        frame_index += 1

        if frame_index % FRAME_MODULUS != 0:
            times["cv2"] += time() - cv2_start
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        times["cv2"] += time() - cv2_start

        mediapipe_start = time()
        results = pose.process(frame_rgb)
        times["mediapipe"] += time() - mediapipe_start
        if results.pose_landmarks:
            frame_landmarks: list[float] = [float(video_index), float(rowing_grade)]
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend((lm.x, lm.y, lm.z))
            all_landmarks.append(frame_landmarks)
    frame_reader.release()
    landmark_array = np.array(all_landmarks, dtype=np.float64)
    return smooth_data(landmark_array)


def normalize_columns(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize each column of ``arr`` to [0, 1]."""
    column_mins = arr.min(axis=0)
    column_maxes = arr.max(axis=0)
    return (arr - column_mins) / (column_maxes - column_mins)


def joints_from_videos(files: list[str], times: TimingStats | None) -> np.ndarray:
    """Extract and stack smoothed landmark rows from every path in ``files``."""
    if times is None:
        times = {"cv2": 0.0, "mediapipe": 0.0}
    all_joints: list[np.ndarray] = []
    for i, file in enumerate(files):
        print(f"Starting file {i + 1} of {len(files)}")
        these_joints = joints_from_video(file, i, times)
        all_joints.append(these_joints)
        print(f"Finished file {i + 1} of {len(files)}")
    frame_data = np.vstack(all_joints)
    print(
        f"All done! Extracted {len(frame_data)} frames of joint data "
        f"from {len(files)} different files."
    )
    return frame_data


def smooth_data(frame_data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a moving average along the time axis for each column."""
    smoothed = np.vstack([
        np.convolve(frame_data[:, j], np.ones(window) / window, mode="valid")
        for j in range(frame_data.shape[1])
    ]).T
    return smoothed


def write_to_file(frame_data: np.ndarray, filename: str = "all_videos_all_joints.csv") -> None:
    """Write landmark rows to a CSV with ``joint_info.HEADER``."""
    np.savetxt(filename, frame_data, delimiter=",", fmt="%.6f", header=joint_info.HEADER)


if __name__ == "__main__":
    print(f"Starting extraction of joint data from frames of videos. {VIDEO_DIRECTORY=}")
    video_file_list = get_movie_files(VIDEO_DIRECTORY)
    print(f"{video_file_list=}")
    timing: TimingStats = {"cv2": 0.0, "mediapipe": 0.0}
    frame_data = joints_from_videos(video_file_list, timing)
    write_to_file(frame_data)
    print("Times per portion:")
    print(f"{timing['cv2']=}\n{timing['mediapipe']=}\n")
