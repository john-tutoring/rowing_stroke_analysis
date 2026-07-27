"""Extract per-frame MediaPipe pose landmarks from rowing videos."""

from __future__ import annotations

import os
import re
from pathlib import Path
from time import time
from typing import TypedDict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import joint_info

MODEL_PATH: Path = Path(__file__).resolve().parent / "pose_landmarker_lite.task"
VIDEO_DIRECTORY: str = "./SampleVideos"
FRAME_MODULUS: int = 2
RESIZE_TARGET: int = 256
SMOOTH_WINDOW: int = 5


class TimingStats(TypedDict):
    """Accumulated seconds spent in OpenCV vs MediaPipe."""

    cv2: float
    mediapipe: float


class VideoInfo(TypedDict):
    """Source video geometry, for callers that draw on top of the footage."""

    fps: float
    width: int
    height: int


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
    return_timestamps: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, VideoInfo]:
    """
    Read ``filename``, run pose on every ``FRAME_MODULUS``-th frame, return smoothed landmarks.

    Each row is ``[video_index, rowing_grade, x, y, z, ...]`` for 33 joints.
    ``rowing_grade`` comes from the argument, or digits before the extension in ``filename``.

    With ``return_timestamps``, also returns the capture time (seconds) of every
    returned row plus the source video geometry. Times are collected per *kept*
    frame and smoothed with the same window as the landmarks, so they stay
    aligned even though frames are sampled, dropped when no pose is found, and
    averaged.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")

    if rowing_grade is None:
        match = re.search(r"(\d+)(?=\.)", filename)
        if match is None:
            raise ValueError(f"No grade digits before extension in filename: {filename}")
        rowing_grade = int(match.group(1))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    frame_reader = cv2.VideoCapture(filename)
    fps = frame_reader.get(cv2.CAP_PROP_FPS) or 30.0
    width = frame_reader.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = frame_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # MediaPipe normalizes x by width and y by height, which distorts every angle on a
    # non-square frame. Put y in x's units so angles are true; z already shares x's scale.
    y_scale = height / width if width else 1.0
    video_info: VideoInfo = {"fps": float(fps), "width": int(width), "height": int(height)}
    all_landmarks: list[list[float]] = []
    frame_times: list[float] = []
    frame_index = 0

    try:
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
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_index / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            times["mediapipe"] += time() - mediapipe_start

            if result.pose_landmarks:
                frame_landmarks: list[float] = [float(video_index), float(rowing_grade)]
                for lm in result.pose_landmarks[0]:
                    frame_landmarks.extend((lm.x, lm.y * y_scale, lm.z))
                all_landmarks.append(frame_landmarks)
                frame_times.append(timestamp_ms / 1000.0)
    finally:
        frame_reader.release()
        landmarker.close()

    landmark_array = np.array(all_landmarks, dtype=np.float64)
    smoothed = smooth_data(landmark_array)
    if not return_timestamps:
        return smoothed
    return smoothed, smooth_timestamps(frame_times), video_info


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


def smooth_data(frame_data: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Apply a moving average along the time axis for each column."""
    smoothed = np.vstack([
        np.convolve(frame_data[:, j], np.ones(window) / window, mode="valid")
        for j in range(frame_data.shape[1])
    ]).T
    return smoothed


def smooth_timestamps(frame_times: list[float], window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Average capture times over the same window ``smooth_data`` uses."""
    times = np.asarray(frame_times, dtype=np.float64)
    if times.size < window:
        return times
    return np.convolve(times, np.ones(window) / window, mode="valid")


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
