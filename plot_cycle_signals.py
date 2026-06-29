"""Plot the cycle-split signal for a video with peak markers at stroke boundaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from extraction import TimingStats, joints_from_video
from split_strokes import x_dif_between_points

# Must match split_strokes.split_cycles
PEAK_DISTANCE = 10
PEAK_PROMINENCE = 0.3

PLOT_SUFFIX = "_cycle_split.png"


def default_plot_path(video_path: Path) -> Path:
    """e.g. WomanRowing_85.mp4 → WomanRowing_85_cycle_split.png (next to the video)."""
    return video_path.with_name(f"{video_path.stem}{PLOT_SUFFIX}")


def split_signal_from_video(video_path: str) -> np.ndarray:
    """Extract pose landmarks and return the wrist–ankle X difference per frame."""
    timing: TimingStats = {"cv2": 0.0, "mediapipe": 0.0}
    frames = joints_from_video(video_path, 0, timing, rowing_grade=0)
    if frames.size == 0:
        raise ValueError("No pose detected in video.")

    # Drop video index; keep grade + joints (same layout as split_strokes input)
    arr = frames[:, 1:]
    return x_dif_between_points(arr, "wrist", "ankle")


def cycle_split_peaks(signal: np.ndarray) -> np.ndarray:
    """Peak frame indices used to slice stroke cycles (same logic as split_cycles)."""
    peaks, _ = find_peaks(signal, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    return peaks


def plot_split_signal(
    signal: np.ndarray,
    peaks: np.ndarray,
    *,
    title: str,
    output_path: Path,
) -> None:
    """Line plot of the split metric with x markers at cycle boundaries."""
    frames = np.arange(len(signal))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frames, signal, color="#1f77b4", linewidth=1.5, label="R wrist X − R ankle X")
    ax.plot(
        peaks,
        signal[peaks],
        "x",
        color="#d62728",
        markersize=7,
        markeredgewidth=1.5,
        linestyle="none",
        label="Cycle split (peaks)",
    )

    ax.set_xlabel("Sampled frame")
    ax.set_ylabel("Wrist X − ankle X")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot wrist–ankle X split signal for a rowing video.",
    )
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output PNG path (default: <video_stem>{PLOT_SUFFIX} beside the video)",
    )
    args = parser.parse_args(argv)

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_plot_path(video_path)
    )

    signal = split_signal_from_video(str(video_path))
    peaks = cycle_split_peaks(signal)
    if len(peaks) < 2:
        print(
            f"Warning: only {len(peaks)} peak(s) found; need at least 2 to form a cycle.",
            file=sys.stderr,
        )

    plot_split_signal(signal, peaks, title=video_path.name, output_path=output_path)
    print(output_path)


if __name__ == "__main__":
    main()
