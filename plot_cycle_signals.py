"""Plot cycle split signal, hip angle, and timing-feature events for one stroke."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import feature_extraction as fe
from split_strokes import faster_index_by_0_column, x_dif_between_points

# Change this to inspect a different stroke in cycle_data.csv
CYCLE_INDEX = 0


def timing_frame_indices(cycle: fe.CycleArray) -> dict[str, int]:
    """
    Return the frame index for each normalized timing feature.

    Matches the argmax/argmin logic in ``feature_extraction`` timing extractors.
    """
    hip = fe.get_angle_vector(cycle, "knee", "hip", "shoulder")
    elbow = fe.get_angle_vector(cycle, "wrist", "elbow", "shoulder")
    knee = fe.get_angle_vector(cycle, "ankle", "knee", "hip")

    hip_accel = np.gradient(np.gradient(hip))
    elbow_accel = np.gradient(np.gradient(elbow))
    knee_accel = np.gradient(np.gradient(knee))

    return {
        "hip max velocity": int(np.argmax(np.gradient(hip))),
        "hip max accel": int(np.argmax(hip_accel)),
        "elbow max accel": int(np.argmax(elbow_accel)),
        "knee min |accel|": int(np.argmin(np.abs(knee_accel))),
    }


def plot_cycle(
    cycle: fe.CycleArray,
    cycle_index: int = 0,
    *,
    show_peaks: bool = True,
) -> None:
    """
    Draw wrist–ankle X (split signal) and hip angle vs frame, with timing markers.

    Timing markers are placed on the hip-angle axis at the corresponding frame.
    Elbow and knee timings use their own angle at that frame for the marker height.
    """
    frames = np.arange(cycle.shape[0])
    split_signal = x_dif_between_points(cycle, "wrist", "ankle")
    hip_angles = fe.get_angle_vector(cycle, "knee", "hip", "shoulder")
    elbow_angles = fe.get_angle_vector(cycle, "wrist", "elbow", "shoulder")
    knee_angles = fe.get_angle_vector(cycle, "ankle", "knee", "hip")

    timings = timing_frame_indices(cycle)
    angle_at_timing = {
        "hip max velocity": hip_angles[timings["hip max velocity"]],
        "hip max accel": hip_angles[timings["hip max accel"]],
        "elbow max accel": elbow_angles[timings["elbow max accel"]],
        "knee min |accel|": knee_angles[timings["knee min |accel|"]],
    }

    marker_style = {
        "hip max velocity": ("o", "#2ca02c"),
        "hip max accel": ("s", "#d62728"),
        "elbow max accel": ("^", "#9467bd"),
        "knee min |accel|": ("D", "#ff7f0e"),
    }

    fig, ax_split = plt.subplots(figsize=(11, 5))
    ax_hip = ax_split.twinx()

    ax_split.plot(
        frames,
        split_signal,
        color="#1f77b4",
        linewidth=1.5,
        label="R wrist X − R ankle X (split signal)",
    )
    ax_hip.plot(
        frames,
        hip_angles,
        color="#e377c2",
        linewidth=1.5,
        label="Hip angle (knee–hip–shoulder)",
    )

    if show_peaks:
        peaks, _ = find_peaks(split_signal, distance=10, prominence=0.3)
        ax_split.plot(
            peaks,
            split_signal[peaks],
            "x",
            color="#1f77b4",
            markersize=8,
            label="Split peaks",
        )

    for name, frame_idx in timings.items():
        shape, color = marker_style[name]
        ax_hip.scatter(
            frame_idx,
            angle_at_timing[name],
            marker=shape,
            s=90,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=name,
        )

    grade = cycle[0, 0]
    ax_split.set_xlabel("Frame (within cycle)")
    ax_split.set_ylabel("Wrist − ankle X")
    ax_hip.set_ylabel("Angle (degrees)")
    ax_split.set_title(f"Cycle {cycle_index}  ·  row grade {grade:g}")

    lines1, labels1 = ax_split.get_legend_handles_labels()
    lines2, labels2 = ax_hip.get_legend_handles_labels()
    ax_split.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()


def load_cycles(path: str = "cycle_data.csv") -> list[np.ndarray]:
    """Load cycles from CSV."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return faster_index_by_0_column(data)


if __name__ == "__main__":
    cycles = load_cycles()
    if not cycles:
        raise SystemExit("No cycles found. Run extraction.py and split_strokes.py first.")
    if CYCLE_INDEX < 0 or CYCLE_INDEX >= len(cycles):
        raise SystemExit(f"CYCLE_INDEX must be 0..{len(cycles) - 1}, got {CYCLE_INDEX}")

    plot_cycle(cycles[CYCLE_INDEX], CYCLE_INDEX)
