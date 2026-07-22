#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


FINGER_CHAINS = {
    "Thumb": ([0, 1, 2, 3, 4], "#E53935"),
    "Index": ([0, 5, 6, 7, 8, 9], "#FB8C00"),
    "Middle": ([0, 10, 11, 12, 13, 14], "#43A047"),
    "Ring": ([0, 15, 16, 17, 18, 19], "#1E88E5"),
    "Pinky": ([0, 20, 21, 22, 23, 24], "#8E24AA"),
}


def load_hand_positions(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        document = json.load(file)
    frames = document["data"]
    left = np.asarray(
        [frame["states"]["openxr"]["left_hand_positions"] for frame in frames],
        dtype=float,
    )
    right = np.asarray(
        [frame["states"]["openxr"]["right_hand_positions"] for frame in frames],
        dtype=float,
    )
    if left.shape[1:] != (25, 3) or right.shape[1:] != (25, 3):
        raise ValueError(f"Expected Nx25x3 positions, got {left.shape} and {right.shape}")
    fps = float(document.get("info", {}).get("image", {}).get("fps", 30.0))
    return left, right, fps


def valid_tracking(hand):
    return bool(
        hand.shape == (25, 3)
        and np.all(np.isfinite(hand))
        and np.linalg.norm(hand[20] - hand[5]) > 1e-4
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("output_path")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    left_world, right_world, source_fps = load_hand_positions(args.json_path)
    left = left_world - left_world[:, 0:1, :]
    right = right_world - right_world[:, 0:1, :]
    combined = np.concatenate((left.reshape(-1, 3), right.reshape(-1, 3)), axis=0)
    finite = combined[np.all(np.isfinite(combined), axis=1)]
    if not len(finite):
        raise ValueError("No finite hand positions found")
    lower = np.percentile(finite, 0.1, axis=0)
    upper = np.percentile(finite, 99.9, axis=0)
    span = np.maximum(upper - lower, 0.08)
    margin = 0.06 * span
    lower -= margin
    upper += margin
    span = upper - lower

    fig = plt.figure(figsize=(11, 5.8), facecolor="white")
    grid = fig.add_gridspec(1, 2, wspace=0.10)
    fig.subplots_adjust(bottom=0.17, top=0.88)
    axes = [
        fig.add_subplot(grid[0, 0], projection="3d"),
        fig.add_subplot(grid[0, 1], projection="3d"),
    ]
    hands = [left, right]
    side_names = ["Left hand", "Right hand"]
    line_artists = []
    point_artists = []
    status_artists = []

    for ax, side_name in zip(axes, side_names):
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_box_aspect(span)
        ax.view_init(elev=22, azim=-58)
        ax.set_xlabel("X from wrist (m)")
        ax.set_ylabel("Y from wrist (m)")
        ax.set_zlabel("Z from wrist (m)")
        ax.set_title(side_name, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.25)

        side_lines = []
        for _, (_, color) in FINGER_CHAINS.items():
            line, = ax.plot([], [], [], color=color, linewidth=3.0, marker="o", markersize=4)
            side_lines.append(line)
        points = ax.scatter([], [], [], color="#263238", s=12, alpha=0.65)
        wrist = ax.scatter([0], [0], [0], color="black", s=45, marker="s", label="Wrist")
        status = ax.text2D(0.03, 0.95, "", transform=ax.transAxes, fontsize=10, fontweight="bold")
        line_artists.append(side_lines)
        point_artists.append((points, wrist))
        status_artists.append(status)

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=3, marker="o", label=name)
        for name, (_, color) in FINGER_CHAINS.items()
    ]
    legend_handles.append(Line2D([0], [0], color="black", marker="s", linestyle="None", label="Wrist"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 0.015))
    title = fig.suptitle("", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.090,
        "WebXR hand tracking: 25 joints x 3 coordinates, displayed relative to each wrist",
        ha="center",
        fontsize=10,
        color="#455A64",
    )

    def update(frame_index):
        artists = [title]
        for side_index, hand_frames in enumerate(hands):
            hand = hand_frames[frame_index]
            is_valid = valid_tracking(hand)
            display = hand if is_valid else np.full((25, 3), np.nan)
            for line, (indices, _) in zip(line_artists[side_index], FINGER_CHAINS.values()):
                chain = display[np.asarray(indices)]
                line.set_data(chain[:, 0], chain[:, 1])
                line.set_3d_properties(chain[:, 2])
                artists.append(line)
            points, _ = point_artists[side_index]
            points._offsets3d = (display[:, 0], display[:, 1], display[:, 2])
            status_artists[side_index].set_text("Tracking OK" if is_valid else "TRACKING LOST / INVALID")
            status_artists[side_index].set_color("#2E7D32" if is_valid else "#C62828")
            artists.extend((points, status_artists[side_index]))

        time_seconds = frame_index / source_fps
        title.set_text(
            f"AVP 3D hand-joint validation | frame {frame_index:04d}/{len(left)-1} | t={time_seconds:.2f}s"
        )
        return artists

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        update(len(left) // 2)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
    else:
        stride = max(1, args.stride)
        frame_indices = list(range(0, len(left), stride))
        movie = animation.FuncAnimation(
            fig,
            update,
            frames=frame_indices,
            interval=1000.0 * stride / source_fps,
            blit=False,
        )
        movie.save(output_path, writer=animation.PillowWriter(fps=source_fps / stride), dpi=90)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
