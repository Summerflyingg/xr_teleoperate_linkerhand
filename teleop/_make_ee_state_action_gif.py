#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


MOTOR_LABELS = [
    "M0  Thumb flex*",
    "M1  Thumb yaw*",
    "M2  Index*",
    "M3  Middle*",
    "M4  Ring*",
    "M5  Pinky*",
]


def load_ee_arrays(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        document = json.load(file)
    frames = document["data"]
    values = {}
    for side in ("left", "right"):
        values[(side, "state")] = np.asarray(
            [frame["states"][f"{side}_ee"]["qpos"] for frame in frames],
            dtype=float,
        )
        values[(side, "action")] = np.asarray(
            [frame["actions"][f"{side}_ee"]["qpos"] for frame in frames],
            dtype=float,
        )
        if values[(side, "state")].shape[1:] != (6,) or values[(side, "action")].shape[1:] != (6,):
            raise ValueError(
                f"Expected Nx6 {side} EE arrays, got "
                f"{values[(side, 'state')].shape} and {values[(side, 'action')].shape}"
            )
    fps = float(document.get("info", {}).get("image", {}).get("fps", 30.0))
    return values, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("output_path")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    values, source_fps = load_ee_arrays(args.json_path)
    frame_count = len(values[("left", "state")])
    if any(len(array) != frame_count for array in values.values()):
        raise ValueError("EE state/action frame counts do not match")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), facecolor="white")
    # Reserve distinct vertical bands for the axes, legend, and footnote.
    fig.subplots_adjust(bottom=0.23, top=0.84, wspace=0.32)
    side_names = ["Left LinkerHand O6", "Right LinkerHand O6"]
    y_positions = np.arange(6)
    state_points = []
    action_points = []
    connectors = []
    value_texts = []
    status_texts = []

    for ax, side_name in zip(axes, side_names):
        ax.axvspan(0.0, 0.15, color="#FFEBEE", alpha=0.8)
        ax.axvspan(0.85, 1.0, color="#E8F5E9", alpha=0.8)
        for y in y_positions:
            ax.plot([0, 1], [y, y], color="#CFD8DC", linewidth=5, solid_capstyle="round", zorder=1)
        state_scatter = ax.scatter(np.zeros(6), y_positions, color="#1976D2", s=80, marker="o", zorder=4)
        action_scatter = ax.scatter(np.zeros(6), y_positions, color="#F57C00", s=90, marker="x", linewidths=2.5, zorder=5)
        connector = LineCollection([], linewidths=3.0, zorder=3)
        ax.add_collection(connector)
        texts = [ax.text(1.025, y, "", va="center", fontsize=8.5, family="monospace") for y in y_positions]
        status = ax.text(0.02, 1.02, "", transform=ax.transAxes, fontsize=10, fontweight="bold")

        ax.set_xlim(-0.05, 1.22)
        ax.set_ylim(-0.65, 5.65)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(MOTOR_LABELS)
        ax.invert_yaxis()
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Normalized qpos   (0 = closed, 1 = open)", labelpad=10)
        ax.set_title(side_name, fontsize=13, fontweight="bold", pad=20)
        ax.grid(True, axis="x", alpha=0.22)
        # Endpoint labels live inside the plot, so they cannot collide with the
        # x-axis label or the figure-level legend below it.
        ax.text(
            0.015,
            0.025,
            "CLOSED",
            transform=ax.transAxes,
            color="#C62828",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.text(
            0.785,
            0.025,
            "OPEN",
            transform=ax.transAxes,
            color="#2E7D32",
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

        state_points.append(state_scatter)
        action_points.append(action_scatter)
        connectors.append(connector)
        value_texts.append(texts)
        status_texts.append(status)

    legend = [
        Line2D([0], [0], color="#1976D2", marker="o", linestyle="None", markersize=8, label="Measured state(t)"),
        Line2D([0], [0], color="#F57C00", marker="x", linestyle="None", markeredgewidth=2.5, markersize=9, label="Target action(t)"),
        Line2D([0], [0], color="#2E7D32", linewidth=3, label="|state-action| <= 0.10"),
        Line2D([0], [0], color="#D84315", linewidth=3, label="|state-action| > 0.10"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.072))
    title = fig.suptitle("", fontsize=14, fontweight="bold")

    def update(frame_index):
        artists = [title]
        rmse_values = []
        for side_index, side in enumerate(("left", "right")):
            state = values[(side, "state")][frame_index]
            action = values[(side, "action")][frame_index]
            difference = state - action
            valid = bool(
                np.all(np.isfinite(state))
                and np.all(np.isfinite(action))
                and np.all((-0.02 <= state) & (state <= 1.02))
                and np.all((-0.02 <= action) & (action <= 1.02))
            )

            state_points[side_index].set_offsets(np.column_stack((state, y_positions)))
            action_points[side_index].set_offsets(np.column_stack((action, y_positions)))
            segments = [np.array([[state[i], y_positions[i]], [action[i], y_positions[i]]]) for i in range(6)]
            connectors[side_index].set_segments(segments)
            connectors[side_index].set_colors(
                ["#2E7D32" if abs(error) <= 0.10 else "#D84315" for error in difference]
            )
            for motor_index, text_artist in enumerate(value_texts[side_index]):
                text_artist.set_text(
                    f"d {difference[motor_index]:+.2f}"
                )
            rmse = float(np.sqrt(np.mean(difference**2)))
            rmse_values.append(rmse)
            status_texts[side_index].set_text(
                f"Same-frame RMSE: {rmse:.3f}   " + ("DATA OK" if valid else "INVALID VALUE")
            )
            status_texts[side_index].set_color("#2E7D32" if valid else "#C62828")
            axes[side_index].set_title(side_names[side_index], fontsize=13, fontweight="bold", pad=26)
            artists.extend(
                [
                    state_points[side_index],
                    action_points[side_index],
                    connectors[side_index],
                    status_texts[side_index],
                    *value_texts[side_index],
                ]
            )

        title.set_text(
            f"LinkerHand O6 same-frame state-action validation | "
            f"frame {frame_index:04d}/{frame_count-1} | t={frame_index/source_fps:.2f}s"
        )
        return artists

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        update(frame_count // 2)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
    else:
        stride = max(1, args.stride)
        frame_indices = list(range(0, frame_count, stride))
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
