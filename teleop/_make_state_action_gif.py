#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pinocchio as pin


JOINT_LABELS = [
    "L shoulder pitch",
    "L shoulder roll",
    "L shoulder yaw",
    "L elbow",
    "L wrist roll",
    "L wrist pitch",
    "L wrist yaw",
    "R shoulder pitch",
    "R shoulder roll",
    "R shoulder yaw",
    "R elbow",
    "R wrist roll",
    "R wrist pitch",
    "R wrist yaw",
]


def load_arm_arrays(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        document = json.load(file)
    frames = document["data"]
    state = np.asarray(
        [
            frame["states"]["left_arm"]["qpos"]
            + frame["states"]["right_arm"]["qpos"]
            for frame in frames
        ],
        dtype=float,
    )
    action = np.asarray(
        [
            frame["actions"]["left_arm"]["qpos"]
            + frame["actions"]["right_arm"]["qpos"]
            for frame in frames
        ],
        dtype=float,
    )
    fps = float(document.get("info", {}).get("image", {}).get("fps", 30.0))
    return state, action, fps


def build_reduced_robot(project_root):
    teleop_dir = Path(project_root) / "teleop"
    os.chdir(teleop_dir)
    from robot_control.robot_arm_ik import G1_29_ArmIK

    return G1_29_ArmIK().reduced_robot


def arm_polyline(model, data, q, side):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    prefix = "left" if side == "left" else "right"
    joint_names = [
        f"{prefix}_shoulder_pitch_joint",
        f"{prefix}_elbow_joint",
        f"{prefix}_wrist_roll_joint",
    ]
    points = [data.oMi[model.getJointId(name)].translation.copy() for name in joint_names]
    frame_name = "L_ee" if side == "left" else "R_ee"
    points.append(data.oMf[model.getFrameId(frame_name)].translation.copy())
    return np.asarray(points)


def precompute_polylines(robot, q_values):
    model = robot.model
    data = model.createData()
    result = np.empty((len(q_values), 2, 4, 3), dtype=float)
    for index, q in enumerate(q_values):
        result[index, 0] = arm_polyline(model, data, q, "left")
        result[index, 1] = arm_polyline(model, data, q, "right")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("output_path")
    parser.add_argument("--project-root", default="/home/xzfang/projects/xr_teleoperate")
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    state, action, source_fps = load_arm_arrays(args.json_path)
    if state.shape != action.shape or state.shape[1] != 14:
        raise ValueError(f"Expected matching Nx14 arm arrays, got {state.shape} and {action.shape}")
    if not 0 <= args.lag < len(state):
        raise ValueError("lag must be smaller than the number of frames")

    action_time_label = "action(t)" if args.lag == 0 else f"action(t-{args.lag})"
    rmse_label = "same-frame RMSE" if args.lag == 0 else "aligned RMSE"

    robot = build_reduced_robot(args.project_root)
    state_lines = precompute_polylines(robot, state)
    action_lines = precompute_polylines(robot, action)

    aligned_error_deg = np.full_like(state, np.nan)
    aligned_error_deg[args.lag :] = np.rad2deg(state[args.lag :] - action[: -args.lag]) if args.lag else np.rad2deg(state - action)
    aligned_rmse_deg = np.full(len(state), np.nan)
    aligned_rmse_deg[args.lag :] = np.sqrt(np.mean(aligned_error_deg[args.lag :] ** 2, axis=1))
    time_axis = np.arange(len(state)) / source_fps

    fig = plt.figure(figsize=(11, 6.5), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 0.85), wspace=0.28)
    ax_robot = fig.add_subplot(grid[0, 0], projection="3d")
    ax_error = fig.add_subplot(grid[0, 1])

    all_points = np.concatenate((state_lines.reshape(-1, 3), action_lines.reshape(-1, 3)), axis=0)
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2.0
    half_span = max(float(np.ptp(all_points, axis=0).max()) * 0.58, 0.35)
    ax_robot.set_xlim(center[0] - half_span, center[0] + half_span)
    ax_robot.set_ylim(center[1] - half_span, center[1] + half_span)
    ax_robot.set_zlim(center[2] - half_span, center[2] + half_span)
    ax_robot.set_box_aspect((1, 1, 1))
    ax_robot.view_init(elev=15, azim=-55)
    ax_robot.set_xlabel("X (m)")
    ax_robot.set_ylabel("Y (m)")
    ax_robot.set_zlabel("Z (m)")
    ax_robot.grid(True, alpha=0.25)

    state_artists = []
    action_artists = []
    for _ in range(2):
        state_line, = ax_robot.plot([], [], [], color="#1976D2", linewidth=4.0, marker="o", markersize=5)
        action_line, = ax_robot.plot([], [], [], color="#F57C00", linewidth=3.0, linestyle="--", marker="x", markersize=6)
        state_artists.append(state_line)
        action_artists.append(action_line)
    ax_robot.legend(
        handles=[
            Line2D([0], [0], color="#1976D2", linewidth=4, marker="o", label="Measured state(t)"),
            Line2D([0], [0], color="#F57C00", linewidth=3, linestyle="--", marker="x", label=f"Target {action_time_label}"),
        ],
        loc="upper left",
    )

    valid_errors = np.abs(aligned_error_deg[np.isfinite(aligned_error_deg)])
    error_limit = max(3.0, float(np.ceil(np.percentile(valid_errors, 99))))
    y_positions = np.arange(14)
    error_bars = ax_error.barh(y_positions, np.zeros(14), color="#5E35B1", alpha=0.85)
    ax_error.axvline(0, color="black", linewidth=0.8)
    ax_error.set_xlim(-error_limit, error_limit)
    ax_error.set_yticks(y_positions, JOINT_LABELS, fontsize=8)
    ax_error.invert_yaxis()
    ax_error.set_xlabel(f"state(t) - {action_time_label} (deg)")
    ax_error.set_title("Same-frame per-joint difference" if args.lag == 0 else "Aligned per-joint tracking error", fontsize=10)
    ax_error.grid(True, axis="x", alpha=0.25)

    title = fig.suptitle("", fontsize=13, fontweight="bold")

    def update(frame_index):
        action_index = max(0, frame_index - args.lag)
        for side_index in range(2):
            state_points = state_lines[frame_index, side_index]
            action_points = action_lines[action_index, side_index]
            state_artists[side_index].set_data(state_points[:, 0], state_points[:, 1])
            state_artists[side_index].set_3d_properties(state_points[:, 2])
            action_artists[side_index].set_data(action_points[:, 0], action_points[:, 1])
            action_artists[side_index].set_3d_properties(action_points[:, 2])

        errors = aligned_error_deg[frame_index]
        for bar, error in zip(error_bars, errors):
            bar.set_width(0.0 if not np.isfinite(error) else error)
            bar.set_color("#2E7D32" if np.isfinite(error) and abs(error) <= 2.0 else "#D84315")
        rmse_value = aligned_rmse_deg[frame_index]
        title.set_text(
            f"G1 arm state-action validation | frame {frame_index:03d}/{len(state)-1} | "
            f"t={time_axis[frame_index]:.2f}s | {rmse_label}={rmse_value:.2f} deg"
        )
        ax_robot.set_title(f"Blue: measured state(t)   Orange: target {action_time_label}")
        return [*state_artists, *action_artists, *error_bars, title]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        update(len(state) // 2)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
    else:
        frame_indices = list(range(max(args.lag, 0), len(state), max(1, args.stride)))
        movie = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000.0 * args.stride / source_fps, blit=False)
        movie.save(output_path, writer=animation.PillowWriter(fps=source_fps / args.stride), dpi=90)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
