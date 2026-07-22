# -*- coding: utf-8 -*-
"""
3D visualize a 25x3 hand skeleton.

Usage:
- Replace POINTS with your (25, 3) numpy array (in meters or your unit).
- python visualize_hand_25x3.py
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----- 1) Skeleton definition from your index map -----
# Index map (for reference):
# 0  wrist
# 1  thumb-metacarpal
# 2  thumb-phalanx-proximal
# 3  thumb-phalanx-distal
# 4  thumb-tip
# 5  index-finger-metacarpal
# 6  index-finger-phalanx-proximal
# 7  index-finger-phalanx-intermediate
# 8  index-finger-phalanx-distal
# 9  index-finger-tip
# 10 middle-finger-metacarpal
# 11 middle-finger-phalanx-proximal
# 12 middle-finger-phalanx-intermediate
# 13 middle-finger-phalanx-distal
# 14 middle-finger-tip
# 15 ring-finger-metacarpal
# 16 ring-finger-phalanx-proximal
# 17 ring-finger-phalanx-intermediate
# 18 ring-finger-phalanx-distal
# 19 ring-finger-tip
# 20 pinky-finger-metacarpal
# 21 pinky-finger-phalanx-proximal
# 22 pinky-finger-phalanx-intermediate
# 23 pinky-finger-phalanx-distal
# 24 pinky-finger-tip

# Per-finger chains:
THUMB_CHAIN  = [(1, 2), (2, 3), (3, 4)]
INDEX_CHAIN  = [(5, 6), (6, 7), (7, 8), (8, 9)]
MIDDLE_CHAIN = [(10,11), (11,12), (12,13), (13,14)]
RING_CHAIN   = [(15,16), (16,17), (17,18), (18,19)]
PINKY_CHAIN  = [(20,21), (21,22), (22,23), (23,24)]

# Wrist to metacarpals:
PALM_EDGES = [(0, 1), (0, 5), (0,10), (0,15), (0,20)]

# All edges:
DEFAULT_EDGES = (
    PALM_EDGES
    + THUMB_CHAIN
    + INDEX_CHAIN
    + MIDDLE_CHAIN
    + RING_CHAIN
    + PINKY_CHAIN
)

def visualize_hand(points: np.ndarray,
                   edges=DEFAULT_EDGES,
                   annotate: bool = True,
                   save_path: str | None = "hand_3d.png",
                   title: str = "Hand Joints (25x3)"):
    """
    points: (25,3) numpy array
    edges:  list of (i,j) index pairs
    """
    assert isinstance(points, np.ndarray), "points must be a numpy array"
    assert points.shape == (25, 3), f"Expected (25,3), got {points.shape}"
    plt.ion()
    # fig = plt.figure(figsize=(7, 7))

    fig = plt.gcf()  # 获取当前 figure（若无则新建）
    # 如果当前没有3D坐标轴，就创建一个
    ax = None
    for a in fig.axes:
        if hasattr(a, "get_zlim"):  # 判断是否是3D轴
            ax = a
            break
    if ax is None:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax.cla()  # 清空旧图像

    # ax = fig.add_subplot(111, projection="3d")

    # joint scatter
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=25)

    # edges (skeleton)
    for (i, j) in edges:
        xs = [points[i, 0], points[j, 0]]
        ys = [points[i, 1], points[j, 1]]
        zs = [points[i, 2], points[j, 2]]
        ax.plot(xs, ys, zs, linewidth=2)

    # annotate indices
    if annotate:
        for idx, (x, y, z) in enumerate(points):
            ax.text(x, y, z, str(idx), fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # equal aspect ratio
    max_range = (points.max(axis=0) - points.min(axis=0)).max()
    center = points.mean(axis=0)
    ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    # plt.show()
    plt.draw()
    plt.pause(0.001)  # ← 刷新画面

def get_hand_joints(path):
    with open(path, "r", encoding="utf8") as fr:
        data = json.load(fr)["data"]
    return data

if __name__ == "__main__":
    points = np.array(get_hand_joints(sys.argv[1]))
    visualize_hand(points)

