# pulse_check.py
from urdfpy import URDF
import numpy as np
import sys

URDF_PATH = sys.argv[1]
wrist = "hand_base_link"
tips  = ["thumb_distal","index_distal","middle_distal","ring_distal","pinky_distal"]
targets = ["thumb_cmc_yaw","thumb_cmc_pitch","index_mcp_pitch","middle_mcp_pitch","ring_mcp_pitch","pinky_mcp_pitch"]

robot = URDF.load(URDF_PATH)

def tip_pos(q=None):
    fk = robot.link_fk(cfg=q or {})
    return {t: fk[robot.link_map[t]][:3,3].copy() for t in tips}

base = tip_pos()
for jn in targets:
    q = {jn: 0.0873}  # +5°
    moved = tip_pos(q)
    disp = {t: np.linalg.norm(moved[t]-base[t]) for t in tips}
    main = max(disp, key=disp.get)
    print(f"{jn:>18s} -> max tip: {main}, disp={disp[main]:.4f} m")
