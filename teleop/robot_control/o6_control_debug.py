import json
import sys
import pathlib
import os
import time

linker_hand_sdk_path = './robot_control/linkerhand-python-sdk'
sys.path.append(linker_hand_sdk_path)
sys.path.append(str(pathlib.Path(os.getcwd()).parent))

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from convert_hand_pose import convert_hand_pose_unitree
from visualize_hand import visualize_hand
import numpy as np

from LinkerHand.linker_hand_api import LinkerHandApi

O6_Num_Motors = 6


joint_limits_rad = {
    # NOTE: The mapping of index (0-5) to the specific joint is an assumption and needs verification from the SDK documentation.
    0: (0.0, 1.36),  # 拇指屈伸 (R_thumb_mcp_yaw): ~78 degrees
    1: (0.0, 0.58),  # 拇指侧摆 (R_thumb_cmc_pitch): ~55 degrees
    2: (0.0, 1.6),  # 食指 (R_index_mcp_pitch): 90 degrees
    3: (0.0, 1.6),  # 中指 (R_middle_mcp_pitch): 90 degrees
    4: (0.0, 1.6),  # 无名指 (R_ring_mcp_pitch): 90 degrees
    5: (0.0, 1.6),  # 小指 (R_pinky_mcp_pitch): 90 degrees
}

def _send_hand_command(left_hand_api, right_hand_api, left_angle_cmd_scaled, right_angle_cmd_scaled):
    # input command in range (0,255)
    left_hand_api.finger_move(pose=left_angle_cmd_scaled)
    right_hand_api.finger_move(pose=right_angle_cmd_scaled)


def init_hand_api(left_can_port = "can1", right_can_port = 'can0'):
    left_hand_api = LinkerHandApi(hand_joint='O6', hand_type="left", can=left_can_port)
    right_hand_api = LinkerHandApi(hand_joint='O6', hand_type="right", can=right_can_port)
    print("[LinkerHand_Controller] Control process started.")
    return left_hand_api, right_hand_api

def get_hand_pose(item: dict):
    head_pose = np.array(item["states"]["openxr"]["head_pose"])
    left_arm_pose = np.array(item["states"]["openxr"]["left_arm_pose"])
    right_arm_pose = np.array(item["states"]["openxr"]["right_arm_pose"])
    left_hand_positions = np.array(item["states"]["openxr"]["left_hand_positions"])
    right_hand_positions = np.array(item["states"]["openxr"]["right_hand_positions"])

    _, _, left_hand_positions, right_hand_positions = convert_hand_pose_unitree(head_pose, left_arm_pose, right_arm_pose, left_hand_positions, right_hand_positions)
    return left_hand_positions, right_hand_positions

def normalize_with_joint_limit(val, min_val, max_val, reverse=False):
    # 1.0: fully open; 0.0: fully closed
    # return np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)
    # if reverse:
    #     return (1 - np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0))
    return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)


def scale_radian_to_command(left_q_target, right_q_target):
    max_cmd = 255
    min_cmd = 0
    scaled_left_cmd = [int(np.clip(val * max_cmd, min_cmd, max_cmd)) for val in left_q_target]
    scaled_right_cmd = [int(np.clip(val * max_cmd, min_cmd, max_cmd)) for val in right_q_target]
    return scaled_left_cmd, scaled_right_cmd

def retarget_to_radian(hand_retargeting: HandRetargeting, left_hand_data, right_hand_data):
    # retarget
    
    ref_left_value = left_hand_data[hand_retargeting.left_indices[1,:]] - left_hand_data[hand_retargeting.left_indices[0,:]]
    ref_right_value = right_hand_data[hand_retargeting.right_indices[1,:]] - right_hand_data[hand_retargeting.right_indices[0,:]]
    left_q_target  = hand_retargeting.left_retargeting.retarget(ref_left_value)[hand_retargeting.left_dex_retargeting_to_hardware]
    right_q_target = hand_retargeting.right_retargeting.retarget(ref_right_value)[hand_retargeting.right_dex_retargeting_to_hardware]

    # normalize
    for i in range(O6_Num_Motors):
        min_rad, max_rad = joint_limits_rad.get(i, (0.0, 1.6)) # 默认值
        # 将弧度值归一化到 [0, 1]
        reverse = False
        # if i == 1:
        #     reverse = True
        left_q_target[i] = normalize_with_joint_limit(left_q_target[i], min_rad, max_rad, reverse)
        right_q_target[i] = normalize_with_joint_limit(right_q_target[i], min_rad, max_rad, reverse)
    return left_q_target, right_q_target

def control_process(data_path):
    left_hand_api, right_hand_api = init_hand_api()
    hand_retargeting = HandRetargeting(HandType.LINKER_O6_HAND)

    print(f"[DEBUG] retarget left indices: {hand_retargeting.left_indices}, retarget right indices: {hand_retargeting.right_indices} ")
    print(f"[DEBUG] hand_retargeting.left_dex_retargeting_to_hardware: {hand_retargeting.left_dex_retargeting_to_hardware}")
    print(f"[DEBUG] hand_retargeting.right_dex_retargeting_to_hardware: {hand_retargeting.right_dex_retargeting_to_hardware}")
    with open(data_path, "r", encoding="utf8") as fr:
        data = json.load(fr)["data"]

    # left_hand_cmd = [0, 0, 0, 0, 0, 0]
    # right_hand_cmd = [0, 0, 0, 0, 0, 0]


    for idx, item in enumerate(data):
        if idx < 20: continue
        print(f"====== current step: {idx} =======")
        left_hand_data, right_hand_data = get_hand_pose(item)

        visualize_hand(left_hand_data)
        left_q_target, right_q_target = retarget_to_radian(hand_retargeting, left_hand_data, right_hand_data)
        # left_hand_cmd = [i + 1 for i in left_hand_cmd]
        # right_hand_cmd = [i + 1 for i in right_hand_cmd]
        # print(left_hand_cmd, right_hand_cmd)
        left_hand_cmd, right_hand_cmd = scale_radian_to_command(left_q_target, right_q_target)
        print("[DEBUG] command send to hand api: ")
        print(left_hand_cmd, right_hand_cmd)
        print("-----------------")
        _send_hand_command(left_hand_api, right_hand_api, left_hand_cmd, right_hand_cmd)
        time.sleep(0.1)


def main():
    data_path = "/home/xzfang/projects/xr_teleoperate/teleop/utils/data/pick cube/episode_0000/data.json"
    control_process(data_path)

if __name__ == "__main__":
    main()