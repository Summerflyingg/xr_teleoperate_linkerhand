#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
from multiprocessing import Process, Array, Lock

import threading
import pathlib
# 根据服务器环境设置LinkerHand SDK的路径
linker_hand_sdk_path = './robot_control/linkerhand-python-sdk'
sys.path.append(linker_hand_sdk_path)
sys.path.append(str(pathlib.Path(os.getcwd()).parent))
print(sys.path)
from LinkerHand.linker_hand_api import LinkerHandApi
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

linker_o6_tip_indices = [4, 9, 14, 19, 24]
O6_Num_Motors = 6

joint_limits_rad = {
    # NOTE: The mapping of index (0-5) to the specific joint is an assumption and needs verification from the SDK documentation.
    0: (0.0, 0.58),  # 拇指屈伸 (thumb_mcp_pitch): ~78 degrees
    1: (0.0, 1.36),  # 拇指侧摆 (R_thumb_cmc_yaw): ~55 degrees
    2: (0.0, 1.6),  # 食指 (R_index_mcp_pitch): 90 degrees
    3: (0.0, 1.6),  # 中指 (R_middle_mcp_pitch): 90 degrees
    4: (0.0, 1.6),  # 无名指 (R_ring_mcp_pitch): 90 degrees
    5: (0.0, 1.6),  # 小指 (R_pinky_mcp_pitch): 90 degrees
}

class O6_Controller:
    """
    用于控制单个Linker Hand O6机械手的控制器。
    """
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None, 
                 dual_hand_action_array = None, left_can_port = "can1", right_can_port = 'can0', fps=100.0, Unit_Test=False, simulation_mode = False):
        """
        初始化O6控制器。

        :param right_hand_array: 共享内存数组，用于从VR设备获取手部数据。
        :param hand_type: 'left' 或 'right'。
        :param can_port: CAN接口名称，例如 'can0'。
        :param fps: 控制循环的频率。
        :param Unit_Test: 是否为单元测试模式。
        """
        logger_mp.info(f"Initializing O6_Controller {left_can_port}, {right_can_port}...")
        self.fps = fps
        self.simulation_mode = simulation_mode
        if self.simulation_mode:
            raise NotImplementedError("Sim env is not supported yet !")

        # 1. 初始化 HandRetargeting
        if Unit_Test:
            # self.hand_retargeting = HandRetargeting(HandType.O6_HAND_Unit_Test)
            logger_mp.warning("Unit test mode for O6 hand not fully implemented yet.")
            self.hand_retargeting = HandRetargeting(HandType.LINKER_O6_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.LINKER_O6_HAND)
        self.left_can_port = left_can_port
        self.right_can_port = right_can_port

        # Init linker hand api
        try:
            self.left_hand_api, self.right_hand_api = self.init_hand_api(self.left_can_port, self.right_can_port)
        except Exception as e:
            logger_mp.error(f"Failed to initialize LinkerHandApi: {e}")
            return
        logger_mp.info(f"LinkerHandApi for both hand is ready.")


        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array  = Array('d', O6_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', O6_Num_Motors, lock=True)
        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # 2. 启动控制进程
        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array, self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info(f"O6_Controller initialized successfully.\n")

        # self.retarget_out_names = getattr(self.hand_retargeting, 'right_retargeting_joint_names', [
        #     'thumb_cmc_pitch', 'thumb_cmc_yaw', 'index_mcp_pitch', 'middle_mcp_pitch', 'ring_mcp_pitch', 'pinky_mcp_pitch'
        # ])

        # logger_mp.info(f"O6 controller retarget names: {self.retarget_out_names}")


    def _subscribe_hand_state(self):
        logger_mp.info("[LinkerHand_Controller] Subscribe thread started.")
        while True:
            # Left Hand
            left_state_msg = self.left_hand_api.get_state()
            if left_state_msg is not None:
                # logger_mp.info(f"left state msg: {left_state_msg}")
                if len(left_state_msg) == O6_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(O6_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg[i] / 255. # [0, 255] -> [0, 1]
                else:
                    logger_mp.warning(f"[LinkerHand_Controller] Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}")
            # Right Hand
            right_state_msg = self.right_hand_api.get_state()
            if right_state_msg is not None:
                # logger_mp.info(f"right state msg: {left_state_msg}")
                if len(right_state_msg) == O6_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(O6_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg[i] / 255.
                else:
                    logger_mp.warning(f"[LinkerHand_Controller] Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}")

            time.sleep(0.002)

    @staticmethod
    def _send_hand_command(left_hand_api, right_hand_api, left_angle_cmd_scaled, right_angle_cmd_scaled):
        # input command in range (0,255)

        left_hand_api.finger_move(pose=left_angle_cmd_scaled)
        right_hand_api.finger_move(pose=right_angle_cmd_scaled)


    @staticmethod
    def init_hand_api(left_can_port = "can1", right_can_port = 'can0'):
        left_hand_api = LinkerHandApi(hand_joint='O6', hand_type="left", can=left_can_port)
        right_hand_api = LinkerHandApi(hand_joint='O6', hand_type="right", can=right_can_port)
        print("[LinkerHand_Controller] Control process started.")
        return left_hand_api, right_hand_api

    @staticmethod
    def normalize_with_joint_limit(val, min_val, max_val):
        # 1.0: fully open; 0.0: fully closed
        # return np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)
        # if reverse:
        #     return (1 - np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0))
        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

    @staticmethod
    def scale_radian_to_command(left_q_target, right_q_target):
        max_cmd = 255
        min_cmd = 0
        scaled_left_cmd = [int(np.clip(val * max_cmd, min_cmd, max_cmd)) for val in left_q_target]
        scaled_right_cmd = [int(np.clip(val * max_cmd, min_cmd, max_cmd)) for val in right_q_target]
        return scaled_left_cmd, scaled_right_cmd


    def retarget_to_radian(self, left_hand_data, right_hand_data):
        # retarget
        
        ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
        ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]
        left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

        # normalize
        for i in range(O6_Num_Motors):
            min_rad, max_rad = joint_limits_rad.get(i, (0.0, 1.6)) # 默认值

            left_q_target[i] = self.normalize_with_joint_limit(left_q_target[i], min_rad, max_rad)
            right_q_target[i] = self.normalize_with_joint_limit(right_q_target[i], min_rad, max_rad)
        return left_q_target, right_q_target


    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):

        left_hand_api, right_hand_api =self.init_hand_api()
        logger_mp.info("[LinkerHand_Controller] Control process started.")
        self.running = True

        left_q_target  = np.full(O6_Num_Motors, 1.0)
        right_q_target = np.full(O6_Num_Motors, 1.0)
        # try:
        while self.running:
            start_time = time.time()
            # get dual hand state
            with left_hand_array.get_lock():
                left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
            with right_hand_array.get_lock():
                right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

            # Read left and right q_state from shared arrays
            state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

            if not np.all(right_hand_data == 0.0) or not np.all(left_hand_data == 0.0): # if hand data has been initialized.
                left_q_target, right_q_target = self.retarget_to_radian(left_hand_data, right_hand_data)
                scaled_left_cmd, scaled_right_cmd = self.scale_radian_to_command(left_q_target, right_q_target)
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data
                self._send_hand_command(left_hand_api, right_hand_api, scaled_left_cmd, scaled_right_cmd)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)

        # except KeyboardInterrupt:
        #     logger_mp.info(f"[LinkerHand_Controller] Control process stopped by user.")
        # except Exception as e:
        #     print(f"!!!!!!!!!!!{e}")
        #     logger_mp.error(f"[LinkerHand_Controller] An error occurred in control_process.")
        # finally:
        #     logger_mp.info(f"[LinkerHand_Controller] Control process for hand has been closed.")


if __name__ == '__main__':
    logger_mp.info("Starting LinkerHand_Controller example...")
    from o6_control_debug import get_hand_pose
    import json
    mock_left_hand_input = Array('d', 75, lock=True)
    mock_right_hand_input = Array('d', 75, lock=True)


    shared_lock = Lock()
    shared_state = Array('d', O6_Num_Motors * 2, lock=False)
    shared_action = Array('d', O6_Num_Motors * 2, lock=False)

    data_path = "/home/xzfang/projects/xr_teleoperate/teleop/utils/data/pick cube/episode_0002/data.json"
    with open(data_path, "r", encoding="utf8") as fr:
        data = json.load(fr)["data"]


    try:

        controller = O6_Controller(
            left_hand_array=mock_left_hand_input,
            right_hand_array=mock_right_hand_input,
            dual_hand_data_lock=shared_lock,
            dual_hand_state_array=shared_state,
            dual_hand_action_array=shared_action,
            fps=50.0,
            Unit_Test=False, # True
        )

        for idx, item in enumerate(data):
            if idx < 20:
                continue
            try:
                time.sleep(1/30)

                left_hand_data, right_hand_data = get_hand_pose(item)
                # Simulate a slight change in human hand input
                with mock_left_hand_input.get_lock():
                    mock_left_hand_input[:] = left_hand_data.flatten()

                with mock_right_hand_input.get_lock():
                    mock_right_hand_input[:] = right_hand_data.flatten()

                with shared_lock:
                    print(f"Cycle {idx} - Logged State: {[f'{x:.3f}' for x in shared_state[:]]}, Logged Action: {[f'{x:.3f}' for x in shared_action[:]]}")
            except KeyboardInterrupt:
                print("Main loop interrupted. Finishing example.")
                break


        # while True:
        #     idx = 185
        #     item = data[idx]
        #     try:
        #         time.sleep(0.1)

        #         left_hand_data, right_hand_data = get_hand_pose(item)
        #         # Simulate a slight change in human hand input
        #         with mock_left_hand_input.get_lock():
        #             mock_left_hand_input[:] = left_hand_data.flatten()

        #         with mock_right_hand_input.get_lock():
        #             mock_right_hand_input[:] = right_hand_data.flatten()

        #         with shared_lock:
        #             print(f"Cycle {idx} - Logged State: {[f'{x:.3f}' for x in shared_state[:]]}, Logged Action: {[f'{x:.3f}' for x in shared_action[:]]}")
        #     except KeyboardInterrupt:
        #         print("Main loop interrupted. Finishing example.")
        #         break

    except Exception as e:
        print(f"An error occurred in the example: {e}")
    finally:
        print("Exiting main program.")