#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
from multiprocessing import Process, Array, Lock

import threading
import pathlib
import requests
import logging_mp
logger_mp = logging_mp.get_logger(__name__)



class Head_Controller:
    """
    用于控制head的控制器。
    """
    def __init__(self, raw_head_rotation_action_array, fps=100.0, Unit_Test=False, simulation_mode = False):
        """
        初始化O6控制器。

        :param right_hand_array: 共享内存数组，用于从VR设备获取手部数据。
        :param hand_type: 'left' 或 'right'。=
        :param can_port: CAN接口名称，例如 'can0'。
        :param fps: 控制循环的频率。
        :param Unit_Test: 是否为单元测试模式。
        """
        self.fps = fps
        self.simulation_mode = simulation_mode
        if self.simulation_mode:
            raise NotImplementedError("Sim env is not supported yet !")

        self.head_rotation_state_array = Array('d', 9, lock=True)
        self.base_url = "http://127.0.0.1:5000"

        self.actual_head_euler_action_array = Array('d', 3, lock=True)

        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_head_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # 2. 启动控制进程
        hand_control_process = Process(target=self.control_process, args=(raw_head_rotation_action_array, self.actual_head_euler_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info(f"O6_Controller initialized successfully.\n")

        # self.retarget_out_names = getattr(self.hand_retargeting, 'right_retargeting_joint_names', [
        #     'thumb_cmc_pitch', 'thumb_cmc_yaw', 'index_mcp_pitch', 'middle_mcp_pitch', 'ring_mcp_pitch', 'pinky_mcp_pitch'
        # ])

        # logger_mp.info(f"O6 controller retarget names: {self.retarget_out_names}")

    def _send_command(self, head_rotmat):
        offset = {"roll": 0.0, "pitch": -16.0, "yaw": -12.0}
        scale = {"roll": 1.0, "pitch": 2.0, "yaw": 1.0}
        req_i = {"rotation_matrix": head_rotmat, "offset": offset, "scale": scale}
        response = requests.post(f"{self.base_url}/move", json=req_i)
        if response.status_code == 200:
            res = response.json()
            return res["yaw"], res["pitch"], res["roll"]

        else:
            
            raise AssertionError(f"no valid response for head controller, {head_rotmat}")

    def get_state(self):
        # 发送 GET 请求到 /status API
        response = requests.get(f"{self.base_url}/status")

        rotation_matrix = np.array(response.json()['rotation_matrix'])
        return rotation_matrix

    def _subscribe_head_state(self):
        logger_mp.info("[Head_Controller] Subscribe thread started.")

        state_msg = self.get_state()
        while True:

               
            with self.head_rotation_state_array.get_lock():
                for i in range(9):
                    self.head_rotation_state_array[i] = state_msg.flatten().tolist()[i]
                # else:
                #     logger_mp.warning(f"[Head_Controller] Received state_msg but attributes are missing or incorrect. Type: {type(state_msg)}, Content: {str(state_msg)}")

            time.sleep(0.002)



    def control_process(self, raw_action_array, execute_action_euler_array):
        logger_mp.info("[Head_Controller] Control process started.")
        self.running = True
        # try:
        while self.running:
            start_time = time.time()
            # get dual hand state

            with raw_action_array.get_lock():
                action_data  = np.array(raw_action_array[:]).reshape(3, 3).copy()
                if not np.all(action_data == 0.0):
                    action_data = action_data.tolist()
                    exeute_cmd = self._send_command(action_data)

                    with execute_action_euler_array.get_lock():
                        execute_action_euler_array[:] = exeute_cmd
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


# if __name__ == '__main__':
#     logger_mp.info("Starting LinkerHand_Controller example...")
#     from o6_control_debug import get_hand_pose
#     import json
#     mock_left_hand_input = Array('d', 75, lock=True)
#     mock_right_hand_input = Array('d', 75, lock=True)


#     shared_lock = Lock()
#     shared_state = Array('d', O6_Num_Motors * 2, lock=False)
#     shared_action = Array('d', O6_Num_Motors * 2, lock=False)

#     data_path = "/home/xzfang/projects/xr_teleoperate/teleop/utils/data/pick cube/episode_0002/data.json"
#     with open(data_path, "r", encoding="utf8") as fr:
#         data = json.load(fr)["data"]


#     try:

#         controller = O6_Controller(
#             left_hand_array=mock_left_hand_input,
#             right_hand_array=mock_right_hand_input,
#             dual_hand_data_lock=shared_lock,
#             dual_hand_state_array=shared_state,
#             dual_hand_action_array=shared_action,
#             fps=50.0,
#             Unit_Test=False, # True
#         )

#         for idx, item in enumerate(data):
#             if idx < 20:
#                 continue
#             try:
#                 time.sleep(1/30)

#                 left_hand_data, right_hand_data = get_hand_pose(item)
#                 # Simulate a slight change in human hand input
#                 with mock_left_hand_input.get_lock():
#                     mock_left_hand_input[:] = left_hand_data.flatten()

#                 with mock_right_hand_input.get_lock():
#                     mock_right_hand_input[:] = right_hand_data.flatten()

#                 with shared_lock:
#                     print(f"Cycle {idx} - Logged State: {[f'{x:.3f}' for x in shared_state[:]]}, Logged Action: {[f'{x:.3f}' for x in shared_action[:]]}")
#             except KeyboardInterrupt:
#                 print("Main loop interrupted. Finishing example.")
#                 break


#         # while True:
#         #     idx = 185
#         #     item = data[idx]
#         #     try:
#         #         time.sleep(0.1)

#         #         left_hand_data, right_hand_data = get_hand_pose(item)
#         #         # Simulate a slight change in human hand input
#         #         with mock_left_hand_input.get_lock():
#         #             mock_left_hand_input[:] = left_hand_data.flatten()

#         #         with mock_right_hand_input.get_lock():
#         #             mock_right_hand_input[:] = right_hand_data.flatten()

#         #         with shared_lock:
#         #             print(f"Cycle {idx} - Logged State: {[f'{x:.3f}' for x in shared_state[:]]}, Logged Action: {[f'{x:.3f}' for x in shared_action[:]]}")
#         #     except KeyboardInterrupt:
#         #         print("Main loop interrupted. Finishing example.")
#         #         break

#     except Exception as e:
#         print(f"An error occurred in the example: {e}")
#     finally:
#         print("Exiting main program.")