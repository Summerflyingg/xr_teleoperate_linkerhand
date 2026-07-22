import numpy as np

"""
(basis) OpenXR Convention : y up, z back, x right. 
(basis) Robot  Convention : z up, y left, x front.  

under (basis) Robot Convention, humanoid arm's initial pose convention:

    # (initial pose) OpenXR Left Arm Pose Convention (hand tracking):
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (initial pose) OpenXR Right Arm Pose Convention (hand tracking):
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.
  
    # (initial pose) Unitree Humanoid Left Arm URDF Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from palm toward back of the hand.
        - the z-axis pointing from pinky toward index.

    # (initial pose) Unitree Humanoid Right Arm URDF Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from back of the hand toward palm. 
        - the z-axis pointing from pinky toward index.

under (basis) Robot Convention, humanoid hand's initial pose convention:

    # (initial pose) OpenXR Left Hand Pose Convention (hand tracking):
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (initial pose) OpenXR Right Hand Pose Convention (hand tracking):
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.

    # (initial pose) Unitree Humanoid Left Hand URDF Convention:
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from pinky toward index.

    # (initial pose) Unitree Humanoid Right Hand URDF Convention:
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from index toward pinky. 
    
p.s. TeleVuer obtains all raw data under the (basis) OpenXR Convention. 
     In addition, arm pose data (hand tracking) follows the (initial pose) OpenXR Arm Pose Convention, 
     while arm pose data (controller tracking) directly follows the (initial pose) Unitree Humanoid Arm URDF Convention (thus no transform is needed).
     Meanwhile, all raw data is in the WORLD frame defined by XR device odometry.

p.s. From website: https://registry.khronos.org/OpenXR/specs/1.1/man/html/openxr.html.
     You can find **(initial pose) OpenXR Left/Right Arm Pose Convention** related information like this below:
     "The wrist joint is located at the pivot point of the wrist, which is location invariant when twisting the hand without moving the forearm. 
     The backward (+Z) direction is parallel to the line from wrist joint to middle finger metacarpal joint, and points away from the finger tips. 
     The up (+Y) direction points out towards back of the hand and perpendicular to the skin at wrist. 
     The X direction is perpendicular to the Y and Z directions and follows the right hand rule."
     Note: The above context is of course under **(basis) OpenXR Convention**.

p.s. **Unitree Arm/Hand URDF initial pose Convention** information come from URDF files.
"""


def safe_mat_update(prev_mat, mat):
    # Return previous matrix and False flag if the new matrix is non-singular (determinant ≠ 0).
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return prev_mat, False
    return mat, True

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

def safe_rot_update(prev_rot_array, rot_array):
    dets = np.linalg.det(rot_array)
    if not np.all(np.isfinite(dets)) or np.any(np.isclose(dets, 0.0, atol=1e-6)):
        return prev_rot_array, False
    return rot_array, True

# constants variable
T_TO_UNITREE_HUMANOID_LEFT_ARM = np.array([[1, 0, 0, 0],
                                           [0, 0,-1, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 1]])

T_TO_UNITREE_HUMANOID_RIGHT_ARM = np.array([[1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0,-1, 0, 0],
                                            [0, 0, 0, 1]])

T_TO_UNITREE_HAND = np.array([[0,  0, 1, 0],
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0,  0, 0, 1]])

T_ROBOT_OPENXR = np.array([[ 0, 0,-1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 1, 0, 0],
                           [ 0, 0, 0, 1]])

T_OPENXR_ROBOT = np.array([[ 0,-1, 0, 0],
                           [ 0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 0, 0, 1]])

R_ROBOT_OPENXR = np.array([[ 0, 0,-1],
                           [-1, 0, 0],
                           [ 0, 1, 0]])

R_OPENXR_ROBOT = np.array([[ 0,-1, 0],
                           [ 0, 0, 1],
                           [-1, 0, 0]])

CONST_HEAD_POSE = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 1.5],
                            [0, 0, 1, -0.2],
                            [0, 0, 0, 1]])

# For Robot initial position
CONST_RIGHT_ARM_POSE = np.array([[1, 0, 0, 0.15],
                                 [0, 1, 0, 1.13],
                                 [0, 0, 1, -0.3],
                                 [0, 0, 0, 1]])

CONST_LEFT_ARM_POSE = np.array([[1, 0, 0, -0.15],
                                [0, 1, 0, 1.13],
                                [0, 0, 1, -0.3],
                                [0, 0, 0, 1]])

CONST_HAND_ROT = np.tile(np.eye(3)[None, :, :], (25, 1, 1))




def convert_hand_pose_unitree(head_pose, left_arm_pose, right_arm_pose, left_hand_positions, right_hand_positions):
    Bxr_world_head, head_pose_is_valid = safe_mat_update(CONST_HEAD_POSE, head_pose)
    Brobot_world_head = T_ROBOT_OPENXR @ Bxr_world_head @ T_OPENXR_ROBOT

    # 'Arm' pose data follows (basis) OpenXR Convention and (initial pose) OpenXR Arm Convention.
    left_IPxr_Bxr_world_arm, left_arm_is_valid  = safe_mat_update(CONST_LEFT_ARM_POSE, left_arm_pose)
    right_IPxr_Bxr_world_arm, right_arm_is_valid = safe_mat_update(CONST_RIGHT_ARM_POSE, right_arm_pose)

    # Change basis convention
    # From (basis) OpenXR Convention to (basis) Robot Convention:
    #   Brobot_Pose = T_{robot}_{openxr} * Bxr_Pose * T_{robot}_{openxr}^T  ==>
    #   Brobot_Pose = T_{robot}_{openxr} * Bxr_Pose * T_{openxr}_{robot}
    # Reason for right multiply T_OPENXR_ROBOT = fast_mat_inv(T_ROBOT_OPENXR):
    #   This is similarity transformation: B = PAP^{-1}, that is B ~ A
    #   For example:
    #   - For a pose data T_r under the (basis) Robot Convention, left-multiplying Brobot_Pose means:
    #       Brobot_Pose * T_r  ==>  T_{robot}_{openxr} * PoseMatrix_openxr * T_{openxr}_{robot} * T_r
    #   - First, transform T_r to the (basis) OpenXR Convention (The function of T_{openxr}_{robot})
    #   - Then, apply the rotation PoseMatrix_openxr in the OpenXR Convention (The function of PoseMatrix_openxr)
    #   - Finally, transform back to the Robot Convention (The function of T_{robot}_{openxr})
    #   - This results in the same rotation effect under the Robot Convention as in the OpenXR Convention.
    left_IPxr_Brobot_world_arm  = T_ROBOT_OPENXR @ left_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT
    right_IPxr_Brobot_world_arm = T_ROBOT_OPENXR @ right_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT

    # Change initial pose convention 
    # From (initial pose) OpenXR Arm Convention to (initial pose) Unitree Humanoid Arm URDF Convention
    # Reason for right multiply (T_TO_UNITREE_HUMANOID_LEFT_ARM) : Rotate 90 degrees counterclockwise about its own x-axis.
    # Reason for right multiply (T_TO_UNITREE_HUMANOID_RIGHT_ARM): Rotate 90 degrees clockwise about its own x-axis.
    left_IPunitree_Brobot_world_arm = left_IPxr_Brobot_world_arm @ (T_TO_UNITREE_HUMANOID_LEFT_ARM if left_arm_is_valid else np.eye(4))
    right_IPunitree_Brobot_world_arm = right_IPxr_Brobot_world_arm @ (T_TO_UNITREE_HUMANOID_RIGHT_ARM if right_arm_is_valid else np.eye(4))

    # Transfer from WORLD to HEAD coordinate (translation adjustment only)
    left_IPunitree_Brobot_head_arm = left_IPunitree_Brobot_world_arm.copy()
    right_IPunitree_Brobot_head_arm = right_IPunitree_Brobot_world_arm.copy()
    left_IPunitree_Brobot_head_arm[0:3, 3]  = left_IPunitree_Brobot_head_arm[0:3, 3] - Brobot_world_head[0:3, 3]
    right_IPunitree_Brobot_head_arm[0:3, 3] = right_IPunitree_Brobot_world_arm[0:3, 3] - Brobot_world_head[0:3, 3]

    # =====coordinate origin offset=====
    # The origin of the coordinate for IK Solve is near the WAIST joint motor. You can use teleop/robot_control/robot_arm_ik.py Unit_Test to visualize it.
    # The origin of the coordinate of IPunitree_Brobot_head_arm is HEAD. 
    # So it is necessary to translate the origin of IPunitree_Brobot_head_arm from HEAD to WAIST.
    left_IPunitree_Brobot_waist_arm = left_IPunitree_Brobot_head_arm.copy()
    right_IPunitree_Brobot_waist_arm = right_IPunitree_Brobot_head_arm.copy()
    left_IPunitree_Brobot_waist_arm[0, 3] +=0.15 # x
    right_IPunitree_Brobot_waist_arm[0,3] +=0.15
    left_IPunitree_Brobot_waist_arm[2, 3] +=0.45 # z
    right_IPunitree_Brobot_waist_arm[2,3] +=0.45


    if left_hand_positions is not None:
        # Homogeneous, [xyz] to [xyz1]
        #   np.concatenate([25,3]^T,(1,25)) ==> Bxr_world_hand_pos.shape is (4,25)
        # Now under (basis) OpenXR Convention, Bxr_world_hand_pos data like this:
        #    [x0 x1 x2 ··· x23 x24]
        #    [y0 y1 y1 ··· y23 y24]
        #    [z0 z1 z2 ··· z23 z24]
        #    [ 1  1  1 ···  1    1]
        left_IPxr_Bxr_world_hand_pos  = np.concatenate([left_hand_positions.T, np.ones((1, left_hand_positions.shape[0]))])

        # Change basis convention
        # From (basis) OpenXR Convention to (basis) Robot Convention
        # Just a change of basis for 3D points. No rotation, only translation. So, no need to right-multiply fast_mat_inv(T_ROBOT_OPENXR).
        left_IPxr_Brobot_world_hand_pos  = T_ROBOT_OPENXR @ left_IPxr_Bxr_world_hand_pos

        # Transfer from WORLD to ARM frame under (basis) Robot Convention:
        #   Brobot_{world}_{arm}^T * Brobot_{world}_pos ==> Brobot_{arm}_{world} * Brobot_{world}_pos ==> Brobot_arm_hand_pos, Now it's based on the arm frame.
        left_IPxr_Brobot_arm_hand_pos  = fast_mat_inv(left_IPxr_Brobot_world_arm) @ left_IPxr_Brobot_world_hand_pos
        
        # Change initial pose convention
        # From (initial pose) XR Hand Convention to (initial pose) Unitree Humanoid Hand URDF Convention:
        #   T_TO_UNITREE_HAND @ IPxr_Brobot_arm_hand_pos ==> IPunitree_Brobot_arm_hand_pos
        #   ((4,4) @ (4,25))[0:3, :].T ==> (4,25)[0:3, :].T ==> (3,25).T ==> (25,3)           
        # Now under (initial pose) Unitree Humanoid Hand URDF Convention, matrix shape like this:
        #    [x0, y0, z0]
        #    [x1, y1, z1]
        #    ···
        #    [x23,y23,z23]
        #    [x24,y24,z24]
        left_IPunitree_Brobot_arm_hand_pos  = (T_TO_UNITREE_HAND @ left_IPxr_Brobot_arm_hand_pos)[0:3, :].T
    else:
        left_IPunitree_Brobot_arm_hand_pos  = np.zeros((25, 3))

    if right_hand_positions is not None:
        # Homogeneous, [xyz] to [xyz1]
        #   np.concatenate([25,3]^T,(1,25)) ==> Bxr_world_hand_pos.shape is (4,25)
        # Now under (basis) OpenXR Convention, Bxr_world_hand_pos data like this:
        #    [x0 x1 x2 ··· x23 x24]
        #    [y0 y1 y1 ··· y23 y24]
        #    [z0 z1 z2 ··· z23 z24]
        #    [ 1  1  1 ···  1    1]
        right_IPxr_Bxr_world_hand_pos = np.concatenate([right_hand_positions.T, np.ones((1, right_hand_positions.shape[0]))])

        # Change basis convention
        # From (basis) OpenXR Convention to (basis) Robot Convention
        # Just a change of basis for 3D points. No rotation, only translation. So, no need to right-multiply fast_mat_inv(T_ROBOT_OPENXR).
        right_IPxr_Brobot_world_hand_pos = T_ROBOT_OPENXR @ right_IPxr_Bxr_world_hand_pos

        # Transfer from WORLD to ARM frame under (basis) Robot Convention:
        #   Brobot_{world}_{arm}^T * Brobot_{world}_pos ==> Brobot_{arm}_{world} * Brobot_{world}_pos ==> Brobot_arm_hand_pos, Now it's based on the arm frame.
        right_IPxr_Brobot_arm_hand_pos = fast_mat_inv(right_IPxr_Brobot_world_arm) @ right_IPxr_Brobot_world_hand_pos
        
        # Change initial pose convention
        # From (initial pose) XR Hand Convention to (initial pose) Unitree Humanoid Hand URDF Convention:
        #   T_TO_UNITREE_HAND @ IPxr_Brobot_arm_hand_pos ==> IPunitree_Brobot_arm_hand_pos
        #   ((4,4) @ (4,25))[0:3, :].T ==> (4,25)[0:3, :].T ==> (3,25).T ==> (25,3)           
        # Now under (initial pose) Unitree Humanoid Hand URDF Convention, matrix shape like this:
        #    [x0, y0, z0]
        #    [x1, y1, z1]
        #    ···
        #    [x23,y23,z23]
        #    [x24,y24,z24]
        right_IPunitree_Brobot_arm_hand_pos = (T_TO_UNITREE_HAND @ right_IPxr_Brobot_arm_hand_pos)[0:3, :].T
    else:
        right_IPunitree_Brobot_arm_hand_pos  = np.zeros((25, 3))

    return left_IPunitree_Brobot_waist_arm, right_IPunitree_Brobot_waist_arm, left_IPunitree_Brobot_arm_hand_pos, right_IPunitree_Brobot_arm_hand_pos