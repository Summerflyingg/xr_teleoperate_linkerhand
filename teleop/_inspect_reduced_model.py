import os

os.chdir("/home/xzfang/projects/xr_teleoperate/teleop")

from robot_control.robot_arm_ik import G1_29_ArmIK

ik = G1_29_ArmIK()
model = ik.reduced_robot.model
print("nq", model.nq, "njoints", model.njoints)
for jid, name in enumerate(model.names):
    print(jid, name, "idx_q", model.joints[jid].idx_q, "nq", model.joints[jid].nq)
