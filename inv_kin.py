import numpy as np
import time
import sys
import gymnasium
import UR_gym
sys.modules["gym"] = gymnasium
from ur_ikfast import ur_kinematics
from tqdm import tqdm
from pyquaternion import Quaternion

rate = np.zeros(100)
if __name__ == "__main__":
    env = gymnasium.make("UR5OriReach-v1", render=True)
    ur5e = ur_kinematics.URKinematics('ur5e')
    joints = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
    forkin = ur5e.forward(joints)
    normal_stat = env.task.get_achieved_goal()
    rot = Quaternion(np.roll(normal_stat[3:], 1)) / Quaternion(np.roll(forkin[3:], 1))
    for trial in tqdm(range(100)):
        count = 0
        for i in range(100):
            env.task.reset()
            goal = env.task.get_goal()
            # angles = env.robot.ee_displacement_to_target_arm_angles(goal)
            # angles =np.array([-0.63185851,  0.35579859, -1.40336046 ,-2.94117845 , 1.10254117 , 0.69276879])
            # print("#########pybullet", angles)
            newrot = rot * Quaternion(goal[3:])
            ngoal = np.concatenate((goal[:3], newrot.elements))
            angles = ur5e.inverse(ngoal, False)
            # print("#########Ikfast", angles)
            cur = env.robot.get_joint_angles()
            print("wanted angles: ", angles)
            if np.max(np.abs(angles)) > 6.28:
                i = i-1
                pass
            else:
                env.robot.set_joint_angles(angles)
                env.sim.step()
                env.sim.step()
                env.sim.step()
                env.sim.step()
            # for j in range(100):
            #     if j < 50:
            #         env.robot.control_joints(target_angles=cur + j*(angles-cur)/50)
            #         env.sim.step()
            #     else:
            #         env.robot.control_joints(target_angles=angles)
            #         env.sim.step()
                    # time.sleep(0.1)
                print("achieved angles: ", env.robot.get_joint_angles())
                stat = env.task.get_achieved_goal()
                success = env.task.is_success(goal, stat)
                if success:
                    count += 1
                    print("success")
                else:
                    print("fail")
        rate[trial] = count/100

    print("average success rate: ", np.average(rate))

