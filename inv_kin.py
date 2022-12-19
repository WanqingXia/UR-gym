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
no_solution = 0
if __name__ == "__main__":
    env = gymnasium.make("UR5OriReach-v1", render=True)
    ur5e = ur_kinematics.URKinematics('ur5e')
    joints = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])

    for trial in tqdm(range(100)):
        count = 0
        for i in range(100):
            env.task.reset()
            goal = env.task.get_goal()
            new_goal = np.concatenate((goal[:3], np.roll(goal[3:], -1)))
            angles = ur5e.inverse(new_goal, False)
            print("wanted angles: ", angles)
            if angles is None:
                no_solution += 1
                continue
            if np.max(np.abs(angles)) > 6.28:
                i = i-1
                pass
            else:
                env.robot.set_joint_angles(angles)
                env.sim.step()

                print("achieved angles: ", env.robot.get_joint_angles())
                stat = env.task.get_achieved_goal()
                success = env.task.is_success(goal, stat)
                if success:
                    count += 1
                    print("success")
                else:
                    print("fail")
        rate[trial] = count/100
    print("no solution occur count: ", no_solution)
    print("average success rate: ", np.average(rate))

