from UR_gym.envs.robots.UR5 import UR5Ori
from UR_gym.envs.tasks.reach import ReachOri
from UR_gym.pyb_setup import PyBullet
from UR_gym.envs.core import RobotTaskEnv
import numpy as np
import time
import sys
import gymnasium
sys.modules["gym"] = gymnasium

rate = np.zeros(100)
if __name__ == "__main__":
    env = gymnasium.make("UR5OriReach-v1", render=True)
    for trial in range(100):
        count = 0
        for i in range(100):
            env.task.reset()
            goal = env.task.get_goal()
            angles = env.robot.ee_displacement_to_target_arm_angles(goal)
            for j in range(100):
                env.robot.control_joints(target_angles=angles)
                env.sim.step()
                time.sleep(0.1)

            stat = env.task.get_achieved_goal()
            success = env.task.is_success(goal, stat)
            if success:
                count += 1
        rate[trial] = count/100

    print("average success rate: ", np.average(rate))

