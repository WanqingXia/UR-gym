import os.path as osp
import os
import time

import pybullet as p
import math
from tqdm import tqdm
import sys
import pybullet_data
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import gymnasium
import UR_gym
sys.modules["gym"] = gymnasium
import numpy as np

import pb_ompl

##################################
# Bug in C++ source code, this code is abandoned
##################################

class UR5e_ompl():
    def __init__(self):
        self.obstacles = []

        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # load robot
        robot_id = p.loadURDF(os.getcwd() + "/UR_gym/envs/robots/urdf/ur5e.urdf", (0, 0, 0), useFixedBase=1)
        robot = pb_ompl.PbOMPLRobot(robot_id)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("PRM")
        self.start = [0, -1.57, 0, 0, 0, 0]
        self.goal = [0, 0, 0, 0, 0, 0]

        # add obstacles
        # self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        # add box
        self.add_box([1, 0, 0.7], [0.5, 0.5, 0.05])

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def set_goal(self, goal):
        self.goal = goal

    def solve(self):
        self.robot.set_state(self.start)
        res, path = self.pb_ompl_interface.plan(self.goal)
        if res:
            self.pb_ompl_interface.execute(path, dynamics=False)
        return res, path


if __name__ == '__main__':

    env = gymnasium.make("UR5DynReach-v1", render=True)
    points = np.loadtxt('testset_dyn.txt')
    success = np.zeros(points.shape[0])

    URompl = pb_ompl.PbOMPL(env.robot)
    URompl.set_planner("RRTStar")

    for trials in tqdm(range(4960, 4970)):
        obs = env.reset()
        ee_pos = points[trials, 0:3]
        ee_ori = env.sim.euler_to_quaternion(points[trials, 3:6])
        angles = env.sim.inverse_kinematics("UR5", 7, ee_pos, ee_ori)
        angles = np.array([ 0.54116476, -0.73962382,  3.50326859, -0.97097593,  1.35965961,  0.75535751])
        env.task.set_goal_and_obstacle(points[trials, :])

        env.robot.set_joint_angles(angles)
        env.sim.step()
        robot_obs = env.robot.get_obs()
        stop =1


        res, path = URompl.plan(angles)
        if res:
            URompl.execute(path, dynamics=False)
            success[trials] = True
            path = np.array(path)
            obs = env.task.get_obs()
            robot_obs = env.robot.get_obs()
            stop = 1
        else:
            pass
        time.sleep(2)

    success_rate = (np.sum(success) / success.size) * 100
    print("The success rate is {}%".format(success_rate))
    stop = 1