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
from ur_ikfast import ur_kinematics
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

    # env = gymnasium.make("UR5OriReach-v1", render=True)
    # ur5e = ur_kinematics.URKinematics('ur5e')
    # initial = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
    # points = np.loadtxt('new_testset.txt')
    # success = np.zeros(points.shape[0])
    # dist = np.zeros(points.shape[0])
    #
    # URompl = pb_ompl.PbOMPL(env.robot)
    # URompl.set_planner("FMT")
    #
    # for trials in tqdm(range(points.shape[0])):
    #     joints = initial.copy()
    #     angles = ur5e.inverse((points[trials, :]), False)
    #     env.task.set_goal(points[trials, :])
    #     obs = env.reset()
    #     if angles is None:
    #         print("found one broken point")
    #         continue
    #
    #     env.robot.set_state(joints)
    #
    #     res, path, solved = URompl.plan(angles)
    #     if res:
    #         print(str(solved))
    #         URompl.execute(path, dynamics=False)
    #         success[trials] = True
    #         path = np.array(path)
    #         for i in range(path.shape[0] - 1):
    #             dist[trials] += np.sum(np.abs(path[i+1, :] - path[i, :]))
    #     else:
    #         pass
    #
    # success_rate = (np.sum(success) / success.size) * 100
    # print("The success rate is {}%".format(success_rate))
    # np.savetxt('RRTStar.txt', np.transpose(dist))

    initial = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
    ur5e = ur_kinematics.URKinematics('ur5e')
    points = np.loadtxt('new_testset.txt')
    success = np.zeros(points.shape[0])
    dist = np.zeros(points.shape[0])
    UR5 = UR5e_ompl()

    for trials in tqdm(range(points.shape[0])):
        joints = initial.copy()
        angles = ur5e.inverse((points[trials, :]), False)
        if angles is None:
            print("found one broken point")
            continue
        UR5.set_goal(angles)
        res, path = UR5.solve()
        if res:
            success[trials] = True
            path = np.array(path)
            for i in range(path.shape[0] - 1):
                dist[trials] += np.sum(np.abs(path[i+1, :] - path[i, :]))
        else:
            pass

        success_rate = (np.sum(success) / success.size) * 100
        print("The success rate is {}%".format(success_rate))
        np.savetxt('RRTStar.txt', np.transpose(dist))


