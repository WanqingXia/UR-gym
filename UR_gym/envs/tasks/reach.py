from typing import Any, Dict
import os
import numpy as np
from pyquaternion import Quaternion
from UR_gym.envs.core import Task
from UR_gym.utils import distance, angular_distance, sample_euler, euler_to_quaternion
from ur_ikfast import ur_kinematics
ur5e = ur_kinematics.URKinematics('ur5e')


class ReachIAI(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        distance_threshold=0.005,
        goal_range=0.8,
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([0.2, -goal_range / 2, 0])
        self.goal_range_high = np.array([0.2 + goal_range / 2, goal_range / 2, goal_range])
        self.collision = False

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-1.04)
        self.sim.create_table(length=1.1, width=1.8, height=0.92, x_offset=0.5, z_offset=-0.12)
        self.sim.create_track(length=0.2, width=1.1, height=0.12, x_offset=0.0, z_offset=0.0)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def check_collision(self) -> bool:
        self.collision = False

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return -d.astype(np.float32)


class ReachReg(Task):
    def __init__(
        self,
        sim,
        robot,
        distance_threshold=0.05,
        goal_range=0.8,
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.robot = robot
        self.goal_range_low = np.array([0.2, -goal_range / 2, 0])
        self.goal_range_high = np.array([0.2 + goal_range / 2, goal_range / 2, goal_range])
        self.action_weight = -1
        self.collision_weight = -200
        self.distance_weight = -200
        self.delta = 0.2
        self.collision = False
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-1.04)
        self.sim.create_table(length=1.1, width=1.8, height=0.92, x_offset=0.5, z_offset=-0.12)
        self.sim.create_track(length=0.2, width=1.1, height=0.12, x_offset=0.0, z_offset=0.0)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.collision = False
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def check_collision(self) -> bool:
        self.collision = self.sim.check_collision()

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        reward = np.float32(0.0)
        d = distance(achieved_goal, desired_goal)

        if d <= self.delta:
            reward += 0.5 * np.square(d) * self.distance_weight
        else:
            reward += self.distance_weight * self.delta * (np.abs(d) - 0.5 * self.delta)
        reward += np.sum(np.square(self.robot.get_action())) * self.action_weight
        reward += self.collision_weight if self.collision else 0
        return reward.astype(np.float32)


class ReachOri(Task):
    def __init__(
        self,
        sim,
        robot,
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = 0.05,  # 5cm
        self.angular_distance_threshold = 0.0873,  # 5 degrees
        self.robot = robot
        self.goal_range_low = np.array([0.4, -0.4, 0.2])  # table width, table length, height
        self.goal_range_high = np.array([0.7, 0.4, 0.6])
        self.action_weight = -1
        self.collision_weight = -500
        self.distance_weight = -6
        self.orientation_weight = -14
        self.delta = 0.2
        self.collision = False
        self.link_dist = np.zeros(5)

        self.test_goal = np.zeros(6)
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-1.04)
        self.sim.create_table(length=1.1, width=1.8, height=0.92, x_offset=0.5, z_offset=-0.12)
        self.sim.create_track(length=0.2, width=1.1, height=0.12, x_offset=0.0, z_offset=0.0)
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * 0.05 / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 1.0]),
            texture=os.getcwd() + "/UR_gym/assets/colored_cube_ori.png",
        )
        self.sim.create_box(
            body_name="zone",
            half_extents=np.array([0.15, 0.4, 0.2]),
            mass=0.0,
            ghost=True,
            position=np.array([0.55, 0.0, 0.4]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array(self.goal)

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        ee_orientation = np.array(self.robot.get_ee_orientation())
        return np.concatenate((ee_position, ee_orientation))

    def reset(self) -> None:
        self.collision = False
        if np.array_equal(self.test_goal, np.zeros(6)):
            self.goal = self._sample_goal()
        else:
            self.goal = self.test_goal
            self.test_goal = np.zeros(6)
        self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])

    def set_goal(self, test_goal):
        self.goal = test_goal

    def _sample_goal(self) -> np.ndarray:

        goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high))
        goal_rot_euler = sample_euler()
        goal_rot_quat = euler_to_quaternion(goal_rot_euler)
        goal = np.concatenate((goal_pos, goal_rot_euler))

        # """Randomize goal."""
        # # Adding the goal verification code
        # valid = False
        # counter = 0
        # goal = 0
        # while valid is False:
        #     print("while")
        #     if counter > 0:
        #         print("retrying {} times".format(counter))
        #     counter += 1
        #     goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high))
        #     goal_rot_euler = sample_euler()
        #     goal_rot_quat = euler_to_quaternion(goal_rot_euler)
        #     goal = np.concatenate((goal_pos, goal_rot_euler))
        #     angles = ur5e.inverse(np.concatenate((goal_pos, goal_rot_quat)), False)
        #     if angles is None or np.max(np.abs(angles)) > 6.28:
        #         print("if")
        #         pass
        #     else:
        #         self.robot.set_joint_angles(angles)
        #         self.sim.step()
        #         valid = self.is_success(goal, self.get_achieved_goal())
        #         print("else")
        #         print(valid)
        # print("pass")
        # self.robot.reset()
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        dr = angular_distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold and dr < self.angular_distance_threshold, dtype=np.bool8)

    def check_collision(self) -> bool:
        self.collision = self.sim.check_collision()
        # self.collision = False
        return self.collision

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        reward = np.float32(0.0)

        # ----------------our reward function---------------
        d = distance(achieved_goal.astype(np.float32), desired_goal.astype(np.float32))
        dr = angular_distance(achieved_goal.astype(np.float32), desired_goal.astype(np.float32))
        """Distance Reward"""
        # if d <= self.delta:
        #     reward += 0.5 * np.square(d) * self.distance_weight
        # else:
        #     reward += self.distance_weight * self.delta * (np.abs(d) - 0.5 * self.delta)

        reward += np.abs(d) * self.distance_weight
        """Orientation Reward"""
        reward += np.abs(dr) * self.orientation_weight
        """Action Reward"""
        # reward += np.sum(np.square(self.robot.get_action())) * self.action_weight
        """Collision Reward"""
        reward += self.collision_weight if self.collision else 0

        # ----------------New reward function 1---------------
        # this reward is from article "Collision-free path planning for a guava-harvesting robot based on
        # recurrent deep reinforcement learning"
        # d = distance(achieved_goal.astype(np.float32), desired_goal.astype(np.float32))
        # du = quaternion_to_euler(achieved_goal.astype(np.float32), desired_goal.astype(np.float32))
        #
        # if du.size == 3:
        #     reward += -d + 50 * np.cos(du[0]) * np.cos(du[1]) * np.cos(du[2])
        # else:
        #     reward += -d + 50 * np.cos(du[:, 0]) * np.cos(du[:, 1]) * np.cos(du[:, 2])

        # ----------------New reward function 2---------------
        # this reward is from article "Reinforcement learning with prior policy guidance for motion planning
        # of dual-arm free-floating space robot"
        # d = distance(achieved_goal.astype(np.float32), desired_goal.astype(np.float32))
        # du = quaternion_to_euler(achieved_goal.astype(np.float32), desired_goal.astype(np.float32))
        #
        # reward += 1 - np.tanh(d) + np.cos(np.max(du))

        return reward.astype(np.float32)


class ReachObs(Task):
    def __init__(
        self,
        sim,
        robot,
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = 0.05
        self.robot = robot
        self.goal_range_low = np.array([0.3, -0.5, -0.1])  # table width, table length, height
        self.goal_range_high = np.array([0.75, 0.5, 0.2])
        self.obs_range_low = np.array([0.3, -0.5, 0.25])  # table width, table length, height
        self.obs_range_high = np.array([1.0, 0.5, 0.55])
        self.action_weight = -1
        self.collision_weight = -500
        self.distance_weight = -100
        self.obs_distance_weight = 100
        self.obstacle = np.zeros(6)
        self.collision = False
        self.link_dist = np.zeros(5)
        self.last_dist = np.zeros(5)

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-1.04)
        self.sim.create_table(length=1.1, width=1.8, height=0.92, x_offset=0.5, z_offset=-0.12)
        self.sim.create_track(length=0.2, width=1.1, height=0.12, x_offset=0.0, z_offset=0.0)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=False,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="obstacle",
            half_extents=np.array([0.2, 0.05, 0.05]),  # make the obstacle a rectangular shape
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([0.1, 1.0, 1.0, 1.0]),
        )
        self.sim.create_box(
            body_name="zone_goal",
            half_extents=np.array([0.225, 0.5, 0.15]),
            mass=0.0,
            ghost=True,
            position=np.array([0.525, 0.0, 0.05]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
        )
        self.sim.create_box(
            body_name="zone_obs",
            half_extents=np.array([0.35, 0.5, 0.15]),
            mass=0.0,
            ghost=True,
            position=np.array([0.65, 0.0, 0.4]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.2]),
        )

    def get_obs(self) -> np.ndarray:
        return np.concatenate((self.goal, self.obstacle, self.link_dist))

    def get_achieved_goal(self) -> np.ndarray:
        return np.array(self.robot.get_ee_position())

    def reset(self) -> None:
        self.collision = False
        distance_fail = True
        while distance_fail:
            self.goal = self._sample_goal()
            self.obstacle = self._sample_obstacle()
            self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle", self.obstacle[:3], self.obstacle[3:])
            distance_fail = self.sim.get_target_to_obstacle_distance() < 0.1
        self.collision = self.sim.check_collision()
        self.link_dist = self.sim.get_link_distances()
        self.last_dist = self.link_dist
        if self.collision:
            print("Collision after reset, this should not happen")

    def set_goal_and_obstacle(self, test_data):
        self.goal = test_data[:3]
        self.obstacle = test_data[3:]
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0]))
        self.sim.set_base_pose("obstacle", self.obstacle[:3], self.obstacle[3:])
        self.collision = self.sim.check_collision()
        self.link_dist = self.sim.get_link_distances()
        self.last_dist = self.link_dist

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def _sample_obstacle(self):
        obstacle_pos = self.np_random.uniform(self.obs_range_low, self.obs_range_high)
        obstacle_rot = sample_euler()
        obstacle = np.concatenate((obstacle_pos, obstacle_rot))
        return obstacle

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def check_collision(self) -> bool:
        self.collision = self.sim.check_collision()
        return self.collision

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        self.link_dist = self.sim.get_link_distances()
        dist_change = self.link_dist - self.last_dist
        self.last_dist = self.link_dist
        # reward calculation refer to "Deep Reinforcement Learning for Collision Avoidance of Robotic Manipulators"
        reward = np.float64(0.0)

        # Assuming achieved_goal and desired_goal are numpy arrays of the same shape
        distances = np.abs(distance(achieved_goal, desired_goal))
        reward += np.where(distances < 0.05, 200, 0).astype(np.float32)
        reward += self.collision_weight if self.collision else 0
        reward += self.distance_weight * distances

        # For the loop, we'll use numpy's vectorized operations
        reward_changes = np.where(self.link_dist < 0.2, self.obs_distance_weight * dist_change, 0)
        reward += reward_changes.sum()  # sum up the rewards from all elements

        return reward


class ReachDyn(Task):
    def __init__(
        self,
        sim,
        robot,
    ) -> None:
        super().__init__(sim)
        self.robot = robot
        self.goal_range_low = np.array([0.3, -0.5, -0.1])  # table width, table length, height
        self.goal_range_high = np.array([0.75, 0.5, 0.2])
        self.obs_range_low = np.array([0.3, -0.5, 0.25])  # table width, table length, height
        self.obs_range_high = np.array([1.0, 0.5, 0.55])

        # margin and weight
        self.distance_threshold = 0.05  # 5cm
        self.orientation_threshold = 0.0873  # 5 degrees
        self.action_weight = -1
        self.collision_weight = -500
        self.distance_weight = -70
        self.orientation_weight = -30
        self.dist_change_weight = 100
        self.success_weight = 200

        # Stored values
        self.obstacle_start = np.zeros(6)
        self.obstacle_end = np.zeros(6)
        self.collision = False
        self.link_dist = np.zeros(5)
        self.last_dist = np.zeros(5)

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-1.04)
        self.sim.create_table(length=1.1, width=1.8, height=0.92, x_offset=0.5, z_offset=-0.12)
        self.sim.create_track(length=0.2, width=1.1, height=0.12, x_offset=0.0, z_offset=0.0)
        self.sim.create_target(
            body_name="target",
            half_extents=np.ones(3) * 0.05 / 2,
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 1.0]),
            texture=os.getcwd() + "/UR_gym/assets/colored_cube_ori.png",  # the robot end-effector should point to blue
        )
        self.sim.create_box(
            body_name="obstacle",
            half_extents=np.array([0.2, 0.05, 0.05]),  # make the obstacle a rectangular shape
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([0.1, 1.0, 1.0, 1.0]),
        )
        self.sim.create_box(
            body_name="zone_goal",
            half_extents=np.array([0.225, 0.5, 0.15]),
            mass=0.0,
            ghost=True,
            position=np.array([0.525, 0.0, 0.05]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
        )
        self.sim.create_box(
            # TODO: do we want to make the obstacle appear in the same region as the zone goal?
            body_name="zone_obs",
            half_extents=np.array([0.35, 0.5, 0.15]),
            mass=0.0,
            ghost=True,
            position=np.array([0.65, 0.0, 0.4]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.2]),
        )

    def get_obs(self) -> np.ndarray:
        obstacle_position = self.sim.get_base_position("obstacle")
        obstacle_rotation = self.sim.get_base_rotation("obstacle")
        obstacle_current = np.concatenate((obstacle_position, obstacle_rotation))
        return np.concatenate((self.goal, obstacle_current, self.link_dist))

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        ee_orientation = np.array(self.robot.get_ee_orientation())
        return np.concatenate((ee_position, ee_orientation))

    def reset(self) -> None:
        self.collision = False
        distance_fail = True
        while distance_fail:
            self.goal = self._sample_goal()
            self.obstacle_start = self._sample_obstacle()
            self.obstacle_end = self._sample_obstacle()
            self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])
            self.sim.set_base_pose("obstacle", self.obstacle_end[:3], self.obstacle_end[3:])
            start_end_dist = distance(self.obstacle_end, self.obstacle_start)
            distance_fail = (self.sim.get_target_to_obstacle_distance() < 0.1) or (start_end_dist < 0.3)

        # set obstacle to start position after checking
        self.sim.set_base_pose("obstacle", self.obstacle_start[:3], self.obstacle_start[3:])
        self.collision = self.sim.check_collision()
        self.link_dist = self.sim.get_link_distances()
        self.last_dist = self.link_dist
        if self.collision:
            print("Collision after reset, this should not happen")

    def set_goal_and_obstacle(self, test_data):
        self.goal = test_data[:6]
        self.obstacle_start = test_data[6:12]
        self.obstacle_end = test_data[12:]

        # set the rotation for obstacle to same value for linear movement
        self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])
        # set final pose to keep the environment constant
        self.sim.set_base_pose("obstacle", self.obstacle_start[:3], self.obstacle_start[3:])
        self.collision = self.sim.check_collision()
        self.link_dist = self.sim.get_link_distances()
        self.last_dist = self.link_dist

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high))
        # TODO: pos cannot be 100% random, it should be in the range of the robot
        goal_rot = sample_euler()
        goal = np.concatenate((goal_pos, goal_rot))
        return goal

    def _sample_obstacle(self):
        obstacle_pos = self.np_random.uniform(self.obs_range_low, self.obs_range_high)
        obstacle_rot = sample_euler()
        obstacle = np.concatenate((obstacle_pos, obstacle_rot))
        return obstacle

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        dist_success = distance(achieved_goal, desired_goal) < self.distance_threshold
        ori_success = angular_distance(achieved_goal, desired_goal) < self.orientation_threshold
        return np.array(dist_success and ori_success, dtype=np.bool8)

    def check_collision(self) -> bool:
        self.collision = self.sim.check_collision()
        return self.collision

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        self.link_dist = self.sim.get_link_distances()
        dist_change = self.link_dist - self.last_dist
        self.last_dist = self.link_dist
        # reward calculation refer to "Deep Reinforcement Learning for Collision Avoidance of Robotic Manipulators"
        reward = np.float64(0.0)

        # Assuming achieved_goal and desired_goal are numpy arrays of the same shape
        dist = np.abs(distance(achieved_goal, desired_goal))
        ori_dist = np.abs(angular_distance(achieved_goal, desired_goal))
        reward += np.where(dist < 0.05, self.success_weight, 0).astype(np.float32)
        reward += self.collision_weight if self.collision else 0
        reward += self.distance_weight * dist
        reward += self.orientation_weight * ori_dist

        # For the loop, we'll use numpy's vectorized operations
        reward_changes = np.where(self.link_dist < 0.2, self.dist_change_weight * dist_change, 0)
        reward += reward_changes.sum()  # sum up the rewards from all elements

        return reward
