from typing import Any, Dict
import os
import numpy as np
from pyquaternion import Quaternion
from UR_gym.envs.core import Task
from UR_gym.utils import distance_single, distance, angle_distance, quaternion_to_euler
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
        d = distance_single(achieved_goal, desired_goal)
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
        distance_threshold=0.05,  # 5cm
        angular_distance_threshold=0.05,  # 9 degrees
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.angular_distance_threshold = angular_distance_threshold
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

        self.test_goal = np.zeros(7)
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
        if np.array_equal(self.test_goal, np.zeros(7)):
            self.goal = self._sample_goal()
        else:
            self.goal = self.test_goal
            self.test_goal = np.zeros(7)
        self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])

    def set_goal(self, new_goal):
        self.test_goal = new_goal

    def set_reward(self):
        self.distance_weight = -10
        self.orientation_weight = -10

    def generate_testset(self):
        # enable this function in core.py to generate points
        goal_range = self.goal_range_high - self.goal_range_low
        rows = int(round((goal_range[0] * 20 + 1) * (goal_range[1] * 20 + 1) * (goal_range[2] * 20 + 1) * 5))
        save_goals = np.zeros((rows, 7))
        counter = 0
        for i in range(int(round((goal_range[0] * 20 + 1)))):
            for j in range(int(round(goal_range[1] * 20 + 1))):
                for k in range(int(round(goal_range[2] * 20 + 1))):
                    for w in range(5):
                        goal = save_goals[counter, :]
                        valid = False
                        tries = 0
                        while valid is False:
                            if tries > 0:
                                print("retrying {} times".format(tries))
                            tries += 1
                            goal_pos = np.zeros(3)
                            goal_pos[0] = i / 20 + self.goal_range_low[0]
                            goal_pos[1] = j / 20 + self.goal_range_low[1]
                            goal_pos[2] = k / 20 + self.goal_range_low[2]
                            goal_rot = np.array(Quaternion.random().elements)
                            goal = np.concatenate((goal_pos, np.roll(goal_rot, -1)))
                            angles = ur5e.inverse(goal, False)
                            if angles is None or np.max(np.abs(angles)) > 6.28:
                                pass
                            else:
                                self.robot.set_joint_angles(angles)
                                self.sim.step()
                                valid = self.is_success(goal, self.get_achieved_goal())
                        self.robot.reset()

                        save_goals[counter, :] = goal
                        counter += 1

        np.savetxt("testset_ori.txt", save_goals)

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        # Adding the goal verification code
        valid = False
        counter = 0
        while valid is False:
            if counter > 0:
                print("retrying {} times".format(counter))
            counter += 1
            goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high))
            goal_rot = np.array(Quaternion.random().elements)
            goal = np.concatenate((goal_pos, np.roll(goal_rot, -1)))
            angles = ur5e.inverse(goal, False)
            if angles is None or np.max(np.abs(angles)) > 6.28:
                pass
            else:
                self.robot.set_joint_angles(angles)
                self.sim.step()
                valid = self.is_success(goal, self.get_achieved_goal())
        self.robot.reset()
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        dr = angle_distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold and dr < self.angular_distance_threshold, dtype=np.bool8)

    def check_collision(self) -> bool:
        # self.collision = self.sim.check_collision()
        self.collision = False

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        reward = np.float32(0.0)

        #----------------our reward function---------------
        d = distance(achieved_goal.astype(np.float32), desired_goal.astype(np.float32))
        dr = angle_distance(achieved_goal.astype(np.float32), desired_goal.astype(np.float32))
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
        distance_threshold=0.05,
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.robot = robot
        self.goal_range_low = np.array([0.4, -0.4, 0.0])  # table width, table length, height
        self.goal_range_high = np.array([0.7, 0.4, 0.2])
        self.obs_range_low = np.array([0.4, -0.4, 0.4])  # table width, table length, height
        self.obs_range_high = np.array([0.7, 0.4, 0.7])
        self.action_weight = -1
        self.collision_weight = -500
        self.distance_weight = -20
        self.obs_distance_weight = -10
        self.obstacle = np.zeros(3)
        self.distances_to_obs = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        self.collision = False
        self.link_dist = np.zeros(5)
        self.last_dist = np.zeros(5)
        self.test_goal = np.zeros(6)

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
        self.sim.create_box(
            body_name="obstacle",
            half_extents=np.ones(3) * 0.2 / 2,
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([0.1, 1.0, 1.0, 1.0]),
        )
        self.sim.create_box(
            body_name="zone_goal",
            half_extents=np.array([0.15, 0.4, 0.1]),
            mass=0.0,
            ghost=True,
            position=np.array([0.55, 0.0, 0.1]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
        )
        self.sim.create_box(
            body_name="zone_obs",
            half_extents=np.array([0.15, 0.4, 0.15]),
            mass=0.0,
            ghost=True,
            position=np.array([0.55, 0.0, 0.55]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.concatenate((self.goal, self.obstacle, self.link_dist))

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.collision = False
        if np.array_equal(self.test_goal, np.zeros(6)):
            self.goal = self._sample_goal()
            self.obstacle = self._sample_obstacle()
            self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle", self.obstacle, np.array([0.0, 0.0, 0.0, 1.0]))
            self.collision, self.link_dist = self.sim.check_collision_obs()
            self.last_dist = self.link_dist
            if self.collision:
                print("Collision after reset, this should not happen")
        else:
            self.goal = self.test_goal[:3]
            self.obstacle = self.test_goal[3:]
            self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle", self.obstacle, np.array([0.0, 0.0, 0.0, 1.0]))
            self.collision, self.link_dist = self.sim.check_collision_obs()
            self.last_dist = self.link_dist

    def set_goal(self, new_goal):
        self.test_goal = new_goal

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def _sample_obstacle(self):
        """Randomize obstacle and keep it 30cm away from the goal."""
        obstacle = self.np_random.uniform(self.obs_range_low, self.obs_range_high)
        return obstacle

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance_single(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def check_collision(self) -> bool:
        self.collision, self.link_dist = self.sim.check_collision_obs()

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        # reward calculation refer to "Deep Reinforcement Learning for Collision Avoidance of Robotic Manipulators"
        reward = np.float32(0.0)

        # reward += np.abs(distance_single(achieved_goal, desired_goal)) * self.distance_weight
        # reward += np.power((0.2 / (np.min(self.link_dist) + 0.2)), 8) * self.obs_distance_weight
        # reward += np.sum(np.square(self.robot.get_action())) * self.action_weight
        # reward += self.collision_weight if self.collision else 0

        # trying new reward
        # Assuming achieved_goal and desired_goal are numpy arrays of the same shape
        distances = np.abs(distance_single(achieved_goal, desired_goal))
        reward += np.where(distances < 0.05, 200, 0).astype(np.float32)
        reward += -500 if self.collision else 0
        reward += -100 * distances

        # For the loop, we'll use numpy's vectorized operations
        reward_changes = np.where(self.link_dist < 0.2, 100 * (self.link_dist - self.last_dist), 0)
        reward += reward_changes.sum()  # sum up the rewards from all elements

        self.last_dist = self.link_dist  # this line is already vectorized
        return reward.astype(np.float32)

    def generate_testset(self):
        # enable this function in core.py to generate points
        # 5000 points, first 3 columns for target position, last 3 columns for obstacle position
        save_goals = np.zeros((5000, 6))
        for counter in range(save_goals.shape[0]):
            save_goals[counter, :3] = self._sample_goal()
            save_goals[counter, 3:] = self._sample_obstacle()
        np.savetxt("testset_obs.txt", save_goals)
