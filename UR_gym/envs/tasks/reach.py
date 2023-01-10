from typing import Any, Dict
import os
import numpy as np
from pyquaternion import Quaternion
from UR_gym.envs.core import Task
from UR_gym.utils import distance, angle_distance
from ur_ikfast import ur_kinematics
ur5e = ur_kinematics.URKinematics('ur5e')


class ReachIAI(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        distance_threshold=0.05,
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
        self.distance_weight = -16
        self.orientation_weight = -4
        self.delta = 0.2
        self.collision = False
        self.link_dist = np.zeros(5)
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
            texture=os.getcwd() + "/UR_gym/assets/colored_cube_heart.png",
        )

    def get_obs(self) -> np.ndarray:
        return np.array(self.goal)

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        ee_orientation = np.array(self.robot.get_ee_orientation())
        return np.concatenate((ee_position, ee_orientation))

    def reset(self) -> None:
        self.collision = False
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        # Adding the goal verification code
        valid = False
        while valid is False:
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
        # print("distance: ", d, "angular distance: ", dr)
        return np.array(d < self.distance_threshold and dr < self.angular_distance_threshold, dtype=np.bool8)

    def check_collision(self) -> bool:
        # self.collision = self.sim.check_collision()
        self.collision = False

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        reward = np.float32(0.0)
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
        return reward.astype(np.float32)


class ReachObs(Task):
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
        self.action_weight = -10
        self.collision_weight = -6
        self.distance_weight = -100
        self.obstacle = np.zeros(3)
        self.delta = 0.1
        self.distances_to_obs = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        self.collision = False
        self.link_dist = np.zeros(5)

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

    def get_obs(self) -> np.ndarray:
        # The environment checks for collision first then get obs so we can return the distance observation as well
        return np.concatenate((self.goal, self.obstacle, self.link_dist))

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.obstacle = self._sample_obstacle()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("obstacle", self.obstacle, np.array([0.0, 0.0, 0.0, 1.0]))
        self.collision, self.link_dist = self.sim.check_collision_obs()
        if self.collision:
            print("Collision after reset, this should not happen")

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        # goal = np.array([0.5, 0.1, 0.2]) # for testing, lock object position
        return goal

    def _sample_obstacle(self):
        """Randomize obstacle and keep it 30cm away from the goal."""
        obstacle = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        while True:
            if distance(obstacle, self.goal) < 0.3:
                obstacle = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
            else:
                # obstacle = np.array([0.4, 0.4, 0.2]) # for testing, lock object position
                return obstacle

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def check_collision(self) -> bool:
        self.collision, self.link_dist = self.sim.check_collision_obs()

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        # reward calculation refer to "Deep Reinforcement Learning for Collision Avoidance of Robotic Manipulators"
        reward = np.float32(0.0)
        d = distance(achieved_goal, desired_goal)

        if d <= self.delta:
            reward += 0.5 * np.square(d) * self.distance_weight
        else:
            reward += self.distance_weight * self.delta * (np.abs(d) - 0.5 * self.delta)

        reward += np.power((0.2 / (np.min(self.link_dist) + 0.2)), 8) * self.collision_weight
        reward += np.sum(np.square(self.robot.get_action())) * self.action_weight
        # reward += self.collision_weight if self.collision else 0
        return reward.astype(np.float32)
