import sys
import time

import gymnasium
sys.modules["gym"] = gymnasium
import os
from UR_gym.envs.robots.UR5 import UR5Ori
from UR_gym.pyb_setup import PyBullet
import numpy as np
from UR_gym.utils import *

class UR5e_CHOMP():
    def __init__(self):
        self.sim = PyBullet(render=True)
        self.robot = UR5Ori(self.sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]))
        self.goal_range_low = np.array([0.3, -0.5, 0.0])  # table width, table length, height
        self.goal_range_high = np.array([0.75, 0.5, 0.2])
        self.obs_range_low = np.array([0.5, -0.5, 0.25])  # table width, table length, height
        self.obs_range_high = np.array([1.0, 0.5, 0.55])
        self.pose = np.zeros(6)

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)
        self.generate_random_pose()
        self.place_goal_and_obstacle()

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
        self.sim.create_cylinder(
            body_name="obstacle",
            radius=0.05,
            height=0.4,
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 0.92, 0.8, 1.0]),
            texture=os.getcwd() + "/UR_gym/assets/cylinder.png",
        )
        self.sim.create_box(
            body_name="zone_goal",
            half_extents=np.array([0.225, 0.5, 0.1]),
            mass=0.0,
            ghost=True,
            position=np.array([0.525, 0.0, 0.1]),
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

    def generate_random_pose(self):
        """Generate a random pose within given limits."""
        pos = np.array(np.random.uniform(self.goal_range_low, self.goal_range_high))
        rot = sample_euler_constrained()
        self.pose = np.concatenate((pos, rot))
        return self.pose

    def place_goal_and_obstacle(self):
        obstacle = np.zeros(6)
        obstacle[:3] = np.random.uniform(self.obs_range_low, self.obs_range_high)
        obstacle[3:] = sample_euler_obstacle()
        self.sim.set_base_pose("target", self.pose[:3], self.pose[3:])
        self.sim.set_base_pose("obstacle", obstacle[:3], obstacle[3:])

    def initialize_trajectory(self, num_waypoints=50):
        """Initialize the trajectory as a straight line between start and end poses."""
        # Linear interpolation between start and end poses
        self.robot.reset()
        start = self.robot.get_obs()[:6]
        ee_trajectory = np.linspace(start, self.pose, num_waypoints)
        joint_trajectory = []
        for ee_pose in ee_trajectory:
            ee_pos = ee_pose[:3]
            ee_ori = self.sim.euler_to_quaternion(ee_pose[3:])
            joint_trajectory.append(self.compute_inverse_kinematics(ee_pos, ee_ori))
        return np.array(joint_trajectory)

    # Example call (using dummy functions for illustration)
    def compute_inverse_kinematics(self, ee_pos, ee_ori):
        return self.sim.inverse_kinematics("UR5", 7, ee_pos, ee_ori)

    def calculate_smallest_distance_to_obstacle(self, joint_positions):
        self.robot.set_joint_angles(angles=joint_positions)
        return self.sim.get_link_distances()

    def compute_collision_cost(self, trajectory):
        """Compute the collision cost for the given trajectory."""
        total_collision_cost = 0.0

        # Iterate through each joint configuration in the trajectory
        for joint_positions in trajectory:
            # Calculate the smallest distance to the obstacle for each link
            distances = self.calculate_smallest_distance_to_obstacle(joint_positions)

            # Compute the collision cost based on the distances
            # Here, we can define a cost function based on the distances, such as an exponential function
            collision_cost = sum(np.exp(-distance) for distance in distances)

            # Add to the total collision cost
            total_collision_cost += collision_cost

        return total_collision_cost

    def compute_smoothness_cost(self, trajectory, weight=1.0):
        """Compute the smoothness cost for the given trajectory."""
        total_smoothness_cost = 0.0
        for joint_config_1, joint_config_2 in zip(trajectory[:-1], trajectory[1:]):
            squared_difference = np.sum((joint_config_1 - joint_config_2) ** 2)
            total_smoothness_cost += weight * squared_difference
        return total_smoothness_cost

    def chomp_gradient_descent(self, trajectory, learning_rate=0.1, max_iterations=100, tolerance=1e-3):
        prev_total_cost = float('inf')
        num_waypoints, num_dof = trajectory.shape
        for iteration in range(max_iterations):
            collision_cost = self.compute_collision_cost(trajectory)
            smoothness_cost = self.compute_smoothness_cost(trajectory)
            total_cost = collision_cost + smoothness_cost
            if abs(prev_total_cost - total_cost) < tolerance:
                break
            collision_gradients = np.zeros_like(trajectory)
            smoothness_gradients = np.zeros_like(trajectory)
            for i in range(1, num_waypoints - 1):
                distances = np.array(self.calculate_smallest_distance_to_obstacle(trajectory[i]))
                collision_gradients[i] = -np.sum(np.exp(-distances))
                smoothness_gradients[i] = 2 * (trajectory[i] - trajectory[i - 1]) + 2 * (
                            trajectory[i] - trajectory[i + 1])
            total_gradients = collision_gradients + smoothness_gradients
            trajectory[1:-1] -= learning_rate * total_gradients[1:-1]
            prev_total_cost = total_cost
        return trajectory


if __name__ == '__main__':
    CHOMP = UR5e_CHOMP()

    eepos = np.array([0.35506645, - 0.03414946,  0.04011879])
    eeori = np.array(CHOMP.sim.euler_to_quaternion([- 1.70473648,  0., - 1.36369828]))
    angles = CHOMP.compute_inverse_kinematics(eepos, eeori)
    CHOMP.robot.set_joint_angles(angles=angles)
    CHOMP.sim.step()
    time.sleep(0.5)
    stop = 1


    initial_trajectory = CHOMP.initialize_trajectory()
    trajectory = CHOMP.chomp_gradient_descent(initial_trajectory)
    for traj in trajectory:
        CHOMP.robot.set_joint_angles(angles=traj)
        CHOMP.sim.step()
        time.sleep(0.5)
    stop = 1
