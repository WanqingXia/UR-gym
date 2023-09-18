import sys
import gymnasium
sys.modules["gym"] = gymnasium
import os
from UR_gym.envs.robots.UR5 import UR5Ori
from UR_gym.pyb_setup import PyBullet
import numpy as np
from pyquaternion import Quaternion
from ur_ikfast import ur_kinematics
from UR_gym.utils import *
"Have to change robot configuration in urdf from 0->3.141592653589793 Line 230 since the ur_ikfast is different from the urdf."


class UR5e_CHOMP():
    def __init__(self):
        self.sim = PyBullet(render=True)
        self.robot = UR5Ori(self.sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]))
        self.goal_range_low = np.array([0.3, -0.5, 0.0])  # table width, table length, height
        self.goal_range_high = np.array([0.75, 0.5, 0.2])
        self.obs_range_low = np.array([0.5, -0.5, 0.25])  # table width, table length, height
        self.obs_range_high = np.array([1.0, 0.5, 0.55])
        self.pose = np.zeros(7)
        self.ur5e = ur_kinematics.URKinematics('ur5e')

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)
        self.generate_random_pose()
        # self.place_goal_and_obstacle()

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
        # self.sim.create_cylinder(
        #     body_name="obstacle",
        #     radius=0.05,
        #     height=0.4,
        #     mass=0.0,
        #     ghost=False,
        #     position=np.array([0.0, 0.0, 1.0]),
        #     rgba_color=np.array([1.0, 0.92, 0.8, 1.0]),
        #     texture=os.getcwd() + "/UR_gym/assets/cylinder.png",
        # )

        self.sim.create_arm(
            body_name="obstacle",
            radius=0.05,
            height=0.4,
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 0.92, 0.8, 1.0]),
            visual_mesh_path=os.getcwd() + "/UR_gym/assets/arm.obj",
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
        rot = np.roll(np.array(Quaternion.random().elements), -1)
        self.pose = np.concatenate((pos, rot))
        return self.pose

    def place_goal_and_obstacle(self):
        obstacle = np.zeros(7)
        obstacle[:3] = np.random.uniform(self.obs_range_low, self.obs_range_high)
        obstacle[3:] = np.roll(np.array(Quaternion.random().elements), -1)
        self.sim.set_base_pose("target", self.pose[:3], self.pose[3:])
        self.sim.set_base_pose("obstacle", obstacle[:3], obstacle[3:])

    def initialize_trajectory(self, num_waypoints=100):
        """Initialize the trajectory as a straight line between start and end poses."""
        # Linear interpolation between start and end poses
        self.robot.reset()
        start = self.robot.get_obs()[:7]
        ee_trajectory = np.linspace(start, self.pose, num_waypoints)
        joint_trajectory = [self.compute_inverse_kinematics(ee_pose) for ee_pose in ee_trajectory]
        return np.array(joint_trajectory)

    # Example call (using dummy functions for illustration)
    def compute_inverse_kinematics(self, ee_pose):
        return self.ur5e.inverse(ee_pose, False)

    def calculate_smallest_distance_to_obstacle(self, joint_positions):
        self.robot.set_joint_angles(angles=joint_positions)
        return self.sim.check_distance_obs()

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

    # initial_trajectory = CHOMP.initialize_trajectory()
    # trajectory = CHOMP.chomp_gradient_descent(initial_trajectory)
    CHOMP.robot.set_joint_angles(np.array([-0.28394961, -2.12822716, -0.59661657, -1.06940711,  0.25196928, -0.11292072]))
    x = CHOMP.robot.get_ee_position()
    y = CHOMP.robot.get_ee_orientation()

    what = CHOMP.sim.physics_client.getQuaternionFromEuler(y)

    a = np.array([0.57299605, -0.50660484, 0.74738161])
    b = np.array([-1.77311978, -0.74027606, -3.08782273])

    u = np.array([0.5, 0, 0.5])
    v = np.array([0, 0, 0])
    while True:
        what = sample_euler_obstacle()
        CHOMP.sim.set_base_pose("obstacle", u, v)
        stop = 1
    # CHOMP.robot.set_joint_angles(trajectory[99])

    # ur5e = ur_kinematics.URKinematics('ur5e')
    # 
    # CHOMP.robot.reset()
    # p = CHOMP.robot.get_ee_position()
    # r = CHOMP.robot.get_ee_orientation()
    # 
    # start = CHOMP.robot.get_obs()[:7]
    # joint = CHOMP.compute_inverse_kinematics(start)
    # CHOMP.robot.set_joint_angles(joint)
    # 
    # x = CHOMP.robot.get_ee_position()
    # y = CHOMP.robot.get_ee_orientation()
    # 
    # CHOMP.generate_random_pose()
    # CHOMP.pose = 0.4, -0.4, 0.2, -0.22863067, 0.1148493, -0.4594511, 0.85055414
    # CHOMP.sim.set_base_pose("target", CHOMP.pose[:3], CHOMP.pose[3:])
    #
    reward = np.float32(0.0)

    # ----------------our reward function---------------
    d = distance(np.concatenate((x, y)), np.concatenate((a, b)))
    dr = angular_distance(np.concatenate((x, y)), np.concatenate((a, b)))
    """Distance Reward"""
    # if d <= self.delta:
    #     reward += 0.5 * np.square(d) * self.distance_weight
    # else:
    #     reward += self.distance_weight * self.delta * (np.abs(d) - 0.5 * self.delta)

    angles = CHOMP.compute_inverse_kinematics([0.32, 0, 0.2, 1, 0, 0, 0])
    angles = [0, 0, 0, 0, 0, 0]
    CHOMP.robot.set_joint_angles(angles)
    # 
    # # problem identified, the robot simulation got accuracy problem.
    # what = CHOMP.pose
    # a = CHOMP.robot.get_obs()
    # # t = CHOMP.pose

    stop = 1

