import sys
import time

import gymnasium
sys.modules["gym"] = gymnasium
import os
from UR_gym.envs.robots.UR5 import UR5Ori
from UR_gym.pyb_setup import PyBullet
import numpy as np
from UR_gym.utils import *
from tqdm import tqdm
import pb_ompl

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
    # sys.path.insert(0, join(dirname(abspath(__file__)), '../whole-body-motion-planning/src/ompl/py-bindings'))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
import pybullet as p
import utils.ompl_utils as utils
import time
from itertools import product
import copy

INTERPOLATE_NUM = 100
DEFAULT_PLANNING_TIME = 5.0


class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler


class UR5e_CHOMP():
    def __init__(self):
        self.sim = PyBullet(render=True)
        self.robot = UR5Ori(self.sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]))
        self.goal_range_low = np.array([0.3, -0.5, 0.0])  # table width, table length, height
        self.goal_range_high = np.array([0.75, 0.5, 0.2])
        self.obs_range_low = np.array([0.5, -0.5, 0.25])  # table width, table length, height
        self.obs_range_high = np.array([1.0, 0.5, 0.55])
        self.pose = np.zeros(6)


        self.robot_id = self.robot.id
        self.space = PbStateSpace(self.robot.num_dim)
        bounds = ob.RealVectorBounds(self.robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                  custom_limits={}, max_distance=0, allow_collision_links=[])

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)
        self.generate_random_pose()
        self.place_goal_and_obstacle(None, None)
        # self.sim.check_collision()

        self.obstacles = [2, 3, 5]
        self.set_obstacles(self.obstacles)
        self.set_planner("RRT")  # RRT by default

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

    def place_goal_and_obstacle(self, goal, obstacle):
        if obstacle is None:
            obstacle = np.zeros(6)
            obstacle[:3] = np.random.uniform(self.obs_range_low, self.obs_range_high)
            obstacle[3:] = sample_euler_obstacle()

        if goal is None:
            goal = self.pose
        self.sim.set_base_pose("target", goal[:3], goal[3:])
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

    def is_success(self, achieved_goal, desired_goal):
        distance_success = distance(achieved_goal, desired_goal) < 0.05
        orientation_success = angular_distance(achieved_goal, desired_goal) < 0.0873
        return np.array(distance_success & orientation_success, dtype=np.bool8)


    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set

        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True, allow_collision_links = []):
        self.check_link_pairs = utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in utils.get_moving_links(robot.id, robot.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time=DEFAULT_PLANNING_TIME):
        '''
        plan a path to gaol from the given robot start state
        '''
        print("start_planning")
        print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = False
        count = 0

        for i in range(1):
            solved = self.ss.solve(allowed_time)
            count += 1
            if str(solved) == 'Exact solution':
                break

        print('Solving the path with {} seconds, {}'.format(count * 5, solved))
        res = False
        sol_path_list = []
        if solved:
            print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            sol_path_geometric.interpolate(INTERPOLATE_NUM)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            for sol_path in sol_path_list:
                self.is_state_valid(sol_path)
            res = True
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list

    def plan(self, goal, allowed_time=DEFAULT_PLANNING_TIME):
        '''
        plan a path to goal from current robot state
        '''
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time)

    def execute(self, path, dynamics=False):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for num, q in enumerate(path):
            if dynamics:
                self.robot.control_joints(target_angles=q)
                # for i in range(self.robot.num_dim):
                    # p.setJointMotorControl2(self.robot.id, i+1, p.POSITION_CONTROL, q[i], force=5 * 240.)
            else:
                self.robot.set_state(q)
            p.stepSimulation()
            time.sleep(0.01)
        # time.sleep(5)


    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]


if __name__ == '__main__':
    CHOMP = UR5e_CHOMP()
    points = np.loadtxt('testset_dyn.txt')
    CHOMP.set_planner("RRTConnect")
    count = 0
    nums = np.array(0)
    for trial in tqdm(range(len(points))): #  4960-4970 can have stable ik solution
        eepos = np.array(points[trial, :3])
        eeorie = np.array(points[trial, 3:6])
        eeori = np.array(CHOMP.sim.euler_to_quaternion(eeorie))
        angles = CHOMP.compute_inverse_kinematics(eepos, eeori)
        CHOMP.robot.set_joint_angles(angles=angles)
        CHOMP.place_goal_and_obstacle(points[trial, :6], points[trial, 12:])
        CHOMP.sim.step()
        x = CHOMP.robot.get_obs()
        if CHOMP.is_success(x[:6], np.array(points[trial, :6])):
            obs = CHOMP.robot.reset()
            CHOMP.sim.step()
            res, path = CHOMP.plan(angles)
            if res:
                CHOMP.execute(path, dynamics=False)
                path = np.array(path)
                print("\n point number: " + str(trial) + " can be solved \n")
                count += 1
                np.vstack((nums, trial))
            else:
                pass
    print("total success times: " + str(count) + "\n")
    np.savetxt("success_nums.txt", nums)




    # initial_trajectory = CHOMP.initialize_trajectory()
    # trajectory = CHOMP.chomp_gradient_descent(initial_trajectory)
    # for traj in trajectory:
    #     CHOMP.robot.set_joint_angles(angles=traj)
    #     CHOMP.sim.step()
    #     time.sleep(0.5)
    # stop = 1
