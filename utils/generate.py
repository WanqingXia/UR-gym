import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym

def generate(env, env_name):
    if env_name == "UR5DynReach-v1":
        return generate_dyn(env)
    elif env_name == "UR5StaReach-v1":
        return generate_sta(env)
    elif env_name == "UR5OriReach-v1":
        return generate_ori(env)
    elif env_name == "UR5ObsReach-v1":
        return generate_obs(env)
    elif env_name == "UR5IAIReach-v1" or env_name == "UR5RegReach-v1":
        print("The environment is not supported for generating test set")
        return None
    else:
        print("This environment does not exist")
        return None

def generate_dyn(env):
    """
    generate 5250 points, first 6 columns for target position and orientation,
    middle 6 columns for obstacle start position and orientation,
    last 6 columns for obstacle end position and orientation,
    """

    low = env.task.goal_range_low
    high = env.task.goal_range_high

    # Calculate the number of steps for each dimension
    num_steps = [int((high[i] - low[i]) / 0.05) + 1 for i in range(len(low))]

    # Create an empty array with 18 columns
    generated_goals = np.empty((0, 18))
    # Iterate through each point in the range with 0.05 increments
    for i in range(num_steps[0]):
        for j in range(num_steps[1]):
            for k in range(num_steps[2]):
                for w in range(5):
                    env.task.reset_generate(i, j, k)
                    obs = env.task.get_obs()
                    data_temp = np.zeros((1, 18))
                    data_temp[0, :6] = obs[:6]
                    data_temp[0, 6:12] = env.task.obstacle_start
                    data_temp[0, 12:] = env.task.obstacle_end
                    # stack data
                    generated_goals = np.vstack((generated_goals, data_temp))
    # Convert the list to a numpy array
    return generated_goals

def generate_sta(env):
    """
    generate 5000 points, first 6 columns for target position and orientation,
    last 6 columns for obstacle position and orientation
    """
    num_point = 5000
    generated_goals = np.empty((0, 12))
    for counter in range(num_point):
        env.task.reset()
        obs = env.task.get_obs()
        generated_goals = np.vstack((generated_goals, obs[:12]))
    return generated_goals

def generate_ori(env):
    """
    generate 5250 points, 6 columns for target position and orientation
    """
    low = env.task.goal_range_low
    high = env.task.goal_range_high
    # Calculate the number of steps for each dimension
    num_steps = [int((high[i] - low[i]) / 0.05) + 1 for i in range(len(low))]
    generated_goals = np.empty((0, 6))
    # Iterate through each point in the range with 0.05 increments
    for i in range(num_steps[0]):
        for j in range(num_steps[1]):
            for k in range(num_steps[2]):
                for w in range(5):
                    env.reset()
                    data_temp = np.zeros((1, 6))
                    data_temp[0, 0] = i / 20 + env.task.goal_range_low[0]
                    data_temp[0, 1] = j / 20 + env.task.goal_range_low[1]
                    data_temp[0, 2] = k / 20 + env.task.goal_range_low[2]
                    data_temp[0, 3:] = env.task.get_goal()[3:]
                    # stack data
                    generated_goals = np.vstack((generated_goals, data_temp))
    return generated_goals

def generate_obs(env):
    """
    generate 5000 points, first 3 columns for target position,
    last 6 columns for obstacle position and orientation
    """
    num_point = 5000
    generated_goals = np.empty((0, 9))
    for counter in range(num_point):
        env.task.reset()
        obs = env.task.get_obs()
        generated_goals = np.vstack((generated_goals, obs[:9]))
    return generated_goals

if __name__ == "__main__":
    env_name = "UR5DynReach-v1"
    env = gymnasium.make(env_name, render=True)
    result = generate(env, env_name)

