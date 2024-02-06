import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym


def generate_dyn():
    """
    generate 5000 points, first 6 columns for target position and orientation,
    middle 6 columns for obstacle start position and orientation,
    last 6 columns for obstacle end position and orientation,
    """

    # ------- Create environment----------
    env = gymnasium.make("UR5DynReach-v1", render=True)

    low = env.task.goal_range_low
    high = env.task.goal_range_high

    # Calculate the number of steps for each dimension
    num_steps = [int((high[i] - low[i]) / 0.05) + 1 for i in range(len(low))]

    # Create an empty array with 18 columns
    save_goals = np.empty((0, 18))

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
                    save_goals = np.vstack((save_goals, data_temp))

    # Convert the list to a numpy array
    np.savetxt("testset_dyn.txt", save_goals)


if __name__ == "__main__":
    generate_dyn()
