import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym


def generate_sta():
    """
    generate 5000 points, first 6 columns for target position and orientation,
    last 6 columns for obstacle position and orientation
    """

    # ------- Create environment----------
    env = gymnasium.make("UR5StaReach-v1", render=True)

    save_goals = np.zeros((5000, 12))
    for counter in range(save_goals.shape[0]):
        env.task.reset()
        obs = env.task.get_obs()
        save_goals[counter, :6] = obs[:6]
        save_goals[counter, 6:] = obs[6:12]
    np.savetxt("testset_sta.txt", save_goals)


if __name__ == "__main__":
    generate_sta()
