import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym


def generate_obs():
    """
    generate 5000 points, first 3 columns for target position,
    last 6 columns for obstacle position and orientation
    """

    # ------------- Create environment------------
    env = gymnasium.make("UR5ObsReach-v1", render=True)

    save_goals = np.zeros((5000, 9))
    for counter in range(save_goals.shape[0]):
        env.task.reset()
        obs = env.task.get_obs()
        save_goals[counter, :3] = obs[:3]
        save_goals[counter, 3:] = obs[3:9]
    np.savetxt("testset_obs.txt", save_goals)


if __name__ == "__main__":
    generate_obs()
