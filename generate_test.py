import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym
from stable_baselines3 import SAC
import signal
import time
from tqdm import tqdm


def sig_handler(signal, frame):
    print("Existing Program...")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)


def generate_hard():
    # ---------------- Create environment
    env = gymnasium.make("UR5ObsReach-v1", render=True)

    # ----------------- Load the pre-trained model from files
    print("load the pre-trained model from files")
    model_path = "RobotLearn2/SAC_NewEnv5/"
    model = SAC.load(model_path + "best_model", env=env)

    save_goals = np.zeros((1000, 10))
    for num in tqdm(range(save_goals.shape[0])):
        success = True
        obs_save = np.zeros(10)
        while success:
            obs = env.reset()
            for steps in range(100):
                action, _states = model.predict(obs[0], deterministic=True)
                obs = env.step(action)
                env.render()
                if steps == 99 or obs[2]:
                    success = obs[4]['is_success']
                    obs_save = obs[0]['observation'][13:23]
                    break
        save_goals[num, :] = obs_save
    np.savetxt("testset_hard.txt", save_goals)


def generate_normal():
    # enable this function in core.py to generate points
    # 5000 points, first 3 columns for target position, last 7 columns for obstacle position and orientation

    # ---------------- Create environment
    env = gymnasium.make("UR5ObsReach-v1", render=True)

    save_goals = np.zeros((5000, 10))
    for counter in range(save_goals.shape[0]):
        env.task.reset()
        obs = env.task.get_obs()
        save_goals[counter, :3] = obs[:3]
        save_goals[counter, 3:] = obs[3:10]
    np.savetxt("testset_normal.txt", save_goals)


if __name__ == "__main__":
    generate_normal()
    # generate_hard()
