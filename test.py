import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym
from stable_baselines3 import SAC
import signal
import time
from UR_gym.utils import distance, angle_distance
from tqdm import tqdm

def sig_handler(signal, frame):
    print("Existing Program...")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)


def test_RLmodel(points):
    # ---------------- Create environment
    env = gymnasium.make("UR5OriReach-v1", render=True)

    # ----------------- Load the pre-trained model from files
    print("load the pre-trained model from files")
    model_path = "RobotLearn/SAC_Ori_t1/"
    model = SAC.load(model_path + "best_model", env=env)

    # env = model.get_env()
    obs = env.reset()
    success = np.zeros(points.shape[0])
    for trials in tqdm(range(success.size)):
        env.task.set_goal(points[trials, :])
        obs = env.reset()
        for steps in range(100):
            action, _states = model.predict(obs[0], deterministic=True)
            obs = env.step(action)
            # env.render()
            if steps == 99 or obs[2]:
                success[trials] = obs[4]['is_success']
                break
    env.close()
    success_rate = (np.sum(success) / success.size) * 100
    print("The success rate is {}%".format(success_rate))


if __name__ == "__main__":
    points = np.loadtxt('test_set.txt')
    test_RLmodel(points)
