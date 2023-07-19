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


def test(points):
    # ---------------- Create environment
    env = gymnasium.make("UR5ObsReach-v1", render=True)

    # ----------------- Load the pre-trained model from files
    print("load the pre-trained model from files")
    model_path = "RobotLearn2/SAC_Env13/"
    model = SAC.load(model_path + "best_model", env=env)
    obs = env.reset()

    num_steps = np.zeros(points.shape[0])
    success = np.zeros(points.shape[0])
    rewards = np.zeros(points.shape[0])

    for trials in tqdm(range(success.size)):
        env.task.set_goal(points[trials, :])
        obs = env.reset()
        for steps in range(100):
            action, _states = model.predict(obs[0], deterministic=True)
            obs = env.step(action)
            env.render()
            rewards[trials] += obs[1]
            if steps == 99 or obs[2]:
                success[trials] = obs[4]['is_success']
                num_steps[trials] = steps
                break
    # time.sleep(3) # for viewing clearly
    env.close()
    success_rate = (np.sum(success) / success.size) * 100
    avg_reward = (np.sum(rewards) / rewards.size)
    print("The success rate is {}%".format(success_rate))
    print("The average reward is {}".format(avg_reward))
    with open(model_path + 'best.txt', "w") as f:
        f.write("The success rate is {}%\n".format(success_rate))
        f.write("The average reward is {}\n".format(avg_reward))
        for num1, num2, num3 in zip(rewards, success, num_steps):
            f.writelines(str(num1) + ' ,' + str(num2) + ' ,' + str(num3) + '\n')
    f.close()


if __name__ == "__main__":
    points = np.loadtxt('testset_normal.txt')
    test(points)
