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
from ur_ikfast import ur_kinematics
from UR_gym.utils import distance, angle_distance


def sig_handler(signal, frame):
    print("Existing Program...")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)


def test_RLmodel(points):

    # ---------------- Create environment
    env = gymnasium.make("UR5OriReach-v1", render=True)

    # ----------------- Load the pre-trained model from files
    print("load the pre-trained model from files")
    model_path = "RobotLearn/SAC_Ori_HER3/"
    model = SAC.load(model_path + "best_model", env=env)
    env.task.set_reward()

    # env = model.get_env()
    obs = env.reset()
    success = np.zeros(points.shape[0])
    rewards = np.zeros(points.shape[0])

    for trials in tqdm(range(success.size)):
        env.task.set_goal(points[trials, :])
        obs = env.reset()
        for steps in range(100):
            action, _states = model.predict(obs[0], deterministic=True)
            obs = env.step(action)
            # env.render()
            rewards[trials] += obs[1]
            if steps == 99 or obs[2]:
                success[trials] = obs[4]['is_success']
                break
    env.close()
    success_rate = (np.sum(success) / success.size) * 100
    avg_reward = (np.sum(rewards) / rewards.size)
    print("The success rate is {}%".format(success_rate))
    print("The average reward is {}".format(avg_reward))
    with open('best.txt', "w") as f:
        f.write("The success rate is {}%\n".format(success_rate))
        f.write("The average reward is {}\n".format(avg_reward))
        for num in rewards:
            f.writelines(str(num) + '\n')
    f.close()


def generate_ideal(points):
    rewards = np.zeros(points.shape[0])
    ur5e = ur_kinematics.URKinematics('ur5e')
    initial = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])

    for trials in tqdm(range(rewards.size)):
        joints = initial.copy()
        angles = ur5e.inverse((points[trials, :]), False)
        if angles is None:
            print("found one broken point")
            continue
        while not np.array_equal(joints, angles):
            for num in range(joints.size):
                if joints[num] - angles[num] > 0.3:
                    joints[num] -= 0.3
                elif joints[num] - angles[num] < -0.3:
                    joints[num] += 0.3
                else:
                    joints[num] = angles[num].copy()
            end_effector = ur5e.forward(joints)
            d = distance(end_effector.astype(np.float32), points[trials, :].astype(np.float32))
            dr = angle_distance(end_effector.astype(np.float32), points[trials, :].astype(np.float32))

            rewards[trials] += np.abs(d) * -10
            rewards[trials] += np.abs(dr) * -10

    avg_reward = (np.sum(rewards) / rewards.size)
    print("The average reward is {}".format(avg_reward))
    with open('ideal.txt', "w") as f:
        f.write("The average reward is {}\n".format(avg_reward))
        for reward in rewards:
            f.writelines(str(reward) + '\n')
    f.close()


if __name__ == "__main__":
    points = np.loadtxt('test_set.txt')
    test_RLmodel(points)
    # generate_ideal(points)
