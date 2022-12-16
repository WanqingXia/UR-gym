import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym
from stable_baselines3 import SAC, DDPG, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
import signal
import time
from UR_gym.utils import distance, angle_distance
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def sig_handler(signal, frame):
    env.close()
    print("Existing Program...")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)
# ---------------- Create environment
env = gymnasium.make("UR5OriReach-v1", render=True)

# ----------------- Load the pre-trained model from files
print("load the pre-trained model from files")
model_path = "RobotLearn/UR5_SAC_Ori/"
model = SAC.load(model_path + "best_model", env=env)

# ------------------ Evaluate the policy
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, render=True)
# print('Mean Reward: ', mean_reward, 'Std Reward: ', std_reward)

env = model.get_env()
obs = env.reset()
num_trials = 10
logs = np.zeros((4, num_trials))
for trials in tqdm(range(num_trials)):
    for steps in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()
        desired = obs['desired_goal'].flatten()
        achieved = obs['achieved_goal'].flatten()
        if steps == 99 or done:
            obs = env.reset()
            what = distance(achieved[:3], desired[:3])
            logs[0, trials] = distance(achieved[:3], desired[:3])
            where = angle_distance(achieved[3:], desired[3:])
            logs[1, trials] = angle_distance(achieved[3:], desired[3:])
            logs[2, trials] = steps
            logs[3, trials] = info[0]['is_success']
            break
env.close()

success_rate = np.sum(logs[3, :])/num_trials
np.savetxt(model_path + 'eval_result.txt', logs.transpose(), fmt='%s', header="Trans_Dis Ori_Dis steps success")

print("success rate is: ", success_rate, ". Log file saved to the model directory.")

