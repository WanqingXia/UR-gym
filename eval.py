import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym
from stable_baselines3 import SAC, DDPG, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
import signal

from scipy.spatial.transform import Rotation as R

def sig_handler(signal, frame):
    env.close()
    print("Existing Program...")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)
# ---------------- Create environment
env = gymnasium.make("UR5IAIOriReachJointsDense-v1", render=True)

# ----------------- Load the pre-trained model from files
print("load the pre-trained model from files")
model = DDPG.load("RobotLearn/UR5_Ori/best_model", env=env)

# ------------------ Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, render=True)
print('Mean Reward: ', mean_reward, 'Std Reward: ', std_reward)

env.close()
