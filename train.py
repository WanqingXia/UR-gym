import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC, HerReplayBuffer, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from utils.callbackFunctions import VisdomCallback
import UR_gym
import os
from datetime import datetime
import signal


def sig_handler(signal, frame):
    env.close()
    print("Existing Program...")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)

# ---------------- Create environment
env = gymnasium.make("UR5RegReachJointsDense-v1", render=True)

# ---------------- Create model and log
model = SAC(policy="MultiInputPolicy", env=env, verbose=1)
log_dir = "./RobotLearn/" + datetime.now().strftime("reg_%m_%d_%H:%M")
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# ---------------- Load model and log
# model = SAC.load("./RobotLearn/UR5_SAC_Ori/best_model", env=env)
# log_dir = "./RobotLearn/UR5_SAC_Ori"
# env = Monitor(env, log_dir)


# ---------------- Callback functions
callback_visdom = VisdomCallback(name='UR-gym', check_freq=10, log_dir=log_dir)
callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000,
                                        deterministic=True, n_eval_episodes=100, render=False)
callback_list = CallbackList([callback_visdom, callback_save_best_model])

# ---------------- Train
model.learn(total_timesteps=200000, callback=callback_list)
# model.learn(total_timesteps=500000)
