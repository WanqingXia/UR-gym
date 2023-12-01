import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC, HerReplayBuffer, DDPG, TD3, A2C
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env
from utils.callbackFunctions import EvalCallback
from typing import Callable
import UR_gym
import os
import numpy as np
from datetime import datetime
import signal
import wandb

epochs = 4000000
learning_rate = 1e-4
gamma = 0.95

wandb.init(
    # set the wandb project where this run will be logged
    project="RL_Obs",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "TD3",
        "epochs": epochs,
    }
)

# ---------------- Load model and continue training

# env = gymnasium.make("UR5DynReach-v1", render=True)
# # check_env(env, warn=True)
# model = SAC.load("./RobotLearn/Dyn_train10con/best_model", env=env)
# log_dir = "./RobotLearn/Dyn_train10con"
# env = Monitor(env, log_dir)


# ---------------- Training from scratch

env = gymnasium.make("UR5DynReach-v1", render=True)
# check_env(env, warn=True)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(
    "MultiInputPolicy",
    env,
    verbose=1,
    # buffer_size=int(1e7),
    action_noise=action_noise,
    learning_rate=learning_rate,
    gamma=gamma,
    # batch_size=256,
)

log_dir = "./RobotLearn/" + "TD3"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
#
# # ---------------- Callback functions
callback_save_best_model = EvalCallback(wandb, env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000,
                                        deterministic=True, n_eval_episodes=100, render=False)
callback_list = CallbackList([callback_save_best_model])


# ---------------- Start Training
model.learn(total_timesteps=epochs, callback=callback_list)
env.close()
