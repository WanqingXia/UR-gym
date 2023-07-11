import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC, HerReplayBuffer, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
# from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from utils.callbackFunctions import EvalCallback
from typing import Callable
import UR_gym
import os
import numpy as np
from datetime import datetime
import signal
import wandb

epochs = 2000000
learning_rate = 1e-4
gamma = 0.95

wandb.init(
    # set the wandb project where this run will be logged
    project="RL_Obs",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "SAC",
        "epochs": epochs,
    }
)


# def linear_schedule(initial_value: float) -> Callable[[float], float]:
#     """
#     Cosine learning rate schedule.
#
#     :param initial_value: Initial learning rate.
#     :return: schedule that computes
#       current learning rate depending on remaining progress
#       warm-up from 1e-5 to 1e-3, then decrease to 1e-5
#     """
#     def func(progress_remaining: float) -> float:
#         """
#         Progress will decrease from 1 (beginning) to 0.
#
#         :param progress_remaining:
#         :return: current learning rate
#         """
#
#         return 0.99 * initial_value * progress_remaining + 0.01 * initial_value
#
#     return func


def sig_handler(signal, frame):
    env.close()
    print("Existing Program...")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)

# ---------------- Load model and continue training

# timesteps = 10000
# env = gymnasium.make("UR5ObsReach-v1", render=True)
# check_env(env, warn=True)
# model = SAC.load("./RobotLearn2/SAC_continue/best_model", env=env)
# log_dir = "./RobotLearn2/SAC_continue"
# env = Monitor(env, log_dir)


# ---------------- Training from scratch

env = gymnasium.make("UR5ObsReach-v1", render=True)
check_env(env, warn=True)

model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    buffer_size=int(1e7),
    learning_rate=learning_rate,
    gamma=gamma,
    batch_size=256,
)

log_dir = "./RobotLearn2/" + "SAC_NewEnv13"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# ---------------- Callback functions
callback_save_best_model = EvalCallback(wandb, env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000,
                                        deterministic=True, n_eval_episodes=10, render=False)
callback_list = CallbackList([callback_save_best_model])


# ---------------- Start Training
model.learn(total_timesteps=epochs, callback=callback_list)
env.close()
