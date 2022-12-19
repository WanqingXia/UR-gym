import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC, HerReplayBuffer, DDPG
from stable_baselines3.sac.policies import MultiInputPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from utils.callbackFunctions import VisdomCallback
from typing import Callable
import UR_gym
import os
import numpy as np
from datetime import datetime
import signal


def sig_handler(signal, frame):
    env.close()
    print("Existing Program...")
    sys.exit(0)

def cos_schedule(initial_value: float, total_timestep: int, warmup_step: int) -> Callable[[float], float]:
    """
    Cosine learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
      warm-up from 1e-5 to 1e-3, then decrease to 1e-5
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        current_step = (1 - progress_remaining) * total_timestep + 1
        if current_step <= warmup_step:
            return (current_step / (warmup_step/100)) * (initial_value / 100)
        else:
            return 0.5 * initial_value * (1 + np.cos(np.pi * (current_step - warmup_step) / float(1000000 - warmup_step)))

    return func


signal.signal(signal.SIGINT, sig_handler)

# ---------------- Load model and log
# model = SAC.load("./RobotLearn/SAC_Ori/best_model", env=env)
# log_dir = "./RobotLearn/SAC_Ori"
# env = Monitor(env, log_dir)


# ---------------- Train

# ---------------- Create environment
timesteps = 2000000
env = gymnasium.make("UR5OriReach-v1", render=True)

# ---------------- Create model and log
model = SAC(MultiInputPolicy, learning_rate=1e-4, gamma=0.95, env=env, verbose=1)
log_dir = "./RobotLearn/" + "SAC_Ori_new4"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# ---------------- Callback functions
callback_visdom = VisdomCallback(name='UR-gym', check_freq=10, log_dir=log_dir)
callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000,
                                        deterministic=True, n_eval_episodes=10, render=False)
callback_list = CallbackList([callback_visdom, callback_save_best_model])

model.learn(total_timesteps=timesteps, callback=callback_list)
# model.learn(total_timesteps=500000)
env.close()
