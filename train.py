import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from utils.callbackFunctions import EvalCallback
import UR_gym
import os
import wandb

epochs = 4000000
learning_rate = 1e-4
gamma = 0.95
# Available environments: UR5IAIReach-v1, UR5RegReach-v1, UR5OriReach-v1, UR5ObsReach-v1, UR5StaReach-v1, UR5DynReach-v1
environment = "UR5DynReach-v1"

wandb.init(
    # set the wandb project where this run will be logged
    project="RL_Obs",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "SAC",
        "epochs": epochs,
        "environment": environment
    }
)

# ---------------- Load model and continue training
# # Make sure to check the environment is same as the model
# env = gymnasium.make(environment, render=True)
# model = SAC.load("./Trained_Models/Trained_Dyn/best_model", env=env)
# log_dir = "./RobotLearn/Dyn_train"
# env = Monitor(env, log_dir)

# ---------------- Training from scratch
env = gymnasium.make(environment, render=True)
model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    buffer_size=int(1e7),
    learning_rate=learning_rate,
    gamma=gamma,
    batch_size=256,
)

log_dir = "./RobotLearn/" + environment
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# ---------------- Callback functions
callback_save_best_model = EvalCallback(wandb, env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000,
                                        deterministic=True, n_eval_episodes=100, render=False)
callback_list = CallbackList([callback_save_best_model])

# ---------------- Start Training
model.learn(total_timesteps=epochs, callback=callback_list)
env.close()
