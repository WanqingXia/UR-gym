import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym
from stable_baselines3 import SAC
import time
from tqdm import tqdm


# The original get_obs() function in core.py is prohibited to use, so we need to define a new one
def get_obs(env):
    robot_obs = env.robot.get_obs().astype(np.float32)  # robot state
    task_obs = env.task.get_obs().astype(np.float32)  # object position, velocity, etc...
    observation = np.concatenate([robot_obs, task_obs])
    achieved_goal = env.task.get_achieved_goal().astype(np.float32)
    obs = {
        "observation": observation,
        "achieved_goal": achieved_goal,
        "desired_goal": env.task.get_goal().astype(np.float32),
    }

    info = {"is_success": env.task.is_success(obs["achieved_goal"], env.task.get_goal())}
    return obs, info


def show_traj(points):
    # ---------------- Create environment
    env = gymnasium.make("UR5DynReach-v1", render=True)

    # ----------------- Load the pre-trained model from files
    print("load the pre-trained model from files")
    model_path = "./RobotLearn/Dyn_train6/"
    model = SAC.load(model_path + "best_model", env=env)
    obs = env.reset()

    for trials in tqdm(range(1, 11)):
        arr = np.empty((0, 35))
        obs = env.reset()
        env.task.set_goal_and_obstacle(points[trials, :])
        obs = get_obs(env)
        # stack arr vertically
        arr = np.vstack((arr, obs[0]['observation']))

        for steps in range(100):
            # time.sleep(0.04)
            action, _states = model.predict(obs[0], deterministic=True)
            obs = env.step(action)
            arr = np.vstack((arr, obs[0]['observation']))
            env.render()
            if steps == 99 or obs[2]:
                break
        np.savetxt(model_path + 'dyn_traj' + str(trials) + '.txt', arr, fmt='%f')
    # time.sleep(3) # for viewing clearly
    env.close()


    # ---------------- Create environment
    env = gymnasium.make("UR5OriReach-v1", render=True)

    # ----------------- Load the pre-trained model from files
    print("load the pre-trained model from files")
    model_path = "./Trained_Ori/"
    model = SAC.load(model_path + "best_model", env=env)
    obs = env.reset()

    for trials in tqdm(range(1, 11)):
        arr = np.empty((0, 18))
        obs = env.reset()
        env.task.set_goal(points[trials, :6])
        obs = get_obs(env)
        # stack arr vertically
        arr = np.vstack((arr, obs[0]['observation']))

        for steps in range(100):
            # time.sleep(0.04)
            action, _states = model.predict(obs[0], deterministic=True)
            obs = env.step(action)
            arr = np.vstack((arr, obs[0]['observation']))
            env.render()
            if steps == 99 or obs[2]:
                break
        np.savetxt(model_path + 'ori_traj' + str(trials) + '.txt', arr, fmt='%f')
    # time.sleep(3) # for viewing clearly
    env.close()


if __name__ == "__main__":
    show_traj(np.loadtxt('testset_dyn.txt'))
