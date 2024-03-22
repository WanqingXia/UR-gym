import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym
from stable_baselines3 import SAC
from tqdm import tqdm
from utils.generate import generate

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


def run_test(env, env_name, model, points):
    obs = env.reset()
    num_steps = np.zeros(points.shape[0])
    success = np.zeros(points.shape[0])
    rewards = np.zeros(points.shape[0])

    for trials in tqdm(range(success.size)):
        obs = env.reset()
        if env_name == "UR5OriReach-v1":
            env.task.set_goal(points[trials, :])
        else:
            env.task.set_goal_and_obstacle(points[trials, :])
        obs = get_obs(env)

        for steps in range(100):
            # time.sleep(0.04)
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
    with open(model_path + 'test_result.txt', "w") as f:
        f.write("The success rate is {}%\n".format(success_rate))
        f.write("The average reward is {}\n".format(avg_reward))
        for num1, num2, num3 in zip(rewards, success, num_steps):
            f.writelines(str(num1) + ' ,' + str(num2) + ' ,' + str(num3) + '\n')
    f.close()

if __name__ == "__main__":
    # Available environments: UR5OriReach-v1, UR5ObsReach-v1, UR5StaReach-v1, UR5DynReach-v1
    env_name = "UR5DynReach-v1"
    env = gymnasium.make(env_name, render=True)
    test_points = generate(env, env_name)
    if test_points is None:
        print("No test points are generated. Please check the environment name.")
        exit(1)
    # choose the trained model
    model_path = "./Trained_Models/Trained_Dyn/best_model"
    model = SAC.load(model_path + "best_model", env=env)
    run_test(env, env_name, model, test_points)

