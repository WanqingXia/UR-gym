import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym


def generate_ori():
    env = gymnasium.make("UR5OriReach-v1", render=True)
    # enable this function in core.py to generate points
    goal_range = env.task.goal_range_high - env.task.goal_range_low
    rows = int(round((goal_range[0] * 20 + 1) * (goal_range[1] * 20 + 1) * (goal_range[2] * 20 + 1) * 5))
    save_goals = np.zeros((rows, 6))
    counter = 0
    for i in range(int(round((goal_range[0] * 20 + 1)))):
        for j in range(int(round(goal_range[1] * 20 + 1))):
            for k in range(int(round(goal_range[2] * 20 + 1))):
                for w in range(5):
                    env.reset()
                    save_goals[counter, 0] = i / 20 + env.task.goal_range_low[0]
                    save_goals[counter, 1] = j / 20 + env.task.goal_range_low[1]
                    save_goals[counter, 2] = k / 20 + env.task.goal_range_low[2]
                    save_goals[counter, 3:] = env.task.get_goal()[3:]
                    counter += 1

    np.savetxt("testset_ori.txt", save_goals)


if __name__ == "__main__":
    generate_ori()
