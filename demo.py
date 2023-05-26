import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym

env = gymnasium.make('UR5OriReach-v1', render="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()