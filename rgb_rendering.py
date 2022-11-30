import sys
import gymnasium
sys.modules["gym"] = gymnasium
# from numpngw import write_apng  # pip install numpngw
import UR_gym
import time


env = gymnasium.make("UR5IAIOriReachJointsDense-v1", render=True)
images = []


observation, info = env.reset()
images.append(env.render("rgb_array"))

time.sleep(20)
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     images.append(env.render("rgb_array"))
#
#     if terminated or truncated:
#         observation, info = env.reset()
#         images.append(env.render("rgb_array"))

env.close()

# write_apng("reach.png", images, delay=40)
