import os
from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

register(
    id="UR5IAIReach-v1",
    entry_point="UR_gym.envs:UR5IAIReachEnv",
    max_episode_steps=100,
)

register(
    id="UR5RegReach-v1",
    entry_point="UR_gym.envs:UR5RegReachEnv",
    max_episode_steps=100,
)

register(
    id="UR5OriReach-v1",
    entry_point="UR_gym.envs:UR5OriReachEnv",
    max_episode_steps=100,
)

register(
    id="UR5ObsReach-v1",
    entry_point="UR_gym.envs:UR5ObsReachEnv",
    max_episode_steps=100,
)
