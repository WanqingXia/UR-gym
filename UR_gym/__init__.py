import os
from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="UR5Reach{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="UR_gym.envs:UR5ReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="UR5IAIReach{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="UR_gym.envs:UR5IAIReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="UR5IAIRegReach{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="UR_gym.envs:UR5IAIRegReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
