import gymnasium as gym
import numpy as np

import UR_gym


def test_seed_reach():
    final_observations = []
    env = gym.make("UR5Reach-v1")
    actions = [
        np.array([-0.931, 0.979, -0.385]),
        np.array([-0.562, 0.391, -0.532]),
        np.array([0.042, 0.254, -0.624]),
        np.array([0.465, 0.745, 0.284]),
        np.array([-0.237, 0.995, -0.425]),
        np.array([0.67, 0.472, 0.972]),
    ]
    for _ in range(2):
        env.reset(seed=12345)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)

    assert np.allclose(final_observations[0]["observation"], final_observations[1]["observation"])
    assert np.allclose(final_observations[0]["achieved_goal"], final_observations[1]["achieved_goal"])
    assert np.allclose(final_observations[0]["desired_goal"], final_observations[1]["desired_goal"])
