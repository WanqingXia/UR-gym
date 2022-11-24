import gymnasium as gym

import UR_gym


def run_env(env):
    """Tests running panda gym envs."""
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()


def test_reach():
    env = gym.make("UR5Reach-v1")
    run_env(env)


def test_dense_reach():
    env = gym.make("UR5ReachDense-v1")
    run_env(env)


def test_dense_reach_joints():
    env = gym.make("UR5ReachJointsDense-v1")
    run_env(env)

