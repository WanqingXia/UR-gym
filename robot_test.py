from UR_gym.envs.robots.UR5 import UR5
from UR_gym.pyb_setup import PyBullet
import numpy as np
import time

sim = PyBullet(render=True)
robot = UR5(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type='joints')

for _ in range(50):
    robot.set_action(np.array([-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]))
    sim.step()
    time.sleep(1)
    sim.render()
    robot.set_action(np.array([-1.3690622952052096, -1.3446774605904932, 1.043946009733127, -1.0708613585093699,
                               -1.3707970583733368, 0.0009377758247187636]))
    sim.step()
    time.sleep(1)
    sim.render()
