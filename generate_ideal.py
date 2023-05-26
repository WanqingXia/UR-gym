import numpy as np
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import signal
from tqdm import tqdm
from ur_ikfast import ur_kinematics
from UR_gym.utils import distance, angle_distance


def sig_handler(signal, frame):
    print("Existing Program...")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)


def generate_ideal(points):
    rewards = np.zeros(points.shape[0])
    ur5e = ur_kinematics.URKinematics('ur5e')
    initial = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])

    for trials in tqdm(range(rewards.size)):
        joints = initial.copy()
        angles = ur5e.inverse((points[trials, :]), False)
        if angles is None or np.max(np.abs(angles)) > 6.28:
            print("found one broken point")
            continue
        while not np.array_equal(joints, angles):
            for num in range(joints.size):
                if joints[num] - angles[num] > 0.3:
                    joints[num] -= 0.3
                elif joints[num] - angles[num] < -0.3:
                    joints[num] += 0.3
                else:
                    joints[num] = angles[num].copy()
            end_effector = ur5e.forward(joints)
            d = distance(end_effector.astype(np.float32), points[trials, :].astype(np.float32))
            dr = angle_distance(end_effector.astype(np.float32), points[trials, :].astype(np.float32))

            rewards[trials] += np.abs(d) * -10
            rewards[trials] += np.abs(dr) * -10

    avg_reward = (np.sum(rewards) / rewards.size)
    print("The average reward is {}".format(avg_reward))
    with open('ideal.txt', "w") as f:
        f.write("The average reward is {}\n".format(avg_reward))
        for reward in rewards:
            f.writelines(str(reward) + '\n')
    f.close()


if __name__ == "__main__":
    points = np.loadtxt('testset.txt')
    generate_ideal(points)
