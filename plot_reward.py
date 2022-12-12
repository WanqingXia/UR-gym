import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def show_reward_tar_obs():
    # Create a mesh grid with x and y ranging from 0 to 1 with a 0.01 step size
    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)
    xx, yy = np.meshgrid(x, y)
    delta = 0.1

    # Calculate reward based on distance to target and distance to obstacle
    reward = np.where(xx < 0.1, -100 * 0.5 * np.square(xx) + np.power((0.2 / (yy + 0.2)), 8) * -6,
                      -100 * delta * (np.abs(xx) - 0.5 * delta) + np.power((0.2 / (yy + 0.2)), 8) * -6)

    # Plot the 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, reward, cmap='plasma')
    ax.set_xlabel("Distance to target")
    ax.set_ylabel("Distance to obstacle")
    ax.set_zlabel("Reward")

    plt.show()


def show_reward_dis_ori():
    # create meshgrid
    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)
    X, Y = np.meshgrid(x, y)

    # calculate z values
    # using 0.1 as margin for both distance and orientation
    x_coe = -160
    y_coe = -40
    z1 = x_coe * 0.5 * np.square(X) + y_coe * 0.1 * (np.abs(Y) - 0.5 * 0.1)
    z2 = x_coe * 0.1 * (np.abs(X) - 0.5 * 0.1) + y_coe * 0.5 * np.square(Y)
    z3 = x_coe * 0.5 * np.square(X) + y_coe * 0.5 * np.square(Y)
    z4 = x_coe * 0.1 * (np.abs(X) - 0.5 * 0.1) + y_coe * 0.1 * (np.abs(Y) - 0.5 * 0.1)

    # create mask for different conditions
    mask1 = (X <= 0.1) & (Y > 0.1)
    mask2 = (Y <= 0.1) & (X > 0.1)
    mask3 = (X <= 0.1) & (Y <= 0.1)
    mask4 = (X > 0.1) & (Y > 0.1)

    # set z values based on conditions
    Z = np.zeros_like(X)
    Z[mask1] = z1[mask1]
    Z[mask2] = z2[mask2]
    Z[mask3] = z3[mask3]
    Z[mask4] = z4[mask4]

    # plot 3d graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_xlabel("Translation distance")
    ax.set_ylabel("Orientation distance")
    ax.set_zlabel("Reward")
    plt.show()


if __name__ == "__main__":
    show_reward_tar_obs()
    # show_reward_dis_ori()
