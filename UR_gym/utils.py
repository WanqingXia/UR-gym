import numpy as np
from scipy.spatial.transform import Rotation as R


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    """Computes the L2 norm of the difference between each pair of elements."""
    if len(a.shape) == 1:
        # one dimensional array
        a = a[:3]
        b = b[:3]
    else:
        # multi dimensional array
        a = a[:, :3]
        b = b[:, :3]
    diff = a - b
    squared_diff = diff ** 2

    # sum up all squares
    dist = np.sqrt(squared_diff.sum(-1))
    # Reshaping to have shape (n,)
    return dist.reshape(-1)


def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the angular distance between two arrays representing Euler angles.
    This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The angular distance between the Euler angles.
    """
    assert a.shape == b.shape

    if len(a.shape) == 1:  # If a and b are 1D arrays
        a_rot = R.from_euler('xyz', a[3:])
        b_rot = R.from_euler('xyz', b[3:])
    else:  # If a and b are multi-dimensional arrays
        a_rot = R.from_euler('xyz', a[:, 3:])
        b_rot = R.from_euler('xyz', b[:, 3:])

    quaternion_a = a_rot.as_quat()
    quaternion_b = b_rot.as_quat()

    dot_product = np.sum(quaternion_a * quaternion_b, axis=-1)

    # Clamp the dot_product values to the range [-1, 1]
    clamped_dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angular distance
    angular_dist = 2 * np.arccos(np.abs(clamped_dot_product))

    if np.isnan(angular_dist).any():
        print("NaN values encountered!")
        print(dot_product)
    # Reshaping to have shape (n,)
    return angular_dist.reshape(-1)


def sample_euler():

    # Generate a random rotation matrix
    rot_mat = R.random()
    # Convert the rotation matrix to Euler angles
    euler_angles = rot_mat.as_euler('xyz', degrees=False)
    # Print the Euler angles
    return euler_angles


def euler_to_quaternion(euler_angles):
    quat = R.from_euler('XYZ', euler_angles).as_quat()
    return np.roll(quat, 1)

