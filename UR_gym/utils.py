import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First quaternion array in (real, i, j, k) format.
        b (np.ndarray): Second quaternion array in (real, i, j, k) format.

    Returns:
        np.ndarray: The normalised distance between the angles.
    """
    assert a.shape == b.shape
    angle = 2 * np.arccos(np.abs(np.sum(np.dot(a, b))))
    return angle / np.pi

    # code useful for HER
    # if len(a) == 4:
    #     dist = 1 - np.inner(a, b) ** 2
    #     return dist
    # else:
    #     dist = [1 - np.inner(row[0], row[1]) ** 2 for row in zip(a, b)]
    #     return np.array(dist)
