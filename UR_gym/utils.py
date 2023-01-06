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
    if len(a) == 7:
        return np.linalg.norm(a[:3] - b[:3], axis=-1)
    else:
        return np.array([np.linalg.norm(row[0] - row[1], axis=-1) for row in zip(a[:, :3], b[:, :3])])


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First quaternion array in (real, i, j, k) format.
        b (np.ndarray): Second quaternion array in (real, i, j, k) format.

    Returns:
        np.ndarray: The normalised distance between the angles.
    """
    assert a.shape == b.shape
    if len(a) == 7:
        return 2 * np.arccos(np.abs(np.sum(np.dot(a[3:], b[3:])))) / np.pi
    else:
        return np.array([2 * np.arccos(np.abs(np.sum(np.dot(row[0], row[1])))) / np.pi for row in zip(a[:, 3:], b[:, 3:])])
