"""Reconstruction util functions."""
import sys

sys.path.append("..")
from typing import Tuple

import numpy as np


def remove_noise_rays(
    data: np.ndarray,
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    traj_z: np.ndarray,
    snr_threshold: float = 0.7,
    tail: float = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove noisy FID rays in the k space data.

    Remove noisy FIDs in the kspace data and their corresponding trajectories.

    Args:
        data (np.ndarray): k space datadata of shape (K, 1)
        x (np.ndarray): x coordinates in trajectory of shape (K, 1)
        y (np.ndarray): y coordinates in trajectory of shape (K, 1)
        z (np.ndarray): z coordinates in trajectory of shape (K, 1)
        thre_snr (float): threshold SNR value
        tail (float, optional): Index to define the tail of FID. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]: Tuple of the data with the
            noisy FIDs removed, x, y, and z k-space coordinates with noisy data removed,
            and the number of FIDs remaining after removal.
    """
    thre_dis = snr_threshold * np.average(abs(data[:, :5]))
    max_tail = np.amax(abs(data[:, tail:]), axis=1)
    good_index = max_tail < thre_dis

    return data[good_index], traj_x[good_index], traj_y[good_index], traj_z[good_index]


def flatten_data(data: np.ndarray) -> np.ndarray:
    """Flatten data for reconstruction.

    Args:
        data (np.ndarray): data of shape (n_projections, n_points)

    Returns:
        np.ndarray: flattened data of shape (n_projections * n_points, 1)
    """
    return data.reshape((data.shape[0] * data.shape[1], 1))


def flatten_traj(traj: np.ndarray) -> np.ndarray:
    """Flatten trajectory for reconstruction.

    Args:
        traj (np.ndarray): trajectory of shape (n_projections, n_points, 3)
    Returns:
        np.ndarray: flattened trajectory of shape (n_projections * n_points, 3)
    """
    return traj.reshape((traj.shape[0] * traj.shape[1], 3))
