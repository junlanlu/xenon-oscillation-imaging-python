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


def complex_align(x: np.ndarray) -> np.ndarray:
    """Flip and take transpose of image volume.

    This may be needed to make the image be in the coronal direction.
    Args:
        x (np.ndarray): image volume

    Returns:
        np.ndarray: Flipped and transposed image volume.
    """
    return np.flip(np.flip(np.flip(np.transpose(x, (2, 1, 0)), 0), 1), 2)


def alignrot(x: np.ndarray) -> np.ndarray:
    """Flip image volume.

    This may be needed to make the image be in the coronal direction.
    Args:
        x (np.ndarray): image volume

    Returns:
        np.ndarray: Flipped image volume
    """
    return np.flip(np.flip(np.flip(x, 0), 1), 2)
