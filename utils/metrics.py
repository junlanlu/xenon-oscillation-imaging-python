"""Metrics for evaluating images."""

import math

import numpy as np
from scipy.ndimage.morphology import binary_dilation


def _get_dilation_kernel(x: int) -> int:
    """Get dilation kernel for binary dilation in 1-dimension."""
    return int((math.ceil(x * 0.025) * 2 + 1))


def snr(image: np.ndarray, mask: np.ndarray, window_size: int = 8):
    """Calculate SNR using sliding windows.

    Args:
        image (np.ndarray): 3-D array of image data.
        mask (np.ndarray): 3-D array of mask data.
        window_size (int): size of the sliding window for noise calculation.
            Defaults to 8.
    Returns:
        Tuple of SNR and Rayleigh SNR.
    """
    shape = np.shape(image)
    # dilate the mask to analyze noise area away from the signal
    kernel_shape = (
        _get_dilation_kernel(shape[0]),
        _get_dilation_kernel(shape[1]),
        _get_dilation_kernel(shape[2]),
    )
    dilate_struct = np.ones((kernel_shape))
    noise_mask = binary_dilation(mask, dilate_struct).astype(bool)

    noise_temp = np.copy(image)
    noise_temp[noise_mask] = np.nan
    # set up for using mini noise cubes to go through the image and calculate std for noise
    n_noise_vox = window_size * window_size * window_size
    mini_vox_std = 0.75 * n_noise_vox  # minimul number of voxels to calculate std

    stepper = 0
    total = 0
    std_dev_mini_noise_vol = []

    for ii in range(0, int(shape[0] / window_size)):
        for jj in range(0, int(shape[1] / window_size)):
            for kk in range(0, int(shape[2] / window_size)):
                mini_cube_noise_dist = noise_temp[
                    ii * window_size : (ii + 1) * window_size,
                    jj * window_size : (jj + 1) * window_size,
                    kk * window_size : (kk + 1) * window_size,
                ]
                mini_cube_noise_dist = mini_cube_noise_dist[
                    ~np.isnan(mini_cube_noise_dist)
                ]
                # only calculate std for the noise when it is long enough
                if len(mini_cube_noise_dist) > mini_vox_std:
                    std_dev_mini_noise_vol.append(np.std(mini_cube_noise_dist, ddof=1))
                    stepper = stepper + 1
                total = total + 1

    image_noise = np.median(std_dev_mini_noise_vol)
    image_signal = np.average(image[mask])

    SNR = image_signal / image_noise
    return SNR, SNR * 0.66
