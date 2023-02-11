"""Image binning util functions.

Currently supports linear binning.
"""

import pdb

import numpy as np


def linear_bin(
    image: np.ndarray, mask: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Bin images using the linear binning technique.

    Supports n-d binning.

    Args:
        image (np.ndarray): image to be binned
        mask (np.ndarray): boolean mask defining region of interest to be binned.
            Everything outside the mask will be 0.
        thresholds (np.ndarray): array of thresholds of length n. The output will be
            the binned image of integer values between 0 and n+1.

    Returns:
        np.ndarray: Binned image of integer values between 0 and n+1. A value of 0
        corresponds to the region outside the mask. A value of i > 0 correspond to
        the bin number i.
    """
    image_binned = np.zeros(image.shape)
    n = len(thresholds)
    left, right = -1, 0
    while right < n + 1:
        left_threshold = thresholds[left] if left >= 0 else -np.inf
        right_threshold = thresholds[right] if right < n else np.inf
        val = right + 1
        image_binned[
            np.logical_and(image <= right_threshold, image >= left_threshold)
        ] = val
        left += 1
        right += 1
    image_binned[mask == 0] = 0
    return image_binned
