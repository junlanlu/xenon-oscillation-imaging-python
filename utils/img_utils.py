"""Miscellaneous util functions mostly image processing."""

import pdb
import sys

sys.path.append("..")
from typing import Any, List, Tuple

import numpy as np
import scipy
import skimage
from scipy import ndimage

from utils import constants


def remove_small_objects(mask: np.ndarray, scale: float = 0.1):
    """Remove small unconnected voxels in the mask.

    Args:
        mask (np.ndarray): boolean mask
        scale (float, optional): scalaing factor to determin minimum size.
            Defaults to 0.015.

    Returns:
        Mask with the unconnected voxels removed
    """
    min_size = np.sum(mask) * scale
    return skimage.morphology.remove_small_objects(
        ar=mask, min_size=min_size, connectivity=1
    ).astype("bool")
