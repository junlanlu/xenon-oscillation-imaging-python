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


def flip_image_complex(image: np.ndarray) -> np.ndarray:
    """Flip image of complex type along all axes.

    Args:
        image (np.ndarray): image to flip
    Returns:
        Flipped image.
    """
    return np.flip(np.flip(np.flip(np.transpose(image, (2, 1, 0)), 0), 1), 2)


def rotate_axial_to_coronal(image: np.ndarray) -> np.ndarray:
    """Rotate axial image to coronal.

    Image is assumed to be of complex datatype.

    Args:
        image (np.ndarray): image viewed in axial orientation.
    Returns:
        Rotated coronal image.
    """
    real = ndimage.rotate(ndimage.rotate(np.real(image), 90, (1, 2)), 270)
    imag = ndimage.rotate(ndimage.rotate(np.imag(image), 90, (1, 2)), 270)
    return real + 1j * imag


def flip_and_rotate_image(
    image: np.ndarray, orientation: str = constants.Orientation.CORONAL
) -> np.ndarray:
    """Flip and rotate image based on orientation.

    Args:
        image (np.ndarray): image to flip and rotate.
        orientation (str, optional): orientation of the image. Defaults to coronal.
    Returns:
        Flipped and rotated image.
    """
    if orientation == constants.Orientation.CORONAL:
        return flip_image_complex(image)
    elif orientation == constants.Orientation.TRANSVERSE:
        return rotate_axial_to_coronal(flip_image_complex(image))
    else:
        raise ValueError("Invalid orientation: {}.".format(orientation))


def correct_B0(
    image: np.ndarray, mask: np.ndarray, max_iterations: int = 20
) -> np.ndarray:
    """Correct B0 inhomogeneity.

    Args:
        image (np.ndarray): image to correct.
        mask (np.ndarray): mask of the image. must be same shape as image.
        max_iterations (int, optional): maximum number of iterations. Defaults to 20.
    Returns:
        Corrected phase of the corrected image.
    """
    iterCount = 0
    meanphase = 1

    while abs(meanphase) > 1e-7:
        iterCount = iterCount + 1
        diffphase = np.angle(image)
        meanphase = np.mean(diffphase[mask])  # type: ignore
        image = np.multiply(image, np.exp(-1j * meanphase))
        if iterCount > max_iterations:
            break

    return np.angle(image)  # type: ignore


def dixon_decomposition(
    image_gas: np.ndarray,
    image_dissolved: np.ndarray,
    mask: np.ndarray,
    rbc_m_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply 1-point dixon decomposition on images.

    Applies phase shift to the dissolved image such that the RBC and membrane are
    separated into the imaginary and real channel respectively. Also applies B0
    inhomogeneity correction by shifting the gas image to have zero mean phase and
    applying the same phase shift to the dissolved image.

    Args:
        image_gas (np.ndarray): gas image
        image_dissolved (np.ndarray): dissolved image
        mask (np.ndarray): boolean mask of the lung. must be the same size as the images.
        rbc_m_ratio (float): RBC:m ratio
    Returns:
        Tuple of decomposed RBC and membrane images.
    """
    desired_angle = np.arctan2(rbc_m_ratio, 1.0)
    total_dissolved = np.sum(image_dissolved[mask])
    current_angle = np.arctan2(np.imag(total_dissolved), np.real(total_dissolved))
    delta_angle = desired_angle - current_angle

    rotVol = np.multiply(image_dissolved, np.exp(1j * delta_angle))

    diffphase = -correct_B0(image_gas, mask)
    rotVol_B = np.multiply(rotVol, np.exp(1j * diffphase))
    return np.imag(rotVol_B), np.real(rotVol_B)
