"""Gridding kernels."""

import logging
import sys
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

sys.path.append("..")
from recon import kernel, proximity


class SystemModel:
    """An abstract class defining a particular system model represenation.

    Attributes:
        verbosity (int): either 0 or 1 whether to log output messages
        proximity_obj (L2Proximity): a subclass that inherits from Proximity class.
        overgrid_factor (int): overgridding factor
        image_size (tuple): reconstructed image size.
    """

    def __init__(
        self,
        proximity_obj: proximity.Proximity,
        overgrid_factor: int,
        image_size: tuple,
        verbosity: int,
    ):
        """Initialize abstract class.

        Args:
            proximity_obj (L2Proximity): a subclass that inherits from Proximity class.
            overgrid_factor (int): overgridding factor
            image_size (tuple): reconstructed image size
            verbosity (int): either 0 or 1 whether to log output messages
        """
        self.verbosity = verbosity
        self.proximity_obj = proximity_obj
        self.overgrid_factor = overgrid_factor
        self.crop_size = np.asarray(image_size)
        self.full_size = np.ceil(
            np.multiply(self.crop_size, self.overgrid_factor)
        ).astype(int)

    def crop(self, uncrop: np.ndarray) -> np.ndarray:
        """Crop the image if overgridding was used.

        Args:
            uncrop (np.ndarray): Uncropped image volume.
        Returns:
            np.ndarray: Cropped image volume
        """
        s_lim = np.round(np.multiply((self.full_size - self.crop_size), 0.5)).astype(
            int
        )
        l_lim = np.round(np.multiply((self.full_size + self.crop_size), 0.5)).astype(
            int
        )
        return uncrop[s_lim[0] : l_lim[0], s_lim[1] : l_lim[1], s_lim[2] : l_lim[2]]
