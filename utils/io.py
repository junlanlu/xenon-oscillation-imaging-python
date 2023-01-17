"""Import and export util functions."""

import os
import pdb
import sys

sys.path.append("..")
import csv
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import nibabel as nib
import numpy as np

from utils import constants


def import_nii(path: str) -> np.ndarray:
    """Import image as np.ndarray.

    Args:
        path: str file path of nifti file
    """
    return nib.load(path).get_fdata()


def export_nii(image: np.ndarray, path: str, fov: Optional[float] = None):
    """Export image as nifti file.

    Args:
        image: np.ndarray 3D image to be exporetd
        path: str file path of nifti file
        fov: float field of view
    """
    nii_imge = nib.Nifti1Image(image, np.eye(4))
    nib.save(nii_imge, path)
