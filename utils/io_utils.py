"""Import and export util functions."""

import os
import pdb
import sys

sys.path.append("..")
import csv
import logging
from typing import Any, Dict, List, Optional, Tuple

import mapvbvd
import matplotlib
import nibabel as nib
import numpy as np

from utils import constants, twix_utils


def import_nii(path: str) -> np.ndarray:
    """Import image as np.ndarray.

    Args:
        path: str file path of nifti file
    """
    return nib.load(path).get_fdata()


def read_dyn_twix(path: str) -> Dict[str, Any]:
    """Read dynamic spectroscopy twix file.

    Args:
        path: str file path of twix file
    Returns: dictionary containing data and metadata extracted from the twix file.
    This includes:
        1. scan date in MM-DD-YYY format.
        2. dwell time in seconds.
        3. TR in seconds.
        4. Center frequency in MHz.
        5. excitation frequency in ppm.
        6. dissolved phase FIDs in format (n_points, n_frames).
    """
    try:
        twix_obj = mapvbvd.mapVBVD(path)
    except:
        raise ValueError("Invalid twix file.")
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False

    # Get scan information
    scan_date = twix_utils.get_scan_date(twix_obj=twix_obj)
    dwell_time = twix_utils.get_dwell_time(twix_obj=twix_obj)
    tr = twix_utils.get_TR(twix_obj=twix_obj)
    freq_center = twix_utils.get_center_freq(twix_obj=twix_obj)
    freq_excitation = twix_utils.get_excitation_freq(twix_obj=twix_obj)
    fids_dis = twix_utils.get_dyn_dissolved_fids(twix_obj)

    return {
        constants.IOFields.SCAN_DATE: scan_date,
        constants.IOFields.DWELL_TIME: dwell_time,
        constants.IOFields.TR: tr,
        constants.IOFields.FREQ_CENTER: freq_center,
        constants.IOFields.FREQ_EXCITATION: freq_excitation,
        constants.IOFields.FIDS_DIS: fids_dis,
    }


def export_nii(image: np.ndarray, path: str, fov: Optional[float] = None):
    """Export image as nifti file.

    Args:
        image: np.ndarray 3D image to be exporetd
        path: str file path of nifti file
        fov: float field of view
    """
    nii_imge = nib.Nifti1Image(image, np.eye(4))
    nib.save(nii_imge, path)
