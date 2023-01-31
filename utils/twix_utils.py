"""Twix file util functions."""

import logging
import pdb
import sys

sys.path.append("..")

import mapvbvd
import numpy as np

from utils import constants


def get_scan_date(twix_obj: mapvbvd._attrdict) -> str:
    """Get the scan date in MM-DD-YYYY format.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        scan date string in MM-DD-YYYY format
    """
    tReferenceImage0 = str(twix_obj.hdr.MeasYaps[("tReferenceImage0",)]).strip('"')
    scan_date = tReferenceImage0.split(".")[-1][:8]
    return scan_date[:4] + "-" + scan_date[4:6] + "-" + scan_date[6:]


def get_dwell_time(twix_obj: mapvbvd._attrdict) -> float:
    """Get the dwell time in seconds.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        dwell time in seconds
    """
    return twix_obj.hdr.Phoenix[("sRXSPEC", "alDwellTime", "0")] * 1e-9


def get_TR(twix_obj: mapvbvd._attrdict) -> float:
    """Get the TR in seconds.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        TR in seconds
    """
    try:
        return float(twix_obj.hdr.Config.TR.split(" ")[0]) * 1e-6
    except:
        pass

    try:
        return float(twix_obj.hdr.Phoenix[("alTR", "0")]) * 1e-6
    except:
        pass

    raise ValueError("Could not find TR from twix object")


def get_dyn_dissolved_fids(
    twix_obj: mapvbvd._attrdict, n_skip_end: int = 20
) -> np.ndarray:
    """Get the dissoled phase FIDS used for dyn. spectroscopy from twix object.

    Args:
        twix_obj: twix object returned from mapVBVD function
        n_skip_end: number of fids to skip from the end. Usually they are calibration
            frames.
    Returns:
        dissolved phase FIDs in shape (number of points in ray, number of projections).
    """
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False

    raw_fids = twix_obj.image[""].astype(np.cdouble)
    return raw_fids[:, 0 : -(1 + n_skip_end)]


def get_center_freq(twix_obj: mapvbvd._attrdict) -> float:
    """Get the center frequency in MHz.

    See: https://mriquestions.com/center-frequency.html for definition of center freq.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        center frequency in MHz.
    """
    try:
        return twix_obj.hdr.Meas.lFrequency * 1e-6
    except:
        pass

    try:
        return int(twix_obj.hdr.Dicom["lFrequency"]) * 1e-6
    except:
        pass

    raise ValueError("Could not find center frequency (MHz) from twix object")


def get_excitation_freq(twix_obj: mapvbvd._attrdict) -> float:
    """Get the excitation frequency in MHz.

    See: https://mriquestions.com/center-frequency.html for definition of center freq.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        rf excitation frequency in ppm.
    """
    excitation = 0
    try:
        excitation = twix_obj.hdr.Phoenix["sWipMemBlock", "alFree", "4"]
    except:
        raise ValueError("Could not excitation frequency from twix object.")

    return round(
        excitation
        / (constants._GRYOMAGNETIC_RATIO * get_field_strength(twix_obj=twix_obj))
    )


def get_field_strength(twix_obj: mapvbvd._attrdict) -> float:
    """Get the magnetic field strength in Tesla.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        magnetic field strength in Tesla.
    """
    try:
        mag_strength = twix_obj.hdr.Dicom.flMagneticFieldStrength
    except:
        logging.warning("Could not find magnetic field strength, using 3T.")
        mag_strength = 3.0
    return mag_strength
