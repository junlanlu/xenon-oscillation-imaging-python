"""Twix file util functions."""

import logging
import pdb
import sys

sys.path.append("..")
import datetime
from typing import Any, Dict

import mapvbvd
import numpy as np

from utils import constants


def get_scan_date(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get the scan date in MM-DD-YYYY format.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        scan date string in MM-DD-YYYY format
    """
    tReferenceImage0 = str(twix_obj.hdr.MeasYaps[("tReferenceImage0",)]).strip('"')
    scan_date = tReferenceImage0.split(".")[-1][:8]
    return scan_date[:4] + "-" + scan_date[4:6] + "-" + scan_date[6:]


def get_dwell_time(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the dwell time in seconds.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        dwell time in seconds
    """
    try:
        return float(twix_obj.hdr.Phoenix[("sRXSPEC", "alDwellTime", "0")]) * 1e-9
    except:
        pass
    try:
        return float(twix_obj.hdr.Meas.alDwellTime.split(" ")[0]) * 1e-9
    except:
        pass
    raise ValueError("Could not find dwell time from twix object")


def get_TR(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
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


def get_TR_dissolved(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the TR in seconds for dissolved phase.

    The dissolved phase TR is defined to be the time between two consecutive dissolved
    phase-FIDS. This is different from the TR in the twix header as the twix header
    provides the TR for two consecutive FIDS. Here, we assume an interleaved sequence.

    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        TR in seconds
    """
    try:
        return 2 * twix_obj.hdr.Config.TR * 1e-6
    except:
        pass
    try:
        return 2 * int(twix_obj.hdr.Config.TR.split(" ")[0]) * 1e-6
    except:
        pass

    raise ValueError("Could not find TR from twix object")


def get_center_freq(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
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


def get_excitation_freq(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the excitation frequency in MHz.

    See: https://mriquestions.com/center-frequency.html for definition of center freq.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        rf excitation frequency in ppm. 0 if not found.
    """
    excitation = 0
    try:
        excitation = twix_obj.hdr.Phoenix["sWipMemBlock", "alFree", "4"]
        return round(
            excitation
            / (constants.GRYOMAGNETIC_RATIO * get_field_strength(twix_obj=twix_obj))
        )
    except:
        pass
    try:
        excitation = twix_obj.hdr.MeasYaps[("sWiPMemBlock", "adFree", "8")]
        return round(
            excitation
            / (constants.GRYOMAGNETIC_RATIO * get_field_strength(twix_obj=twix_obj))
        )
    except:
        logging.warning("Could not get excitation frequency from twix object.")

    return round(
        excitation
        / (constants.GRYOMAGNETIC_RATIO * get_field_strength(twix_obj=twix_obj))
    )


def get_field_strength(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
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


def get_ramp_time(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the ramp time in micro-seconds.

    See: https://mriquestions.com/gradient-specifications.html

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        ramp time in us
    """
    ramp_time = 0.0
    try:
        ramp_time = float(twix_obj.hdr.Meas.RORampTime)
    except:
        pass

    try:
        ramp_time = float(twix_obj["hdr"]["Meas"]["alRegridRampupTime"].split()[0])
    except:
        pass

    return max(100, ramp_time) if ramp_time < 100 else ramp_time


def get_flag_removeOS(twix_obj: mapvbvd._attrdict.AttrDict) -> bool:
    """Get the flag to remove oversampling.

    Returns false by default.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        flag to remove oversampling
    """
    try:
        return twix_obj.image.flagRemoveOS
    except:
        return False


def get_software_version(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get the software version.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        software version
    """
    try:
        return twix_obj.hdr.Dicom.SoftwareVersions
    except:
        pass

    return "unknown"


def get_FOV(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the FOV in cm.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        FOV in cm. 40cm if not found.
    """
    try:
        return float(twix_obj.hdr.Config.ReadFoV) / 10.0
    except:
        pass
    logging.warning("Could not find FOV from twix object. Returning 40cm.")
    return 40.0


def get_TE90(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the TE90 in seconds.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        TE90 in seconds
    """
    return twix_obj.hdr.Phoenix[("alTE", "0")] * 1e-6


def get_flipangle_dissolved(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the dissolved phase flip angle in degrees.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        flip angle in degrees
    """
    scan_date = get_scan_date(twix_obj=twix_obj)
    YYYY, MM, DD = scan_date.split("-")
    if datetime.datetime(int(YYYY), int(MM), int(DD)) < datetime.datetime(2020, 5, 30):
        logging.info("Checking for flip angle in old format.")
        try:
            return float(twix_obj.hdr.MeasYaps[("sWipMemBlock", "adFree", "6")])
        except:
            pass
        try:
            return float(twix_obj.hdr.MeasYaps[("sWiPMemBlock", "adFree", "6")])
        except:
            pass
    try:
        return float(twix_obj.hdr.Meas["adFlipAngleDegree"].split(" ")[1])
    except:
        pass
    try:
        return float(twix_obj.hdr.MeasYaps[("adFlipAngleDegree", "1")])
    except:
        pass
    try:
        return float(twix_obj.hdr.MeasYaps[("adFlipAngleDegree", "0")])
    except:
        pass
    raise ValueError("Unable to find dissolved-phase flip angle in twix object.")


def get_flipangle_gas(twix_obj: mapvbvd._attrdict.AttrDict) -> float:
    """Get the gas phase flip angle in degrees.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        flip angle in degrees. Returns 0.5 degrees if not found.
    """
    try:
        return float(twix_obj.hdr.Meas["adFlipAngleDegree"].split(" ")[0])
    except:
        pass
    try:
        assert float(twix_obj.hdr.MeasYaps[("adFlipAngleDegree", "0")]) < 10.0
        return float(twix_obj.hdr.MeasYaps[("adFlipAngleDegree", "0")])
    except:
        pass
    try:
        return float(twix_obj.hdr.MeasYaps[("sWipMemBlock", "adFree", "5")])
    except:
        pass
    try:
        return float(twix_obj.hdr.MeasYaps[("sWiPMemBlock", "adFree", "5")])
    except:
        pass
    logging.info("Returning default flip angle of 0.5 degrees.")
    return 0.5


def get_orientation(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get the orientation of the image.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        orientation. Returns coronal if not found.
    """
    orientation = ""
    try:
        orientation = str(twix_obj.hdr.Dicom.tOrientation)
    except:
        logging.info("Unable to find orientation from twix object, returning coronal.")
    return orientation.lower() if orientation else constants.Orientation.CORONAL


def get_protocol_name(twix_obj: mapvbvd._attrdict.AttrDict) -> str:
    """Get the protocol name.

    Args:
        twix_obj: twix object returned from mapVBVD function.
    Returns:
        protocol name. Returns "unknown" if not found.
    """
    try:
        return str(twix_obj.hdr.Config.ProtocolName)
    except:
        return "unknown"


def get_dyn_dissolved_fids(
    twix_obj: mapvbvd._attrdict.AttrDict, n_skip_end: int = 20
) -> np.ndarray:
    """Get the dissolved phase FIDS used for dyn. spectroscopy from twix object.

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


def get_gx_data(twix_obj: mapvbvd._attrdict.AttrDict) -> Dict[str, Any]:
    """Get the dissolved phase and gas phase FIDs from twix object.

    For reconstruction, we also need important information like the gradient delay,
    number of fids in each phase, etc. Note, this cannot be trivially read from the
    twix object, and need to hard code some values. For example, the gradient delay
    is slightly different depending on the scanner.
    Args:
        twix_obj: twix object returned from mapVBVD function
    Returns:
        a dictionary containing
        1. dissolved phase FIDs in shape (number of projections,
            number of points in ray).
        2. gas phase FIDs in shape (number of projections, number of points in ray).
        3. number of fids in each phase, used for trajectory calculation. Note:
            this may not always be equal to the shape in 1 and 2.
        4. number of FIDs to skip from the beginning. This may be due to a noise frame.
        5. number of FIDs to skip from the end. This may be due to calibration.
        6. gradient delay x in microseconds.
        7. gradient delay y in microseconds.
        8. gradient delay z in microseconds.
    """
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False
    raw_fids = np.transpose(twix_obj.image.unsorted().astype(np.cdouble))
    flip_angle_dissolved = get_flipangle_dissolved(twix_obj)
    if flip_angle_dissolved == 12:
        if raw_fids.shape[0] == 4200:
            logging.info("Reading in fast dixon data on Siemens Trio.")
            data_gas = raw_fids[0::2, :]
            data_dis = raw_fids[1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        else:
            raise ValueError("Cannot get data from 'fast' dixon twix object.")
    elif flip_angle_dissolved == 20:
        if raw_fids.shape[0] == 2030:
            logging.info("Reading in 'normal' dixon data on Siemens Prisma.")
            data_gas = raw_fids[:-30][0::2, :]
            data_dis = raw_fids[:-30][1::2, :]
            n_frames = data_dis.shape[0]
            n_skip_start = 0
            n_skip_end = 0
            grad_delay_x, grad_delay_y, grad_delay_z = -5, -5, -5
        else:
            raise ValueError("Cannot get data from normal dixon twix object.")
    else:
        raise ValueError("Cannot get data from twix object.")
    return {
        constants.IOFields.FIDS_GAS: data_gas,
        constants.IOFields.FIDS_DIS: data_dis,
        constants.IOFields.N_FRAMES: n_frames,
        constants.IOFields.N_SKIP_START: n_skip_start,
        constants.IOFields.N_SKIP_END: n_skip_end,
        constants.IOFields.GRAD_DELAY_X: grad_delay_x,
        constants.IOFields.GRAD_DELAY_Y: grad_delay_y,
        constants.IOFields.GRAD_DELAY_Z: grad_delay_z,
    }
