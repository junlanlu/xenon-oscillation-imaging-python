"""MRD util functions."""
import logging
import pdb
import sys
from typing import Any, Dict, List, Tuple

import ismrmrd
import numpy as np

sys.path.append("..")
from utils import constants


def get_institution_name(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get the institution name from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        str: institution name
    """
    return header.acquisitionSystemInformation.institutionName


def get_dwell_time(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the dwell time from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        float: dwell time in seconds
    """
    return 1e-6 * header.encoding[0].trajectoryDescription.userParameterDouble[0].value


def get_dyn_fids(dataset: ismrmrd.hdf5.Dataset, n_skip_end: int = 20) -> np.ndarray:
    """Get the dissolved phase FIDS used for dyn. spectroscopy from mrd object.

    Args:
        header (ismrmrd.hdf5.Dataset): MRD dataset
        n_skip_end: number of fids to skip from the end. Usually they are calibration
            frames.
    Returns:
        dissolved phase FIDs in shape (number of points in ray, number of projections).
    """
    raw_fids = []
    n_projections = dataset.number_of_acquisitions() - n_skip_end
    for i in range(0, int(n_projections)):  # type: ignore
        raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
    return np.transpose(np.asarray(raw_fids))


def get_excitation_freq(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the excitation frequency from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        float: excitation frequency in MHz
    """
    return header.encoding[0].trajectoryDescription.userParameterDouble[1].value


def get_center_freq(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the center frequency from the MRD header.

    See: https://mriquestions.com/center-frequency.html for definition of center freq.
    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        float: center frequency in MHz
    """
    return 1e-6 * float(header.userParameters.userParameterLong[0].value)


def get_TR(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TR from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        float: TR in seconds
    """
    return 1e-3 * header.sequenceParameters.TR[0]


def get_excitation_freq(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the excitation frequency from the MRD header.

    See: https://mriquestions.com/center-frequency.html
    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        float: excitation frequency in ppm
    """
    return header.userParameters.userParameterDouble[0].value


def get_scan_date(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the scan date from the MRD header in MM-DD-YYYY format.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        str: scan date in MM-DD-YYYY format.
    """
    xml_date = header.measurementInformation.seriesDate
    MM = "0" + str(xml_date[0]) if len(str(xml_date[0])) == 1 else str(xml_date[0])
    DD = "0" + str(xml_date[1]) if len(str(xml_date[1])) == 1 else str(xml_date[1])
    YYYY = str(xml_date[2])
    return MM + "-" + DD + "-" + YYYY


def get_flipangle_dissolved(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the dissolved phase flip angle in degrees.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        flip angle in degrees
    """
    return header.sequenceParameters.flipAngle_deg[1]


def get_flipangle_gas(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the gasd phase flip angle in degrees.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        flip angle in degrees
    """
    return header.sequenceParameters.flipAngle_deg[0]


def get_FOV(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the FOV in cm.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        FOV in cm. 40cm if not found.
    """
    try:
        return float(header.encoding[0].reconSpace.fieldOfView_mm.x / 10.0)
    except:
        pass
    logging.warning("Could not find FOV from twix object. Returning 40cm.")
    return 40.0


def get_orientation(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the orientation of the image.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        orientation. Returns coronal if not found.
    """
    orientation = ""
    institution = get_institution_name(header)
    try:
        orientation = str(header.userParameters.userParameterString[0].value)
    except:
        logging.info("Unable to find orientation from twix object, returning coronal.")

    if institution == "CCHMC" and orientation.lower() == constants.Orientation.CORONAL:
        return constants.Orientation.CORONAL_CCHMC
    else:
        return orientation.lower() if orientation else constants.Orientation.CORONAL


def get_protocol_name(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the protocol name.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        protocol name. Returns "unknown" if not found.
    """
    try:
        return str(header.measurementInformation.protocolName)
    except:
        return "unknown"


def get_ramp_time(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the ramp time in micro-seconds.

    See: https://mriquestions.com/gradient-specifications.html

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        ramp time in us
    """
    ramp_time = 0.0
    try:
        ramp_time = float(
            header.encoding[0].trajectoryDescription.userParameterLong[0].value
        )
    except:
        pass

    return max(100, ramp_time) if ramp_time < 100 else ramp_time


def get_TE90(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TE90 in seconds.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        TE90 in seconds
    """
    return header.sequenceParameters.TE[0] * 1e-3


def get_TR_dissolved(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TR in seconds for dissolved phase.

    The dissolved phase TR is defined to be the time between two consecutive dissolved
    phase-FIDS. This is different from the TR in the mrd header as the mrd header
    provides the TR for two consecutive FIDS. Here, we assume an interleaved sequence.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        TR in seconds
    """
    try:
        return 2 * header.sequenceParameters.TR[0] * 1e-3
    except:
        pass

    raise ValueError("Could not find TR from twix object")


def get_gx_data(
    dataset: ismrmrd.hdf5.Dataset,
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> Dict[str, Any]:
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
        3. trajectory in shape (number of projections, number of points in ray, 3).
            assumed that the trajectory is the same for both phases.
        3. number of fids in each phase, used for trajectory calculation. Note:
            this may not always be equal to the shape in 1 and 2.
        4. number of FIDs to skip from the beginning. This may be due to a noise frame.
        5. number of FIDs to skip from the end. This may be due to calibration.
        6. gradient delay x in microseconds.
        7. gradient delay y in microseconds.
        8. gradient delay z in microseconds.
    """
    # get the raw FIDs
    raw_fids = []
    n_projections = dataset.number_of_acquisitions()
    for i in range(0, int(n_projections)):  # type: ignore
        raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
    raw_fids = np.asarray(raw_fids)
    # get the trajectories
    raw_traj = np.empty((raw_fids.shape[0], raw_fids.shape[1], 3))
    for i in range(0, int(n_projections)):  # type: ignore
        raw_traj[i, :, :] = dataset.read_acquisition(i).traj
    return {
        constants.IOFields.FIDS_GAS: raw_fids[0::2, :],
        constants.IOFields.FIDS_DIS: raw_fids[1::2, :],
        constants.IOFields.TRAJ: raw_traj[0::2, :, :],
        constants.IOFields.N_FRAMES: raw_fids.shape[0] // 2,
        constants.IOFields.N_SKIP_START: 0,
        constants.IOFields.N_SKIP_END: 0,
        constants.IOFields.GRAD_DELAY_X: 0,
        constants.IOFields.GRAD_DELAY_Y: 0,
        constants.IOFields.GRAD_DELAY_Z: 0,
    }
