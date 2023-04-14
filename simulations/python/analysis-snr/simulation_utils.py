"""Simulation utilities."""
import sys
from typing import Literal, Tuple

import numpy as np

sys.path.append(".")
from utils import constants, signal_utils


def bin_rbc_oscillations(
    data_rbc: np.ndarray,
    data_gas: np.ndarray,
    TR: float,
    smooth: bool = False,
    method: str = constants.BinningMethods.BANDPASS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin rbc data into high and low signal bins.

    Args:
        data_rbc: rbc data of shape (n_projections, n_points)
        data_gas: gas phase data of shape (n_projections, n_points)
        TR: repetition time in seconds
        method: method to use for binning
    Returns:
        Tuple of detrendend data, high and low signal indices respectively.
    """
    data_rbc_k0 = data_rbc[:, 0]
    data_gas_k0 = data_gas[:, 0]
    # smooth data
    window_size = int(1 / (5 * TR))
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    if smooth:
        data_rbc_k0 = signal_utils.smooth(data=data_rbc_k0, window_size=window_size)
    if method == constants.BinningMethods.BANDPASS:
        # normalize and detrend by gas k0
        data_rbc_k0_proc = data_rbc_k0 / np.abs(data_gas_k0)
        # apply bandpass filter
        data_rbc_k0_proc = signal_utils.bandpass(
            data=data_rbc_k0_proc, lowcut=0.5, highcut=2.5, fs=1 / TR
        )
        # calculate the heart rate
        heart_rate = signal_utils.get_heartrate(data_rbc_k0_proc, ts=TR)
    elif method == constants.BinningMethods.FIT_SINE:
        # fit data to biexponential decay and remove trend
        data_rbc_k0_proc = signal_utils.detrend(data_rbc_k0)
        # fit sine wave to data
        data_rbc_k0_proc = signal_utils.fit_sine(data_rbc_k0_proc)
        # calculate the heart rate
        heart_rate = signal_utils.get_heartrate(data_rbc_k0_proc, ts=TR)
    else:
        heart_rate = 60
        data_rbc_k0_proc = data_rbc_k0

    # bin data to high and low signal bins
    high_indices, low_indices = signal_utils.find_high_low_indices(
        data=data_rbc_k0_proc,
        peak_distance=int((60 / heart_rate) / TR),
        method=constants.BinningMethods.THRESHOLD,
    )
    return data_rbc_k0_proc, high_indices, low_indices
