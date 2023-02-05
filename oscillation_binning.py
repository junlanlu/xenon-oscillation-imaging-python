"""Bin dissolved phase data into high and low signal bins."""

from typing import Tuple

import numpy as np

from utils import signal_utils


def bin_rbc_oscillations(
    data_gas: np.ndarray, data_dis: np.ndarray, TR: float, rbc_m_ratio: float
):
    """Bin dissolved phase data into high and low signal bins.

    Args:
        data_gas: gas phase data of shape (n_projections, n_points)
        data_dis: dissolved phase data of shape (n_projections, n_points)
        TR: repetition time in seconds
        rbc_m_ratio: RBC:m ratio
    Returns:
        Tuple of high and low signal indices respectively.
    """
    data_rbc_k0, _ = signal_utils.dixon_decomposition(data_dis, rbc_m_ratio)[0][:, 0]
    data_gas_k0 = data_gas[:, 0]
    # normalize and detrend by gas k0
    data_rbc_k0 = data_rbc_k0 / data_gas_k0
    # smooth data
    data_rbc_k0 = signal_utils.smooth(data=data_rbc_k0, window_size=int(5 / TR))
    # filter data
    data_rbc_k0 = signal_utils.bandpass(
        data=data_rbc_k0, lowcut=0.5, highcut=2.5, fs=1 / TR
    )
    # calculate the heart rate
    heart_rate = signal_utils.get_heartrate(data_rbc_k0, fs=1 / TR)
    # bin data to high and low signal bins
    high_indices, low_indices = signal_utils.find_high_low_indices(
        data=data_rbc_k0, peak_distance=int(heart_rate / 60 / TR)
    )
    return high_indices, low_indices
