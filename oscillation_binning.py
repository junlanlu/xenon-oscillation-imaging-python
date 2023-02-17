"""Bin dissolved phase data into high and low signal bins."""

import pdb
from typing import Tuple

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
import numpy as np

from utils import signal_utils


def bin_rbc_oscillations(
    data_gas: np.ndarray, data_dissolved: np.ndarray, TR: float, rbc_m_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Bin dissolved phase data into high and low signal bins.

    Args:
        data_gas: gas phase data of shape (n_projections, n_points)
        data_dis: dissolved phase data of shape (n_projections, n_points)
        TR: repetition time in seconds
        rbc_m_ratio: RBC:m ratio
    Returns:
        Tuple of detrendend data, high and low signal indices respectively.
    """
    # get the k0 data for gas, rbc and membrane
    data_rbc, data_membrane = signal_utils.dixon_decomposition(
        data_dissolved, rbc_m_ratio
    )
    data_rbc_k0, data_membrane_k0 = data_rbc[:, 0], data_membrane[:, 0]
    data_gas_k0 = data_gas[:, 0]
    # normalize and detrend by gas k0
    data_rbc_k0_proc = data_rbc_k0 / np.abs(data_gas_k0)
    # negate data if mean is negative
    data_rbc_k0_proc = (
        -data_rbc_k0_proc if np.mean(data_rbc_k0_proc) < 0 else data_rbc_k0_proc
    )
    # smooth data
    window_size = int(1 / (5 * TR))
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    data_rbc_k0_proc = signal_utils.smooth(
        data=data_rbc_k0_proc, window_size=window_size
    )
    # detrend residual exponential decay
    # data_rbc_k0_proc = signal_utils.detrend(data_rbc_k0_proc)
    # apply low pass filter
    data_rbc_k0_proc = signal_utils.bandpass(
        data=data_rbc_k0_proc, lowcut=0.5, highcut=2.5, fs=1 / TR
    )
    # calculate the heart rate
    heart_rate = signal_utils.get_heartrate(data_rbc_k0_proc, ts=TR)
    # bin data to high and low signal bins
    high_indices, low_indices = signal_utils.find_high_low_indices(
        data=data_rbc_k0_proc, peak_distance=int((60 / heart_rate) / TR)
    )
    # calculate the mean RBC:m ratio for high and low signal bins
    rbc_m_high = np.abs(
        np.mean(data_rbc_k0[high_indices]) / np.mean(data_membrane_k0[high_indices])
    )
    rbc_m_low = np.abs(
        np.mean(data_rbc_k0[low_indices]) / np.mean(data_membrane_k0[low_indices])
    )
    return data_rbc_k0_proc, high_indices, low_indices, rbc_m_high, rbc_m_low
