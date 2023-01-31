"""Spectroscopy util functions."""
import math
import pdb
import sys
from typing import Optional, Tuple

import numpy as np

from spect import nmrTimeFit as fit


def get_breathhold_indices(
    t: np.ndarray, start_time: int, end_time: int
) -> Tuple[int, int]:
    """Get the start and stop index based on the start and stop time.

    Find the index in the time array corresponding to the start time and the end time.
    If the start index is not found, return 0.
    If the stop index is not found return the last index of the array.

    Args:
        t (np.ndarray): array of time points each FID is collected in units of seconds.
        start_time (int): start time (in seconds) of window to analyze t.
        end_time (int): stop time (in seconds) of window to analyze t.

    Returns:
        Tuple of the indices corresponding to the start time and stop time.
    """

    def round_up(x: float, decimals: int = 0) -> float:
        """Round number to the nearest decimal place.

        Args:
            x: floating point number to be rounded up.
            decimals: number of decimal places to round by.

        Returns:
            rounded up value of x.
        """
        return math.ceil(x * 10**decimals) / 10**decimals

    start_ind = np.argwhere(np.array([round_up(x, 2) for x in t]) == start_time)
    end_ind = np.argwhere(np.array([round_up(x, 2) for x in t]) == end_time)

    if np.size(start_ind) == 0:
        start_ind = [0]
    if np.size(end_ind) == 0:
        end_ind = [np.size(t)]

    return (
        start_ind[np.ceil(np.size(start_ind) / 2)],
        end_ind[np.ceil(np.size(end_ind) / 2)],
    )


def get_frequency_guess(
    data: Optional[np.ndarray], center_freq: float, rf_excitation: int
):
    """Get the three-peak initial frequency guess.

    This can be modified in the future to include automated peak finding.

    Args:
        data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
        center_freq (float): center frequency in MHz.
        rf_excitation (int): excitation frequency in ppm.

    Returns: 3-element array of initial frequency guesses corresponding to the RBC,
        membrane, and gas frequencys in MHz
    """
    if rf_excitation == 208:
        return np.array([0, -20.7, -218.4]) * center_freq
    elif rf_excitation == 218:
        return np.array([10, -21.7, -208.4]) * center_freq
    else:
        raise ValueError("Invalid excitation frequency {}".format(rf_excitation))


def get_area_guess(data: Optional[np.ndarray], center_freq: float, rf_excitation: int):
    """Get the three-peak initial area guess.

    This can be modified in the future to include automated peak finding.

    Args:
        data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
        center_freq (float): center frequency in MHz.
        rf_excitation (int): excitation frequency in ppm.

    Returns: 3-element array of initial area guesses corresponding to the RBC,
        membrane, and gas frequencys in MHz
    """
    if rf_excitation == 208:
        return np.array([1, 1, 1])
    elif rf_excitation == 218:
        return np.array([1, 1, 1])
    else:
        raise ValueError("Invalid excitation frequency {}".format(rf_excitation))


def calculate_static_spectroscopy(
    fid: np.ndarray,
    dwell_time: float,
    tr: float,
    center_freq: float,
    rf_excitation: int = 218,
    n_avg: int = 50,
    method: str = "voigt",
):
    """Fit static spectroscopy data to Voigt model extract RBC:M ratio.

    Args:
        fid (np.ndarray): _description_
        dwell_time (float): _description_
        tr (float): _description_
        center_freq (float): _description_
        rf_excitation (int, optional): _description_. Defaults to 218.
        n_avg (int, optional): _description_. Defaults to 50.

    Returns:
        _type_: _description_
    """
    t = np.array(range(0, np.shape(fid)[0])) * dwell_time
    t_tr = np.array(range(0, np.shape(fid)[1])) * tr

    start_ind, _ = get_breathhold_indices(t=t_tr, start_time=2, end_time=10)
    data_dis_avg = np.average(fid[:, start_ind : start_ind + n_avg], axis=1)

    disFit = fit.NMR_TimeFit(
        time_signal=data_dis_avg,
        t=t,
        area=get_area_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
        freq=get_frequency_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
        fwhmL=np.array([8.8, 5.0, 1.2]) * center_freq,
        fwhmG=np.array([0, 6.1, 0]) * center_freq,
        phase=[0, 0, 0],
        line_boardening=0,
        zeropad_size=np.size(t),
        method=method,
    )
    lb = np.stack(
        (
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
        )
    ).flatten()
    ub = np.stack(
        (
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
        )
    ).flatten()
    bounds = (lb, ub)
    disFit.fit_time_signal(bounds)
    rbc_barrier_ratio = round(disFit.area[0] / np.sum(disFit.area[1]), 2)
    return rbc_barrier_ratio
