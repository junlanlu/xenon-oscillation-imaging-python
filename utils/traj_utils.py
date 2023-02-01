"""Trajectory calculation util functions."""
import logging
import math
import pdb
import sys
from typing import Callable

sys.path.append("..")
import numpy as np
from absl import app, flags

from utils import constants

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_proj", 100, "number of projections per frame.")
flags.DEFINE_string("traj_type", "halton", "trajectory type.")


_GOLDMEAN1 = 0.465571231876768
_GOLDMEAN2 = 0.682327803828019


def _swap(arr: np.ndarray, i: int, j: int):
    """Swap two elements in an array.

    Args:
        arr (np.ndarray): array
        i (int): index of first element
        j (int): index of second element
    """
    arr[i], arr[j] = arr[j], arr[i]


def _partition(
    ht_polar: np.ndarray, sp_polar: np.ndarray, sp_azi: np.ndarray, low: int, high: int
) -> int:
    """Places the pivot element at its correct position in sorted array.

    Also places all smaller (smaller than pivot) to left of pivot and all greater elements
    to right of pivot

    For more details, see: https://www.geeksforgeeks.org/quick-sort/

    Returns:
        int: index of pivot element
    """
    pivot = ht_polar[high]
    i = low - 1
    for j in range(low, high):
        if ht_polar[j] <= pivot:
            i += 1
            _swap(ht_polar, i, j)
            _swap(sp_polar, i, j)
            _swap(sp_azi, i, j)
    _swap(ht_polar, i + 1, high)
    _swap(sp_polar, i + 1, high)
    _swap(sp_azi, i + 1, high)

    return i + 1


def quickSort(
    ht_polar: np.ndarray, sp_polar: np.ndarray, sp_azi: np.ndarray, low: int, high: int
):
    """Sort ht_polar in ascending order. Use sorted indices to sort sp_polar and sp_azi.

    Algorithm explanation: https://www.geeksforgeeks.org/quick-sort/

    """
    if low < high:
        pi = _partition(ht_polar, sp_polar, sp_azi, low, high)
        quickSort(ht_polar, sp_polar, sp_azi, low, pi - 1)
        quickSort(ht_polar, sp_polar, sp_azi, pi + 1, high)


def halton_number(index: int, base: int) -> float:
    """Calculate halton number.

    Reference: https://en.wikipedia.org/wiki/Halton_sequence

    Args:
        index (int): index of halton sequence
        base (int): base of halton sequence

    Returns:
        float: halton number
    """
    result = 0
    f = 1.0
    i = index
    while i > 0:
        f = f / base
        result += f * math.fmod(i, base)
        i = int(i / base)
    return result


def haltonSeq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate halton sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    p1 = 2
    p2 = 3
    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame * num_projPerFrame
            z = halton_number(lk + 1, p1) * 2 - 1
            phi = 2 * math.pi * halton_number(lk + 1, p2)
            arr_polar_angle[linter] = math.acos(z)
            arr_azimuthal_angle[linter] = phi


def spiralSeq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate spiral sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    dPreviousAngle = 0
    num_totalProjections = num_frames * num_projPerFrame
    for lk in range(num_projPerFrame):
        for lFrame in range(num_frames):
            llin = lFrame + lk * num_frames
            linter = lk + lFrame * num_projPerFrame
            dH = -1.0 + 2.0 * llin / float(num_totalProjections)
            arr_polar_angle[linter] = math.acos(dH)
            if llin == 0:
                arr_azimuthal_angle[linter] = 0
            else:
                arr_azimuthal_angle[linter] = math.fmod(
                    dPreviousAngle
                    + 3.6 / (math.sqrt(num_totalProjections * (1.0 - dH * dH))),
                    2.0 * math.pi,
                )
            dPreviousAngle = arr_azimuthal_angle[linter]


def archimedianSeq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate archimedian sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    dAngle = (3.0 - math.sqrt(5.0)) * math.pi
    dZ = 2.0 / (num_projPerFrame - 1.0)

    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame * num_projPerFrame
            arr_polar_angle[linter] = math.acos(1.0 - dZ * lk)
            arr_azimuthal_angle[linter] = lk * dAngle


def dgoldenMSeq(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_frames: int,
    num_projPerFrame: int,
):
    """Generate golden mean sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    for lFrame in range(num_frames):
        for lk in range(num_projPerFrame):
            linter = lk + lFrame * num_projPerFrame
            arr_polar_angle[linter] = math.acos(2.0 * math.fmod(lk * _GOLDMEAN1, 1) - 1)
            arr_azimuthal_angle[linter] = 2 * math.pi * math.fmod(lk * _GOLDMEAN2, 1)


def randomSpiral(
    arr_azimuthal_angle: np.ndarray,
    arr_polar_angle: np.ndarray,
    num_projPerFrame: int,
):
    """Generate random spiral sequence.

    Updates arrays in place.
    Args:
        arr_azimuthal_angle (np.ndarray): azimuthal angle array.
        arr_polar_angle (np.ndarray): polar angle array.
        num_frames (int): number of frames.
        num_projPerFrame (int): number of projections per frame.
    """
    ht_adAzimu = np.zeros(num_projPerFrame)
    ht_adPolar = np.zeros(num_projPerFrame)
    haltonSeq(ht_adAzimu, ht_adPolar, 1, num_projPerFrame)
    quickSort(ht_adPolar, arr_polar_angle, arr_azimuthal_angle, 0, num_projPerFrame - 1)


def traj_factory(traj_type: str) -> Callable:
    """Get trajectory generation function.

    Args:
        traj_type: Trajectory type.
    """
    if traj_type == constants.TrajType.SPIRAL:
        return spiralSeq
    elif traj_type == constants.TrajType.HALTON:
        return haltonSeq
    elif traj_type == constants.TrajType.SPIRALRANDOM:
        return randomSpiral
    elif traj_type == constants.TrajType.ARCHIMEDIAN:
        return archimedianSeq
    elif traj_type == constants.TrajType.GOLDENMEAN:
        return dgoldenMSeq
    else:
        raise ValueError("Invalid trajectory type {}.".format(traj_type))


def gen_traj(num_projPerFrame: int, traj_type: str) -> np.ndarray:
    """Generate trajectory by trajectory type.

    Args:
        n_ProjectionsPerFrame (int): number of projections per frame.
        traj_type (str): trajectory type.

    Returns:
        np.ndarray: trajectory coordinates of shape
        [coordinates_x + coordinates_y + coordinates z, 1]
    """
    m_adAzimuthalAngle = np.zeros(num_projPerFrame)
    m_adPolarAngle = np.zeros(num_projPerFrame)
    coordinates = np.zeros(num_projPerFrame * 3)
    num_frames = 1

    traj_factory(traj_type)(
        m_adAzimuthalAngle, m_adPolarAngle, num_frames, num_projPerFrame
    )

    for k in range(num_projPerFrame):
        coordinates[k] = math.sin(m_adPolarAngle[k]) * math.cos(m_adAzimuthalAngle[k])
        coordinates[k + num_projPerFrame] = math.sin(m_adPolarAngle[k]) * math.sin(
            m_adAzimuthalAngle[k]
        )
        coordinates[k + 2 * num_projPerFrame] = math.cos(m_adPolarAngle[k])
    return coordinates


def main(argv):
    """Generate trajectories for the given number of projections and trajectory type."""
    pdb.set_trace()
    logging.info(gen_traj(FLAGS.n_proj, FLAGS.traj_type))


if __name__ == "__main__":
    app.run(main)
