"""Preprocessing util functions."""

import pdb
import sys

sys.path.append("..")
from typing import Any, Dict, Tuple

import numpy as np

from utils import constants, recon_utils, traj_utils


def prepare_data_and_traj(
    data_dict: Dict[str, Any], generate_traj: bool = True
) -> Tuple[np.ndarray, ...]:
    """Prepare data and trajectory for reconstruction.

    Uses a trajectory generated from the metadata in the twix file if generate_traj is
    True. Otherwise, it uses a manually imported trajectory. Then, it removes noise.

    Args:
        data_dict: dictionary containing data and metadata extracted from the twix file.
        Optionally also contains trajectories.
        generate_traj: bool flag to generate trajectory from metadata in twix file.

    Returns:
        A tuple of data and trajectory arrays in the following order:
        1. Dissolved phase FIDs of shape (n_projections, n_points)
        2. Dissolved phase trajectory of shape (n_projections, n_points, 3)
        3. Gas phase FIDs of shape (n_projections, n_points)
        4. Trajectory of shape (n_projections, n_points, 3)
    """
    data_gas = data_dict[constants.IOFields.FIDS_GAS]
    data_dis = data_dict[constants.IOFields.FIDS_DIS]
    if generate_traj:
        traj_x, traj_y, traj_z = traj_utils.generate_trajectory(
            dwell_time=1e6 * data_dict[constants.IOFields.DWELL_TIME],
            ramp_time=data_dict[constants.IOFields.RAMP_TIME],
            n_frames=data_dict[constants.IOFields.N_FRAMES],
            n_points=data_gas.shape[1],
        )
    else:
        raise ValueError("Manual trajectory import not implemented yet.")
    indices_dis = recon_utils.remove_noise_rays(
        data=data_dis,
    )
    indices_gas = recon_utils.remove_noise_rays(
        data=data_gas,
    )
    indices = np.logical_and(indices_dis, indices_gas)
    data_gas, traj_gas_x, traj_gas_y, traj_gas_z = recon_utils.apply_indices_mask(
        data=data_gas,
        traj_x=traj_x,
        traj_y=traj_y,
        traj_z=traj_z,
        indices=indices,
    )
    data_dis, traj_dis_x, traj_dis_y, traj_dis_z = recon_utils.apply_indices_mask(
        data=data_dis,
        traj_x=traj_x,
        traj_y=traj_y,
        traj_z=traj_z,
        indices=indices,
    )
    traj_dis = np.stack([traj_dis_x, traj_dis_y, traj_dis_z], axis=-1)
    traj_gas = np.stack([traj_gas_x, traj_gas_y, traj_gas_z], axis=-1)

    return data_dis, traj_dis, data_gas, traj_gas


def prepare_data_and_traj_keyhole(
    data: np.ndarray,
    traj: np.ndarray,
    bin_indices: np.ndarray,
    key_radius: int = 9,
):
    """Prepare data and trajectory for keyhole reconstruction.

    Uses bin indices to construct a keyhole mask.

    Args:
        data: data FIDs of shape (n_projections, n_points)
        traj: trajectory of shape (n_projections, n_points, 3)
        high_bin_indices: indices of binned projections.
        key_radius: radius of keyhole in pixels.
    Returns:
        A tuple of data and trajectory arrays. The data is flattened to a 1D array
        of shape (K, 1)
        The trajectory is flattened to a 2D array of shape (K, 3)
    """
    data_copy = data.copy()
    data = data.copy()
    data[:, 0:key_radius] = 0.0
    normalization = (
        np.mean(np.abs(data_copy[bin_indices, 0])) * 1 / np.abs(data_copy[:, 0])
    )
    data = np.divide(data, np.expand_dims(normalization, -1))
    data[bin_indices, 0:key_radius] = data_copy[bin_indices, 0:key_radius]
    return np.delete(
        recon_utils.flatten_data(data), np.where(data.flatten() == 0.0), axis=0
    ), np.delete(
        recon_utils.flatten_traj(traj), np.where(data.flatten() == 0.0), axis=0
    )


def normalize_data(data: np.ndarray, normalization: np.ndarray) -> np.ndarray:
    """Normalize data by a given normalization array.

    Args:
        data: data FIDs of shape (n_projections, n_points)
        normalization: normalization array of shape (n_projections,)
    """
    return np.divide(data, np.expand_dims(normalization, -1))


def truncate_data_and_traj(
    data: np.ndarray,
    traj: np.ndarray,
    n_skip_start: int = 200,
    n_skip_end: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate data and trajectory to a specified number of points.

    Args:
        data_dis: data FIDs of shape (n_projections, n_points)
        traj_dis: trajectory of shape (n_projections, n_points, 3)
        n_skip_start: number of projections to skip at the start.
        n_skip_end: number of projections to skip at the end of the trajectory.

    Returns:
        A tuple of data and trajectory arrays with beginning and end projections
        removed.
    """
    shape_data = data.shape
    shape_traj = traj.shape
    return (
        data[n_skip_start : shape_data[0] - (n_skip_end)],
        traj[n_skip_start : shape_traj[0] - (n_skip_end)],
    )
