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
        traj_x, traj_y, traj_z = traj_utils.generate_traj(
            dwell_time=data_dict[constants.IOFields.DWELL_TIME],
            ramp_time=data_dict[constants.IOFields.RAMP_TIME],
            n_frames=data_dict[constants.IOFields.N_FRAMES],
            n_points=data_gas.shape[1],
        )
    else:
        raise ValueError("Manual trajectory import not implemented yet.")
    data_dis, traj_dis_x, traj_dis_y, traj_dis_z = recon_utils.remove_noise_rays(
        data=data_dis,
        traj_x=traj_x,
        traj_y=traj_y,
        traj_z=traj_z,
    )
    data_gas, traj_gas_x, traj_gas_y, traj_gas_z = recon_utils.remove_noise_rays(
        data=data_gas,
        traj_x=traj_x,
        traj_y=traj_y,
        traj_z=traj_z,
    )
    traj_dis = np.stack([traj_dis_x, traj_dis_y, traj_dis_z], axis=-1)
    traj_gas = np.stack([traj_gas_x, traj_gas_y, traj_gas_z], axis=-1)

    return data_dis, traj_dis, data_gas, traj_gas
