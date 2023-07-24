"""Scripts to analyze effect of SNR on reconstruction error."""
import logging
import pdb
import sys

import numpy as np
import simulation_utils

sys.path.append(".")

from typing import Tuple

from absl import app, flags

import preprocessing
import reconstruction
from utils import constants, img_utils, io_utils, plot, recon_utils

RECON_SIZE = 128
TR = 0.015  # in seconds


def load_data(n: int) -> Tuple:
    """Load simulated data from mat file.

    Args:
        n: number of projections to include from the beginning.
    Returns:
        Tuple of data, trajectory, mask and ground truth image.
        Data is of shape (n_projections * n_points, 1)
        Trajectory is of shape (n_projections * n_points, 3)
    """
    mdict = io_utils.import_mat(
        path="simulations/data/data_snrplot.mat",
    )
    n_points = 128
    n_projections = mdict["data_static"].shape[0] // n_points
    data_static = np.reshape(mdict["data_static"], (n_projections, n_points))
    data_evolve = np.reshape(mdict["data_evolve"], (n_projections, n_points))
    traj_static = np.reshape(mdict["traj_static"], (n_projections, n_points, 3))
    traj_evolve = np.reshape(mdict["traj_evolve"], (n_projections, n_points, 3))

    return (
        data_static[0:n, :],
        traj_static[0:n, :, :],
        data_evolve[0:n, :],
        traj_evolve[0:n, :, :],
        mdict["mask"],
        mdict["image_osc_gt"],
    )


def main(unused_argv):
    """Analysis of SNR effect on reconstruction error."""
    data_static, traj_static, data_rbc, traj_rbc, mask, image_osc_gt = load_data(1000)
    (data_k0, high_indices, low_indices) = simulation_utils.bin_rbc_oscillations(
        data_gas=data_static,
        data_rbc=data_rbc,
        TR=0.015,
        method=constants.BinningMethods.NONE,
    )
    data_rbc_high, traj_rbc_high = preprocessing.prepare_data_and_traj_keyhole(
        data=data_rbc,
        traj=traj_rbc,
        bin_indices=high_indices,
        key_radius=8,
    )
    data_rbc_low, traj_rbc_low = preprocessing.prepare_data_and_traj_keyhole(
        data=data_rbc,
        traj=traj_rbc,
        bin_indices=low_indices,
        key_radius=8,
    )
    image_rbc_high = img_utils.flip_and_rotate_image(
        reconstruction.reconstruct(
            data=data_rbc_high,
            traj=traj_rbc_high,
            kernel_sharpness=0.2,
        )
    )
    image_rbc_low = img_utils.flip_and_rotate_image(
        reconstruction.reconstruct(
            data=data_rbc_low, traj=traj_rbc_low, kernel_sharpness=0.2
        )
    )
    image_rbc = img_utils.flip_and_rotate_image(
        reconstruction.reconstruct(
            data=recon_utils.flatten_data(data_rbc),
            traj=recon_utils.flatten_traj(traj_rbc),
        )
    )
    image_rbc_osc = (
        mask * (np.abs(image_rbc_high) - np.abs(image_rbc_low)) / np.abs(image_rbc)
    )
    io_utils.export_nii(np.abs(image_rbc_osc), "tmp/image_rbc_osc.nii")
    plot.plot_data_rbc_k0(
        t=np.arange(data_rbc.shape[0]) * TR,
        data=data_k0,
        path="tmp/data_k0.png",
        high=high_indices,
        low=low_indices,
    )


if __name__ == "__main__":
    app.run(main)
