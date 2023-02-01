"""Reconstruction util functions."""
import ctypes as ct
import os
import pdb
import platform
import re
import sys

sys.path.append("..")
from typing import Tuple

import GX_Recon_classmap
import numpy as np
from numpy.ctypeslib import ndpointer

from recon import kernel
from utils import constants, traj_utils


def generate_radial_1D_traj(
    dwell_time: float,
    grad_delay_time: float,
    ramp_time: float,
    plat_time: float,
    decay_time: float,
    npts: int,
) -> np.ndarray:
    """Generate 1D radial distance.

    Generate 1d radial distance array based on the timing and the amplitude
        and the gradient delay.

    Args:
        dwell_time (float): dwell time in us
        grad_delay_time (float): gradient delay time in us
        ramp_time (float): gradient ramp time in us
        plat_time (float): plateau time in us
        decay_time (float): decay time in us
        npts (int): number of points in radial projection

    Returns:
        np.ndarray: 1D radial distances
    """
    grad_delay_npts = grad_delay_time / dwell_time
    ramp_npts = ramp_time / dwell_time
    plat_npts = plat_time / dwell_time
    decay_npts = decay_time / dwell_time
    pts_vec = np.array(range(0, npts))
    # calculate sample number of each region boundary
    ramp_start_pt = grad_delay_npts
    plat_start_pt = ramp_start_pt + ramp_npts
    decay_start_pt = plat_start_pt + plat_npts
    decay_end_pt = decay_start_pt + decay_npts
    # calculate binary mask for each region
    in_ramp = (pts_vec >= ramp_start_pt) & (pts_vec < plat_start_pt)
    in_plat = (pts_vec >= plat_start_pt) & (pts_vec < decay_start_pt)
    in_decay = (pts_vec >= decay_start_pt) & (pts_vec < decay_end_pt)
    # calculate times in each region
    ramp_pts_vec = np.multiply((pts_vec - ramp_start_pt), in_ramp)
    plat_pts_vec = np.multiply((pts_vec - plat_start_pt), in_plat)
    decay_pts_vec = np.multiply((pts_vec - decay_start_pt), in_decay)
    # calculate the gradient amplitude  over time(assume plateau is 1)
    ramp_g = ramp_pts_vec / ramp_npts
    plat_g = in_plat
    decay_g = np.multiply((1.0 - decay_pts_vec / decay_npts), in_decay)
    # calculate radial position (0.5)
    ramp_dist = 0.5 * np.multiply(ramp_pts_vec, ramp_g)
    plat_dist = 0.5 * ramp_npts * in_plat + np.multiply(plat_pts_vec, plat_g)
    decay_dist = (0.5 * ramp_npts + plat_npts) * in_decay + np.multiply(
        in_decay, np.multiply(decay_pts_vec * 0.5, (1.0 + decay_g))
    )
    radial_distance = (ramp_dist + plat_dist + decay_dist) / npts
    return radial_distance


def generate_traj(
    dwell_time: float,
    ramp_time: float,
    plat_time: float,
    decay_time: float,
    npts: int = 64,
    del_x: float = 0,
    del_y: float = 0,
    del_z: float = 0,
    nFrames: int = 1000,
    traj_type: str = constants.TrajType.HALTON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate and vectorize trajectory and data.

    Args:
        dwell_time (float): dwell time in us
        grad_delay_time (float): gradient delay time in us
        ramp_time (float): gradient ramp time in us
        plat_time (float): plateau time in us
        decay_time (float): decay time in us
        npts (int): number of points in radial projection
        del_x (float): gradient delay in x-direction in us
        del_y (float): gradient delay in y-direction in us
        del_z (float): gradient delay in z-direction in us
        nFrames (int): number of radial projections
        traj_type (int): trajectory type
            1: Spiral
            2. Halton
            3. Haltonized Spiral
            4. ArchimedianSeq
            5. Double Golden Mean

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: trajectory coodinates in the x, y,
            and z directions.
    """
    traj_para = {
        "npts": npts,
        "dwell_time": dwell_time,
        "ramp_time": ramp_time,
        "plat_time": plat_time,
        "decay_time": decay_time,
    }

    traj_para.update({"grad_delay_time": del_x})
    radial_distance_x = generate_radial_1D_traj(**traj_para)
    traj_para.update({"grad_delay_time": del_y})
    radial_distance_y = generate_radial_1D_traj(**traj_para)
    traj_para.update({"grad_delay_time": del_z})
    radial_distance_z = generate_radial_1D_traj(**traj_para)

    result = traj_utils.gen_traj(nFrames, traj_type)
    x = result[:nFrames]
    y = result[nFrames : 2 * nFrames]
    z = result[2 * nFrames : 3 * nFrames]

    x = np.array([radial_distance_x]).transpose().dot(np.array([x])).transpose()
    y = np.array([radial_distance_y]).transpose().dot(np.array([y])).transpose()
    z = np.array([radial_distance_z]).transpose().dot(np.array([z])).transpose()

    return x, y, z


def remove_noise_rays(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    thre_snr: float,
    tail: float = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Remove noisy FID rays in the k space data.

    Remove noisy FIDs in the kspace data and their corresponding trajectories.

    Args:
        data (np.ndarray): k space datadata of shape (K, 1)
        x (np.ndarray): x coordinates in trajectory of shape (K, 1)
        y (np.ndarray): y coordinates in trajectory of shape (K, 1)
        z (np.ndarray): z coordinates in trajectory of shape (K, 1)
        thre_snr (float): threshold SNR value
        tail (float, optional): Index to define the tail of FID. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]: Tuple of the data with the
            noisy FIDs removed, x, y, and z k-space coordinates with noisy data removed,
            and the number of FIDs remaining after removal.
    """
    thre_dis = thre_snr * np.average(abs(data[:, :5]))
    max_tail = np.amax(abs(data[:, tail:]), axis=1)
    good_index = max_tail < thre_dis
    n_Frames_good = np.sum(good_index)
    data = data[good_index]
    x = x[good_index]
    y = y[good_index]
    z = z[good_index]

    return data, x, y, z, n_Frames_good


def complex_align(x: np.ndarray) -> np.ndarray:
    """Flip and take transpose of image volume.

    This may be needed to make the image be in the coronal direction.
    Args:
        x (np.ndarray): image volume

    Returns:
        np.ndarray: Flipped and transposed image volume.
    """
    return np.flip(np.flip(np.flip(np.transpose(x, (2, 1, 0)), 0), 1), 2)


def alignrot(x: np.ndarray) -> np.ndarray:
    """Flip image volume.

    This may be needed to make the image be in the coronal direction.
    Args:
        x (np.ndarray): image volume

    Returns:
        np.ndarray: Flipped image volume
    """
    return np.flip(np.flip(np.flip(x, 0), 1), 2)


def recon(
    data: np.ndarray,
    traj: np.ndarray,
    kernel_sharpness: float,
    kernel_extent: float,
    overgrid_factor: int,
    image_size: tuple,
    n_pipe_iter: int,
    verbosity: int,
) -> np.ndarray:
    """Reconstruct k-space data and trajectory.

    Args:
        data (np.ndarray): k space data of shape (K, 1)
        traj (np.ndarray): k space trajectory of shape (K, 3)
        kernel_sharpness (float): kernel sharpness. larger kernel sharpness is sharper
            image
        kernel_extent (float): kernel extent
        overgrid_factor (int): overgridding factor
        image_size (tuple): image size
        n_pipe_iter (int): number of dcf iterations
        verbosity (int): either 0 or 1 whether to log output messages

    Returns:
        np.ndarray: reconstructed image volume
    """
    kernel_obj = kernel.Gaussian(
        kernel_extent=kernel_extent, kernel_sigma=kernel_sharpness, verbosity=verbosity
    )
    prox_obj = GX_Recon_classmap.L2Proximity(kernel_obj=kernel_obj, verbosity=verbosity)
    system_obj = GX_Recon_classmap.MatrixSystemModel(
        proximity_obj=prox_obj,
        overgrid_factor=overgrid_factor,
        image_size=image_size,
        traj=traj,
        verbosity=verbosity,
    )
    dcf_obj = GX_Recon_classmap.IterativeDCF(
        system_obj=system_obj, dcf_iterations=n_pipe_iter, verbosity=verbosity
    )
    recon_obj = GX_Recon_classmap.LSQgridded(
        system_obj=system_obj, dcf_obj=dcf_obj, verbosity=verbosity
    )
    reconVol = recon_obj.reconstruct(data=data, traj=traj)
    return reconVol
