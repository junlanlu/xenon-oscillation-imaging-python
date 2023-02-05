"""Reconstruct 3D image from k-space data and trajectory."""


import numpy as np
from absl import app, logging

from recon import dcf, kernel, proximity, recon_model, system_model
from utils import io_utils


def reconstruct(
    data: np.ndarray,
    traj: np.ndarray,
    kernel_sharpness: float = 0.32,
    overgrid_factor: int = 3,
    image_size: int = 128,
    n_dcf_iter: int = 15,
    verbosity: bool = True,
) -> np.ndarray:
    """Reconstruct k-space data and trajectory.

    Args:
        data (np.ndarray): k space data of shape (K, 1)
        traj (np.ndarray): k space trajectory of shape (K, 3)
        kernel_sharpness (float): kernel sharpness. larger kernel sharpness is sharper
            image
        overgrid_factor (int): overgridding factor
        image_size (int): target reconstructed image size
            (image_size, image_size, image_size)
        n_pipe_iter (int): number of dcf iterations
        verbosity (bool): Log output messages

    Returns:
        np.ndarray: reconstructed image volume
    """
    prox_obj = proximity.L2Proximity(
        kernel_obj=kernel.Gaussian(
            kernel_extent=9 * kernel_sharpness,
            kernel_sigma=kernel_sharpness,
            verbosity=verbosity,
        ),
        verbosity=verbosity,
    )
    system_obj = system_model.MatrixSystemModel(
        proximity_obj=prox_obj,
        overgrid_factor=overgrid_factor,
        image_size=np.array([image_size, image_size, image_size]),
        traj=traj,
        verbosity=verbosity,
    )
    dcf_obj = dcf.IterativeDCF(
        system_obj=system_obj, dcf_iterations=n_dcf_iter, verbosity=verbosity
    )
    recon_obj = recon_model.LSQgridded(
        system_obj=system_obj, dcf_obj=dcf_obj, verbosity=verbosity
    )
    return recon_obj.reconstruct(data=data, traj=traj)


def main(argv):
    """Demonstrate non-cartesian reconstruction.

    Uses demo data from the assets folder.
    """
    data = io_utils.import_mat("assets/demo_radial_mri_data.mat")["data"]
    traj = io_utils.import_mat("assets/demo_radial_mri_traj.mat")["traj"]
    image = reconstruct(data=data, traj=traj)
    io_utils.export_nii(np.abs(image), "tmp/demo.nii")
    logging.info("done!")


if __name__ == "__main__":
    app.run(main)
