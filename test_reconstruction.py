import pdb

import numpy as np
from absl import app

import preprocessing
import reconstruction
from utils import img_utils, io_utils, recon_utils, traj_utils

twix_path = (
    "/mnt/d/Patients/007-028B/meas_MID00232_FID13653_xe_radial_Dixon_cor_2105_670.dat"
)

RECON_IMAGE_SIZE = 128


def test_reconstruction(twix_path: str):
    out_dict = io_utils.read_dis_twix(twix_path)
    (
        data_dis,
        traj_dis,
        data_gas,
        traj_gas,
    ) = preprocessing.prepare_data_and_traj(out_dict)
    data_gas, traj_gas = preprocessing.truncate_data_and_traj(data_gas, traj_gas)
    pdb.set_trace()
    image_gas = reconstruction.reconstruct(
        data=recon_utils.flatten_data(data_gas),
        traj=traj_utils.get_scaling_factor(RECON_IMAGE_SIZE, data_gas.shape[1])
        * recon_utils.flatten_traj(traj_gas),
        kernel_sharpness=0.14,
        image_size=RECON_IMAGE_SIZE,
    )
    image_gas = img_utils.flip_and_rotate_image(image_gas)
    io_utils.export_nii(np.abs(image_gas), "tmp/test_reconstruction.nii")


def main(argv):
    test_reconstruction(twix_path=twix_path)


if __name__ == "__main__":
    app.run(main)
