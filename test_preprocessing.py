import logging
import pdb

import numpy as np
from absl import app

import preprocessing
import reconstruction
from utils import constants, io_utils, spect_utils, traj_utils

twix_path = (
    "/mnt/d/Patients/007-028B/meas_MID00232_FID13653_xe_radial_Dixon_cor_2105_670.dat"
)
RECON_IMAGE_SIZE = 128


def test_preprocessing(twix_path: str):
    out_dict = io_utils.read_dis_twix(twix_path)
    data_dis, traj_dis, data_gas, traj_gas = preprocessing.prepare_data_and_traj(
        out_dict
    )
    bin_indices = np.arange(0, data_dis.shape[0], 10)
    data, traj = preprocessing.prepare_data_and_traj_keyhole(
        data_dis,
        traj_dis,
        bin_indices,
    )
    pdb.set_trace()
    image = reconstruction.reconstruct(
        data, traj_utils.get_scaling_factor(RECON_IMAGE_SIZE, data_gas.shape[1]) * traj
    )
    io_utils.export_nii(np.abs(image), "tmp/test_preprocessing.nii")


def main(argv):
    test_preprocessing(twix_path=twix_path)


if __name__ == "__main__":
    app.run(main)
