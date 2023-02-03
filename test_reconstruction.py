import logging
import pdb

import numpy as np
from absl import app

import preprocessing
import reconstruction
from utils import io_utils

twix_path = (
    "/mnt/d/Patients/007-028B/meas_MID00232_FID13653_xe_radial_Dixon_cor_2105_670.dat"
)


def test_reconstruction(twix_path: str):
    out_dict = io_utils.read_dis_twix(twix_path)
    (
        data_dis,
        traj_dis,
        data_gas,
        traj_gas,
    ) = preprocessing.prepare_data_and_traj(out_dict)
    pdb.set_trace()
    image_gas = reconstruction.reconstruct(
        data=data_gas.reshape((data_gas.shape[0] * data_gas.shape[1], 1)),
        traj=traj_gas.reshape((np.prod(traj_gas.shape[0:2]), 3)),
    )
    io_utils.export_nii(np.abs(image_gas), "tmp/test_reconstruction.nii")
    pdb.set_trace()


def main(argv):
    test_reconstruction(twix_path=twix_path)


if __name__ == "__main__":
    app.run(main)
