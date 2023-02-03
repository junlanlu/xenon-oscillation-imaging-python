import logging
import pdb

import constants
import io_utils
import preprocessing
import spect_utils
from absl import app

twix_path = (
    "/mnt/d/Patients/007-028B/meas_MID00232_FID13653_xe_radial_Dixon_cor_2105_670.dat"
)


def test_preprocessing(twix_path: str):
    out_dict = io_utils.read_dis_twix(twix_path)
    preprocessing.prepare_data_and_traj(out_dict)


def main(argv):
    test_preprocessing(twix_path=twix_path)


if __name__ == "__main__":
    app.run(main)
