"""Test io_utils.py.

User must specify the path to the twix file.
"""
import pdb

import io_utils
from absl import app

twix_path = (
    "/mnt/d/Patients/007-028B/meas_MID00232_FID13653_xe_radial_Dixon_cor_2105_670.dat"
)


def test_read_dyn_twix(twix_path: str):
    """Call ready dynamic spectroscopy twix file.

    Args:
        twix_path: path to twix file.
    """
    out_dict = io_utils.read_dyn_twix(twix_path)
    pdb.set_trace()


def test_read_dis_twix(twix_path: str):
    """Call ready dynamic spectroscopy twix file.

    Args:
        twix_path: path to twix file.
    """
    out_dict = io_utils.read_dis_twix(twix_path)
    pdb.set_trace()


def main(argv):
    """Run tests."""
    # test_read_dyn_twix(twix_path=twix_path)
    test_read_dis_twix(twix_path=twix_path)


if __name__ == "__main__":
    app.run(main)
