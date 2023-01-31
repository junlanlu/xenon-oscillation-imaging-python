import constants
import io_utils
import spect_utils
from absl import app

twix_path = "/mnt/c/Users/DEEPXENON2/Desktop/Patients/007-02-003/007-02-003_s1/meas_MID00052_FID31022_fid_xe_calibration_2108__67.dat"


def test_spec_utils(twix_path: str):
    out_dict = io_utils.read_dyn_twix(twix_path)
    spect_utils.calculate_static_spectroscopy(
        fid=out_dict[constants.IOFields.FIDS_DIS], dwell_time=constants.IO
    )


def main(argv):
    test_spec_utils(twix_path=twix_path)


if __name__ == "__main__":
    app.run(main)
