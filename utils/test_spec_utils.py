import pdb

import constants
import io_utils
import spect_utils
from absl import app

# twix_path = "/mnt/c/Users/DEEPXENON2/Desktop/Patients/007-02-003/007-02-003_s1/meas_MID00052_FID31022_fid_xe_calibration_2108__67.dat"
# twix_path = "/Users/junlanlu/Desktop/Patients/102007/meas_MID00174_FID05543_fid_xe_calibration_2108__67.dat"
# twix_path = "/Users/junlanlu/Desktop/Patients/test/meas_MID00209_FID04031_fid_xe_calibration_2108__67.dat"
twix_path = "/Users/junlanlu/Desktop/Patients/LH-011A/meas_MID00209_FID04031_fid_xe_calibration_2108__67.dat"


def test_spec_utils(twix_path: str):
    out_dict = io_utils.read_dyn_twix(twix_path)
    rbc2m = spect_utils.calculate_static_spectroscopy(
        fid=out_dict[constants.IOFields.FIDS_DIS],
        dwell_time=out_dict[constants.IOFields.DWELL_TIME],
        tr=out_dict[constants.IOFields.TR],
        center_freq=out_dict[constants.IOFields.FREQ_CENTER],
        rf_excitation=out_dict[constants.IOFields.FREQ_EXCITATION],
        n_avg=50,
    )
    print(rbc2m)


def main(argv):
    test_spec_utils(twix_path=twix_path)


if __name__ == "__main__":
    app.run(main)
