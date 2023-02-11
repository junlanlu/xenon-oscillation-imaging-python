import logging
import pdb

import constants
import io_utils
import spect_utils
from absl import app

twix_path = (
    "/mnt/d/Patients/007-028B/meas_MID00231_FID13652_fid_xe_calibration_2108__67.dat"
)


def test_spec_utils(twix_path: str):
    out_dict = io_utils.read_dyn_twix(twix_path)
    rbc2m = spect_utils.calculate_static_spectroscopy(
        fid=out_dict[constants.IOFields.FIDS_DIS],
        dwell_time=out_dict[constants.IOFields.DWELL_TIME],
        tr=out_dict[constants.IOFields.TR],
        # center_freq=out_dict[constants.IOFields.FREQ_CENTER],
        center_freq=24.0922,
        rf_excitation=out_dict[constants.IOFields.FREQ_EXCITATION],
        n_avg=67,
        plot=False,
    )
    logging.info("RBC:M ratio: {}".format(rbc2m))


def main(argv):
    test_spec_utils(twix_path=twix_path)


if __name__ == "__main__":
    app.run(main)
