"""Script to get the healthy reference distribution."""
import glob
import importlib
import logging
import os
import pdb

import numpy as np
from absl import app, flags

from config import base_config
from subject_classmap import Subject
from utils import io_utils, plot, signal_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("cohort", "healthy", "cohort folder name in config folder")
flags.DEFINE_bool("segmentation", False, "run segmentation again.")
CONFIG_PATH = "config/"


def oscillation_mapping_analysis(config: base_config.Config) -> np.ndarray:
    """Run the pipeline to get the RBC amplitude oscillation.

    Args:
        config (config_dict.ConfigDict): config dict
    Returns:
        image_rbc_osc (np.ndarray): RBC amplitude oscillation image
    """
    subject = Subject(config)
    subject.read_mat_file()
    if FLAGS.segmentation:
        subject.segmentation()
    subject.dixon_decomposition()
    subject.dissolved_analysis()
    subject.dissolved_binning()
    subject.oscillation_analysis()
    return subject.image_rbc_osc * subject.mask_rbc


def compile_distribution():
    """Generate the reference distribution for healthy subjects.

    Import the config file and load in the mat file for all
    subjects specified in by the cohort flag.
    """
    if FLAGS.cohort == "healthy":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "healthy", "*py"))
    elif FLAGS.cohort == "cteph":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "cteph", "*py"))
    elif FLAGS.cohort == "all":
        subjects = glob.glob(os.path.join(CONFIG_PATH, "healthy", "*py"))
        subjects += glob.glob(os.path.join(CONFIG_PATH, "cteph", "*py"))
    else:
        raise ValueError("Invalid cohort name")

    hist_osc = np.array([])

    for subject in subjects:
        config_obj = importlib.import_module(
            name=subject[:-3].replace("/", "."), package=None
        )
        config = config_obj.get_config()
        logging.info("Processing subject: %s", config.subject_id)
        image_rbc_osc = oscillation_mapping_analysis(config)
        hist_osc = np.append(hist_osc, image_rbc_osc[image_rbc_osc != 0])
    io_utils.export_np(hist_osc, "data/reference_dist.npy")


def get_thresholds():
    """Get the thresholds for the healthy reference distribution.

    Apply box-cox transformation to the healthy reference distribution.
    """
    data_osc = io_utils.import_np(path="data/reference_dist.npy")
    scale_factor = 100
    data_trans, lambda_ = signal_utils.boxcox(data=data_osc + scale_factor)

    mean_data_trans = np.mean(data_trans)
    std_data_trans = np.std(data_trans)
    logging.info("mean: {}".format(np.mean(data_osc)))
    logging.info("std: {}".format(np.std(data_osc)))
    plot.plot_histogram_rbc_osc(data_osc, "tmp/healty_hist.png")
    logging.info("Lambda: %s", lambda_)

    for z in range(-2, 5):
        threshold = signal_utils.inverse_boxcox(
            lambda_, mean_data_trans + z * std_data_trans, scale_factor
        )
        logging.info("Box-cox threshold: %s", threshold)

    for z in range(-2, 5):
        threshold = np.mean(data_osc) + z * np.std(data_osc)
        logging.info("Gaussian threshold: %s", threshold)


def main(argv):
    """Run the main function.

    Compile the healthy reference distribution and get the thresholds.
    """
    compile_distribution()
    get_thresholds()


if __name__ == "__main__":
    """Run the main function."""
    app.run(main)
