"""Script to run the 2D ventilation pipeline."""
import logging
import pdb

from absl import app, flags
from ml_collections import config_flags

from config import base_config
from subject_classmap import Subject

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file("config", None, "config file.")


def oscillation_mapping_reconstruction(config: base_config.Config):
    """Run the 2D ventilation pipeline.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = Subject(config=config)
    subject.read_twix_files()
    logging.info("Getting RBC:M ratio from static spectroscopy.")
    subject.calculate_rbc_m_ratio()
    logging.info("Reconstructing images")
    subject.preprocess()
    subject.reconstruction_gas()
    subject.reconstruction_dissolved()
    subject.reconstruction_rbc_oscillation()
    subject.save_subject_to_mat()
    logging.info("Segmenting Proton Mask")
    subject.segmentation()
    subject.dixon_decomposition()
    subject.dissolved_analysis()
    subject.dissolved_binning()
    subject.oscillation_analysis()
    subject.oscillation_binning()
    subject.get_statistics()
    subject.generate_figures()
    subject.generate_pdf()
    logging.info("Complete")


def oscillation_mapping_readin(config: base_config.Config):
    """Run the 2D ventilation pipeline.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = Subject(config=config)
    subject.read_twix_files()
    subject.read_mat_file()
    logging.info("Segmenting Proton Mask")
    subject.segmentation()
    subject.save_subject_to_mat()
    subject.dixon_decomposition()
    subject.dissolved_analysis()
    subject.dissolved_binning()
    subject.oscillation_analysis()
    subject.oscillation_binning()
    subject.get_statistics()
    subject.generate_figures()
    subject.generate_pdf()
    logging.info("Complete")


def main(argv):
    """Run the 2D ventilation pipeline."""
    config = _CONFIG.value
    if config.processes.oscillation_mapping_recon:
        logging.info("Oscillation imaging mapping with reconstruction.")
        oscillation_mapping_reconstruction(config)
    elif config.processes.oscillation_mapping_readin:
        logging.info("Oscillation imaging mapping with reconstruction.")
        oscillation_mapping_readin(config)


if __name__ == "__main__":
    app.run(main)
