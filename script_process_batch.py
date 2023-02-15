"""Scripts to run oscillation mapping pipeline in batches."""
import glob
import importlib
import logging
import os
import pdb

from absl import app, flags

from main import oscillation_mapping_readin, oscillation_mapping_reconstruction

FLAGS = flags.FLAGS

flags.DEFINE_string("cohort", "healthy", "cohort folder name in data config folder")
CONFIG_PATH = "config/"


def main(argv):
    """Run the oscillation imaging pipeline in multiple subjects.

    Import the config file and run the oscillation imaging pipeline on all
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

    for subject in subjects:
        try:
            config_obj = importlib.import_module(
                name=subject[:-3].replace("/", "."), package=None
            )
            config = config_obj.get_config()
            logging.info("Processing subject: %s", config.subject_id)
            if config.processes.oscillation_mapping_recon:
                oscillation_mapping_reconstruction(config)
            else:
                oscillation_mapping_readin(config)
        except:
            logging.warning("Failed to process subject: %s", subject)


if __name__ == "__main__":
    app.run(main)
