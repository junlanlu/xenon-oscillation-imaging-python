"""Base configuration file."""
import sys

from ml_collections import config_dict

# parent directory
sys.path.append("..")

from utils import constants


class Config(config_dict.ConfigDict):
    """Base config file."""

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.data_dir = ""
        self.filepath_twix_dyn = ""
        self.processes = Process()
        self.recon = Recon()
        self.platform = constants.Platform
        self.scan_type = constants.ScanType.NORMALDIXON.value
        self.segmentation_key = constants.SegmentationKey.CNN_VENT.value
        self.site = constants.Site.DUKE.value
        self.subject_id = "test"
        self.rbc_m_ratio = 0.0


class Process(object):
    """Define the evaluation processes."""

    def __init__(self):
        """Initialize the process parameters."""
        super().__init__()
        self.oscillation_mapping_recon = True
        self.oscillation_mapping_readin = False


class Recon(object):
    """Define reconstruction configurations."""

    def __init__(self):
        """Initialize the reconstruction parameters."""
        super().__init__()
        self.n_begin_cutoff = 100
        self.n_end_cutoff = 0
        self.key_radius = 9
        self.key_radius_percentage = 30
        self.oscillation_mapping_readin = False


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
