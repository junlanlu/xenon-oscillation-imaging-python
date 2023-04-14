"""Demo configuration file."""
import sys

from ml_collections import config_dict

# parent directory
sys.path.append("..")

from config import base_config
from utils import constants


class Config(base_config.Config):
    """Demo config file.

    Inherit from base_config.Config and override the parameters.
    """

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.data_dir = "assets/twix/"
        self.platform = constants.Platform.SIEMENS.value
        self.recon = Recon()
        self.scan_type = constants.ScanType.NORMALDIXON.value
        self.segmentation_key = constants.SegmentationKey.CNN_VENT.value
        self.site = constants.Site.DUKE.value
        self.subject_id = "test"
        self.rbc_m_ratio = 0.0


class Recon(base_config.Recon):
    """Define reconstruction configurations.

    Attributes:
        kernel_sharpness_lr: float, the kernel sharpness for low resolution, higher
            SNR images
        kernel_sharpness_hr: float, the kernel sharpness for high resolution, lower
            SNR images
        n_skip_start: int, the number of frames to skip at the beginning
        n_skip_end: int, the number of frames to skip at the end
        key_radius: int, the key radius for the keyhole image
    """

    def __init__(self):
        """Initialize the reconstruction parameters."""
        self.key_radius = 5


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
