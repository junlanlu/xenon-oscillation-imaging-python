"""Module for oscillation imaging subject."""
import datetime
import glob
import logging
import os
import pdb

import nibabel as nib
import numpy as np
import scipy.io as sio
from ml_collections import config_dict

import oscillation_binning as ob
import preprocessing
import reconstruction
import segmentation
from utils import (
    binning,
    constants,
    img_utils,
    io_utils,
    metrics,
    recon_utils,
    spect_utils,
)


class Subject(object):
    """Module to for processing oscillation imaging.

    Attributes:
        config (config_dict.ConfigDict): config dict

    """

    def __init__(self, config: config_dict.ConfigDict):
        """Init object."""
        logging.info("Initializing oscillation imaging subject.")
        self.config = config
        self.data_dis = np.array([])
        self.data_dis_high = np.array([])
        self.data_dis_low = np.array([])
        self.data_gas = np.array([])
        self.dict_dis = {}
        self.dict_dyn = {}
        self.high_indices = np.array([0.0])
        self.image_dissolved = np.array([0.0])
        self.image_gas = np.array([0.0])
        self.image_membrane = np.array([0.0])
        self.image_rbc = np.array([0.0])
        self.image_rbc_high = np.array([0.0])
        self.image_rbc_low = np.array([0.0])
        self.image_rbc_osc = np.array([0.0])
        self.low_indices = np.array([0.0])
        self.manual_segmentation_filepath = str(config.manual_seg_filepath)
        self.mask = np.array([0.0])
        self.rbc_m_ratio = 0.0
        self.segmentation_key = str(config.segmentation_key)
        self.traj_dis = np.array([])
        self.traj_dis_high = np.array([])
        self.traj_dis_low = np.array([])
        self.traj_gas = np.array([])

    def read_files(self):
        """Read in files.

        Read in the dynamic spectroscopy (if it exists) and the dissolved-phase image
        data. Currently only supports twix files but will be extended to support
        other files.
        """
        self.dict_dyn = io_utils.read_dyn_twix(
            io_utils.get_dyn_twix_files(str(self.config.data_dir))
        )
        self.dict_dis = io_utils.read_dis_twix(
            io_utils.get_dis_twix_files(str(self.config.data_dir))
        )

    def calculate_rbc_m_ratio(self):
        """Calculate RBC:M ratio using static spectroscopy.

        If a manual RBC:M ratio is specified, use that instead.
        """
        if self.config.rbc_m_ratio > 0:  # type: ignore
            self.rbc_m_ratio = float(self.config.rbc_m_ratio)  # type: ignore
            logging.info("Using manual RBC:M ratio of {}".format(self.rbc_m_ratio))
        else:
            logging.info("Calculating RBC:M ratio from static spectroscopy.")
            self.rbc_m_ratio = spect_utils.calculate_static_spectroscopy(
                fid=self.dict_dyn[constants.IOFields.FIDS_DIS],
                dwell_time=self.dict_dyn[constants.IOFields.DWELL_TIME],
                tr=self.dict_dyn[constants.IOFields.TR],
                center_freq=self.dict_dyn[constants.IOFields.FREQ_CENTER],
                rf_excitation=self.dict_dyn[constants.IOFields.FREQ_EXCITATION],
                plot=False,
            )

    def readMatFile(self):
        """Read in Mat files."""
        return

    def preprocess(self):
        """Prepare data and trajectory for reconstruction."""
        (
            self.data_dissolved,
            self.traj_dissolved,
            self.data_gas,
            self.traj_gas,
        ) = preprocessing.prepare_data_and_traj(self.dict_dis)
        self.data_dissolved, self.traj_dissolved = preprocessing.truncate_data_and_traj(
            self.data_dissolved,
            self.traj_dissolved,
            n_skip_start=int(self.config.recon.n_skip_start),
            n_skip_end=int(self.config.recon.n_skip_end),
        )

    def reconstruction_gas(self):
        """Reconstruct the gas phase image."""
        self.image_gas = reconstruction.reconstruct(
            data=recon_utils.flatten_data(self.data_gas),
            traj=recon_utils.flatten_traj(self.traj_gas),
            kernel_sharpness=0.32,
        )

    def reconstruction_dissolved(self):
        """Reconstruct the dissolved phase image."""
        self.image_dissolved = reconstruction.reconstruct(
            data=recon_utils.flatten_data(self.data_dissolved),
            traj=recon_utils.flatten_traj(self.traj_dissolved),
            kernel_sharpness=0.14,
        )
        self.image_rbc, self.image_membrane = img_utils.dixon_decomposition(
            image_gas=self.image_gas,
            image_dissolved=self.image_dissolved,
            mask=self.mask,
            rbc_m_ratio=self.rbc_m_ratio,
        )

    def reconstruction_rbc_oscillation(self):
        """Reconstruct the RBC oscillation image."""
        self.high_indices, self.low_indices = ob.bin_rbc_oscillations(
            data_gas=self.data_gas,
            data_dis=self.data_dis,
            rbc_m_ratio=self.rbc_m_ratio,
            TR=self.dict_dis[constants.IOFields.TR],
        )
        data_dis_high, traj_dis_high = preprocessing.prepare_data_and_traj_keyhole(
            self.data_dis, self.traj_dis, self.high_indices
        )
        data_dis_low, traj_dis_low = preprocessing.prepare_data_and_traj_keyhole(
            self.data_dis, self.traj_dis, self.low_indices
        )
        image_dissolved_high = reconstruction.reconstruct(
            data=data_dis_high, traj=traj_dis_high, kernel_sharpness=0.14
        )
        image_dissolved_low = reconstruction.reconstruct(
            data=data_dis_low, traj=traj_dis_low, kernel_sharpness=0.14
        )
        self.image_rbc_high, _ = img_utils.dixon_decomposition(
            image_gas=self.image_gas,
            image_dissolved=image_dissolved_high,
            mask=self.mask,
            rbc_m_ratio=self.rbc_m_ratio,
        )
        self.image_rbc_low, _ = img_utils.dixon_decomposition(
            image_gas=self.image_gas,
            image_dissolved=image_dissolved_low,
            mask=self.mask,
            rbc_m_ratio=self.rbc_m_ratio,
        )
        self.image_rbc_osc = img_utils.calculate_rbc_oscillation(
            self.image_rbc_high, self.image_rbc_low, self.image_rbc
        )

    def segmentation(self):
        """Segment the thoracic cavity."""
        if self.segmentation_key == constants.SegmentationKey.CNN_VENT.value:
            logging.info("Performing neural network segmenation.")
            self.mask = segmentation.predict(self.image_gas)
        elif self.segmentation_key == constants.SegmentationKey.SKIP.value:
            self.mask = np.ones_like(self.image_gas)
        elif self.segmentation_key == constants.SegmentationKey.MANUAL_VENT.value:
            logging.info("loading mask file specified by the user.")
            try:
                mask = glob.glob(self.manual_segmentation_filepath)[0]
                self.mask = np.squeeze(np.array(nib.load(mask).get_fdata()))
            except ValueError:
                logging.error("Invalid mask nifti file.")
        else:
            raise ValueError("Invalid segmentation key.")

    def oscillation_binning(self):
        """Bin oscillation image to colormap bins."""
        return

    def generate_statistics(self):
        """Calculate ventilation image statistics."""

    def generate_figures(self):
        """Export image figures."""

    def generateHtmlPdf(self):
        """Generate HTML and PDF files."""

    def generateCSV(self):
        """Generate a CSV file."""

    def saveMat(self):
        """Save the instance variables into a mat file."""
        return

    def savefiles(self):
        """Save select images to nifti files and instance variable to mat."""
        return
