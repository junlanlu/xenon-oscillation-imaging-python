"""Module for oscillation imaging subject."""
import datetime
import glob
import logging
import os
import pdb

import matplotlib
import nibabel as nib
import numpy as np
import scipy.io as sio

from config import base_config

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import oscillation_binning as ob
import preprocessing as pp
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
    traj_utils,
)


class Subject(object):
    """Module to for processing oscillation imaging.

    Attributes:
        config (config_dict.ConfigDict): config dict

    """

    def __init__(self, config: base_config.Config):
        """Init object."""
        logging.info("Initializing oscillation imaging subject.")
        self.config = config
        self.data_dissolved = np.array([])
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
        self.metrics = {}
        self.rbc_m_ratio = 0.0
        self.segmentation_key = str(config.segmentation_key)
        self.traj_dissolved = np.array([])
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
        """Prepare data and trajectory for reconstruction.

        Also, calculates the scaling factor for the trajectory.
        """
        (
            self.data_dissolved,
            self.traj_dissolved,
            self.data_gas,
            self.traj_gas,
        ) = pp.prepare_data_and_traj(self.dict_dis)
        self.data_dissolved, self.traj_dissolved = pp.truncate_data_and_traj(
            self.data_dissolved,
            self.traj_dissolved,
            n_skip_start=int(self.config.recon.n_skip_start),
            n_skip_end=int(self.config.recon.n_skip_end),
        )
        self.data_gas, self.traj_gas = pp.truncate_data_and_traj(
            self.data_gas,
            self.traj_gas,
            n_skip_start=int(self.config.recon.n_skip_start),
            n_skip_end=int(self.config.recon.n_skip_end),
        )
        self.traj_scaling_factor = traj_utils.get_scaling_factor(
            recon_size=int(self.config.recon.recon_size),
            n_points=self.data_gas.shape[1],
            scale=True,
        )
        self.traj_dissolved *= self.traj_scaling_factor
        self.traj_gas *= self.traj_scaling_factor

    def reconstruction_gas(self):
        """Reconstruct the gas phase image."""
        self.image_gas = reconstruction.reconstruct(
            data=(recon_utils.flatten_data(self.data_gas)),
            traj=recon_utils.flatten_traj(self.traj_gas),
            kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
            kernel_extent=9 * float(self.config.recon.kernel_sharpness_lr),
        )
        self.image_gas = img_utils.flip_and_rotate_image(
            self.image_gas, orientation=constants.Orientation.CORONAL
        )

    def reconstruction_dissolved(self):
        """Reconstruct the dissolved phase image."""
        # divide the data by the gas phase k0 data.
        self.data_dissolved_norm = pp.normalize_data(
            data=self.data_dissolved, normalization=np.abs(self.data_gas[:, 0])
        )
        self.image_dissolved_norm = reconstruction.reconstruct(
            data=(recon_utils.flatten_data(self.data_dissolved_norm)),
            traj=recon_utils.flatten_traj(self.traj_dissolved),
            kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
            kernel_extent=9 * float(self.config.recon.kernel_sharpness_lr),
        )
        self.image_dissolved = reconstruction.reconstruct(
            data=(recon_utils.flatten_data(self.data_dissolved)),
            traj=recon_utils.flatten_traj(self.traj_dissolved),
            kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
            kernel_extent=9 * float(self.config.recon.kernel_sharpness_lr),
        )
        self.image_dissolved_norm = img_utils.flip_and_rotate_image(
            self.image_dissolved_norm, orientation=constants.Orientation.CORONAL
        )
        self.image_dissolvedm = img_utils.flip_and_rotate_image(
            self.image_dissolved, orientation=constants.Orientation.CORONAL
        )

    def reconstruction_rbc_oscillation(self):
        """Reconstruct the RBC oscillation image."""
        # bin rbc oscillations
        (
            self.data_rbc_k0,
            self.high_indices,
            self.low_indices,
            self.rbc_m_ratio_high,
            self.rbc_m_ratio_low,
        ) = ob.bin_rbc_oscillations(
            data_gas=self.data_gas,
            data_dissolved=self.data_dissolved,
            rbc_m_ratio=self.rbc_m_ratio,
            TR=self.dict_dis[constants.IOFields.TR],
        )
        # TODO remove these lines
        mat_dict = io_utils.import_mat("tmp/indices.mat")
        # self.high_indices = mat_dict["High_Bin_Index"]
        # self.low_indices = mat_dict["Low_Bin_Index"]
        mat_dict = io_utils.import_mat("tmp/subject.mat")
        mask_mat = mat_dict["protonMask"]
        self.mask = np.rot90(np.rot90(mask_mat, 3, axes=(0, 2)), 1, axes=(0, 1))
        self.mask = img_utils.flip_and_rotate_image(self.mask)
        self.mask = np.flip(self.mask, axis=1).astype(bool)
        image_rbc_mat = mat_dict["RBC_Tot"]
        image_rbc_mat = np.rot90(
            np.rot90(image_rbc_mat, 3, axes=(0, 2)), 1, axes=(0, 1)
        )
        image_rbc_mat = img_utils.flip_and_rotate_image(image_rbc_mat)
        image_rbc_mat = np.flip(image_rbc_mat, axis=1)
        io_utils.export_nii(np.abs(image_rbc_mat), "tmp/rbc_mat.nii")
        # TODO remove above lines
        # prepare data and traj for reconstruction
        data_dis_high, traj_dis_high = pp.prepare_data_and_traj_keyhole(
            data=self.data_dissolved_norm,
            traj=self.traj_dissolved,
            bin_indices=self.high_indices,
        )
        data_dis_low, traj_dis_low = pp.prepare_data_and_traj_keyhole(
            data=self.data_dissolved_norm,
            traj=self.traj_dissolved,
            bin_indices=self.low_indices,
        )
        # reconstruct data
        self.image_dissolved_high = reconstruction.reconstruct(
            data=data_dis_high,
            traj=traj_dis_high,
            kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
            kernel_extent=9 * float(self.config.recon.kernel_sharpness_lr),
        )
        self.image_dissolved_low = reconstruction.reconstruct(
            data=data_dis_low,
            traj=traj_dis_low,
            kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
            kernel_extent=9 * float(self.config.recon.kernel_sharpness_lr),
        )
        # flip and rotate images
        self.image_dissolved_high = img_utils.flip_and_rotate_image(
            self.image_dissolved_high, orientation=constants.Orientation.CORONAL
        )
        self.image_dissolved_low = img_utils.flip_and_rotate_image(
            self.image_dissolved_low, orientation=constants.Orientation.CORONAL
        )

    def segmentation(self):
        """Segment the thoracic cavity."""
        if self.segmentation_key == constants.SegmentationKey.CNN_VENT.value:
            logging.info("Performing neural network segmenation.")
            self.mask = segmentation.predict(self.image_gas, erosion=3)
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
        self.mask = img_utils.flip_and_rotate_image(self.mask)

    def dixon_decomposition(self):
        """Perform Dixon decomposition on the dissolved-phase images."""
        self.image_rbc, self.image_membrane = img_utils.dixon_decomposition(
            image_gas=self.image_gas,
            image_dissolved=self.image_dissolved,
            mask=self.mask,
            rbc_m_ratio=self.rbc_m_ratio,
        )
        self.image_rbc_norm, _ = img_utils.dixon_decomposition(
            image_gas=self.image_gas,
            image_dissolved=self.image_dissolved_norm,
            mask=self.mask,
            rbc_m_ratio=self.rbc_m_ratio,
        )
        self.image_rbc_high, _ = img_utils.dixon_decomposition(
            image_gas=self.image_gas,
            image_dissolved=self.image_dissolved_high,
            mask=self.mask,
            rbc_m_ratio=self.rbc_m_ratio_high,
        )
        self.image_rbc_low, _ = img_utils.dixon_decomposition(
            image_gas=self.image_gas,
            image_dissolved=self.image_dissolved_low,
            mask=self.mask,
            rbc_m_ratio=self.rbc_m_ratio_low,
        )

    def dissolved_analysis(self):
        """Calculate the dissolved-phase images relative to gas image."""
        self.image_rbc2gas = img_utils.divide_images(
            image1=self.image_rbc, image2=np.abs(self.image_gas), mask=self.mask
        )

    def dissolved_binning(self):
        """Bin dissolved images to colormap bins."""
        self.image_rbc_binned = binning.linear_bin(
            image=self.image_rbc2gas,
            mask=self.mask,
            thresholds=self.config.params.threshold_rbc,
        )

    def oscillation_analysis(self):
        """Calculate the oscillation image from the rbc high, low, and normal images."""
        # calculate the mask for the RBC image with sufficient SNR, excluding defects
        self.mask_rbc = np.logical_and(self.mask, self.image_rbc_binned > 1)
        self.image_rbc_osc = img_utils.calculate_rbc_oscillation(
            self.image_rbc_high, self.image_rbc_low, self.image_rbc_norm, self.mask
        )

    def oscillation_binning(self):
        """Bin oscillation image to colormap bins."""
        self.image_rbc_osc_binned = binning.linear_bin(
            image=self.image_rbc_osc,
            mask=self.mask,
            thresholds=self.config.params.threshold_oscillation,
        )
        # set unanalyzed voxels to -1
        # self.image_rbc_osc_binned[np.logical_and(self.mask, ~self.mask_rbc)] = -1
        io_utils.export_nii(self.image_rbc_osc_binned, "tmp/osc_binned.nii")
        io_utils.export_nii(self.image_rbc_binned, "tmp/rbc_binned.nii")
        io_utils.export_nii(np.abs(self.image_gas), "tmp/gas.nii")
        io_utils.export_nii(np.abs(self.image_rbc), "tmp/rbc.nii")
        io_utils.export_nii(np.abs(self.image_membrane), "tmp/membrane.nii")
        io_utils.export_nii(self.mask.astype(float), "tmp/mask.nii")
        io_utils.export_nii(self.mask_rbc.astype(float), "tmp/mask_rbc.nii")
        io_utils.export_nii(self.image_rbc_osc * self.mask, "tmp/osc.nii")
        io_utils.export_nii(np.abs(self.image_dissolved), "tmp/dissolved.nii")

        pdb.set_trace()
        # plt.hist(self.image_rbc2gas[self.mask > 0].flatten(), 50)
        # plt.hist(self.image_rbc_osc[self.mask > 0].flatten(), 50)
        # plt.plot(self.data_rbc_k0)
        # plt.xlim(-10, 30)
        # plt.show()
        # pdb.set_trace()
        # plt.figure()
        # t = np.arange(self.data_rbc_k0.shape[0])
        # plt.plot(t, self.data_rbc_k0, "k.")
        # plt.plot(t[self.high_indices], self.data_rbc_k0[self.high_indices], "r.")
        # plt.plot(t[self.low_indices], self.data_rbc_k0[self.low_indices], "b.")
        # plt.show()

    def get_statistics(self):
        """Calculate image statistics."""
        self.stats_dict = {
            constants.StatsIOFields.SUBJECT_ID: self.config.subject_id,
            constants.StatsIOFields.INFLATION: metrics.inflation_volume(
                self.mask, self.dict_dis[constants.IOFields.FOV]
            ),
            constants.StatsIOFields.SCAN_DATE: self.dict_dis[
                constants.IOFields.SCAN_DATE
            ],
            constants.StatsIOFields.PROCESS_DATE: metrics.process_date(),
            constants.StatsIOFields.SNR_RBC: metrics.snr(self.image_rbc, self.mask)[0],
            constants.StatsIOFields.SNR_RBC_HIGH: metrics.snr(
                self.image_rbc_high, self.mask
            )[0],
            constants.StatsIOFields.SNR_RBC_LOW: metrics.snr(
                self.image_rbc_low, self.mask
            )[0],
        }

    def generate_figures(self):
        """Export image figures."""

    def generateHtmlPdf(self):
        """Generate HTML and PDF files."""

    def saveMat(self):
        """Save the instance variables into a mat file."""
        return

    def savefiles(self):
        """Save select images to nifti files and instance variable to mat."""
        return
