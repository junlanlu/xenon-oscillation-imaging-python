"""Module for oscillation imaging subject."""
import glob
import logging
import os
import pdb

import nibabel as nib
import numpy as np

import oscillation_binning as ob
import preprocessing as pp
import reconstruction
import segmentation
from config import base_config
from utils import (
    binning,
    constants,
    img_utils,
    io_utils,
    metrics,
    plot,
    recon_utils,
    report,
    signal_utils,
    spect_utils,
    traj_utils,
)


class Subject(object):
    """Module to for processing oscillation imaging.

    Attributes:
        config (config_dict.ConfigDict): config dict
        data_dissolved (np.array): dissolved-phase data of shape
            (n_projections, n_points)
        data_dis_high (np.array): high-key dissolved-phase data of shape
            (n_projections, n_points)
        data_dis_low (np.array): low-key dissolved-phase data of shape
            (n_projections, n_points)
        data_gas (np.array): gas-phase data of shape (n_projections, n_points)
        dict_dis (dict): dictionary of dissolved-phase data and metadata
        dict_dyn (dict): dictionary of dynamic spectroscopy data and metadata
        high_indices (np.array): indices of high projections of shape (n, )
        low_indices (np.array): indices of low projections of shape (n, )
        image_dissolved (np.array): dissolved-phase image
        image_dissolved_norm (np.array): dissolved-phase image reconstructed with
            the data normalized by gas-phase k0
        image_gas (np.array): gas-phase image
        image_membrane (np.array): membrane image
        image_membrane2gas (np.array): membrane image normalized by gas-phase image
        image_rbc (np.array): RBC image
        image_rbc_norm (np.array): RBC image normalized of image_dissolved_norm
        image_rbc2gas (np.array): RBC image normalized by gas-phase image
        image_rbc_high (np.array): RBC image reconstructed with high-key data
        image_rbc_low (np.array): RBC image reconstructed with low-key data
        image_rbc_osc (np.array): RBC oscillation amplitude image
        image_rbc_osc_binned (np.array): RBC oscillation amplitude image binned
        low_indices (np.array): indices of low projections of shape (n, )
        mask (np.array): thoracic cavity mask
        mask_rbc (np.array): thoracic cavity mask with low SNR RBC voxels removed
        rbc_m_ratio (float): RBC to M ratio
        rbc_m_ratio_high (float): RBC to M ratio of high-key data
        rbc_m_ratio_low (float): RBC to M ratio of low-key data
        stats_dict (dict): dictionary of statistics
        traj_dissolved (np.array): dissolved-phase trajectory of shape
            (n_projections, n_points, 3)
        traj_gas (np.array): gas-phase trajectory of shape
            (n_projections, n_points, 3)
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
        self.image_dissolved_norm = np.array([0.0])
        self.image_gas = np.array([0.0])
        self.image_membrane = np.array([0.0])
        self.image_membrane2gas = np.array([0.0])
        self.image_rbc = np.array([0.0])
        self.image_rbc_norm = np.array([0.0])
        self.image_rbc2gas = np.array([0.0])
        self.image_rbc_high = np.array([0.0])
        self.image_rbc_low = np.array([0.0])
        self.image_rbc_osc = np.array([0.0])
        self.image_rbc_osc_binned = np.array([0.0])
        self.low_indices = np.array([0.0])
        self.mask = np.array([0.0])
        self.mask_rbc = np.array([0.0])
        self.rbc_m_ratio = 0.0
        self.rbc_m_ratio_high = 0.0
        self.rbc_m_ratio_low = 0.0
        self.stats_dict = {}
        self.traj_dissolved = np.array([])
        self.traj_dis_high = np.array([])
        self.traj_dis_low = np.array([])
        self.traj_gas = np.array([])

    def read_twix_files(self):
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

    def read_mat_file(self):
        """Read in mat file of reconstructed images.

        Note: The mat file variable names are matched to the instance variable names.
        Thus, if the variable names are changed in the mat file, they must be changed.
        """
        mdict = io_utils.import_mat(io_utils.get_mat_file(str(self.config.data_dir)))
        self.data_dissolved = mdict["data_dissolved"]
        self.data_dissolved_norm = mdict["data_dissolved_norm"]
        self.data_dissolved_norm = mdict["data_dissolved_norm"]
        self.data_rbc_k0 = mdict["data_rbc_k0"].flatten()
        self.high_indices = mdict["high_indices"].flatten()
        self.mask = mdict["mask"]
        self.image_dissolved = mdict["image_dissolved"]
        self.image_dissolved_high = mdict["image_dissolved_high"]
        self.image_dissolved_low = mdict["image_dissolved_low"]
        self.image_dissolved_norm = mdict["image_dissolved_norm"]
        self.image_gas = mdict["image_gas"]
        self.low_indices = mdict["low_indices"].flatten()
        self.rbc_m_ratio = float(mdict["rbc_m_ratio"])
        self.rbc_m_ratio_high = float(mdict["rbc_m_ratio_high"])
        self.rbc_m_ratio_low = float(mdict["rbc_m_ratio_low"])
        self.traj_dissolved = mdict["traj_dissolved"]
        self.traj_gas = mdict["traj_gas"]

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
        self.image_dissolved = img_utils.flip_and_rotate_image(
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
        # prepare data and traj for reconstruction
        data_dis_high, traj_dis_high = pp.prepare_data_and_traj_keyhole(
            data=self.data_dissolved_norm,
            traj=self.traj_dissolved,
            bin_indices=self.high_indices,
            key_radius=self.config.recon.key_radius,
        )
        data_dis_low, traj_dis_low = pp.prepare_data_and_traj_keyhole(
            data=self.data_dissolved_norm,
            traj=self.traj_dissolved,
            bin_indices=self.low_indices,
            key_radius=self.config.recon.key_radius,
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
        if self.config.segmentation_key == constants.SegmentationKey.CNN_VENT.value:
            logging.info("Performing neural network segmenation.")
            self.mask = segmentation.predict(self.image_gas, erosion=5)
        elif self.config.segmentation_key == constants.SegmentationKey.SKIP.value:
            self.mask = np.ones_like(self.image_gas)
        elif (
            self.config.segmentation_key == constants.SegmentationKey.MANUAL_VENT.value
        ):
            logging.info("loading mask file specified by the user.")
            try:
                self.mask = np.squeeze(
                    np.array(nib.load(self.config.manual_seg_filepath).get_fdata())
                ).astype(bool)
            except ValueError:
                logging.error("Invalid mask nifti file.")
        else:
            raise ValueError("Invalid segmentation key.")

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
        self.image_membrane2gas = img_utils.divide_images(
            image1=self.image_membrane, image2=np.abs(self.image_gas), mask=self.mask
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
        image_noise = metrics.snr(self.image_rbc, self.mask)[2]
        self.mask_rbc = np.logical_and(self.mask, self.image_rbc > image_noise)
        self.image_rbc_osc = img_utils.calculate_rbc_oscillation(
            self.image_rbc_high, self.image_rbc_low, self.image_rbc_norm, self.mask_rbc
        )

    def oscillation_binning(self):
        """Bin oscillation image to colormap bins."""
        self.image_rbc_osc_binned = binning.linear_bin(
            image=self.image_rbc_osc,
            mask=self.mask,
            thresholds=self.config.params.threshold_oscillation,
        )
        # set unanalyzed voxels to -1
        self.image_rbc_osc_binned[np.logical_and(self.mask, ~self.mask_rbc)] = -1

    def get_statistics(self):
        """Calculate image statistics."""
        self.stats_dict = {
            constants.StatsIOFields.SUBJECT_ID: self.config.subject_id,
            constants.StatsIOFields.INFLATION: metrics.inflation_volume(
                self.mask, self.dict_dis[constants.IOFields.FOV]
            ),
            constants.StatsIOFields.RBC_M_RATIO: self.rbc_m_ratio,
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
            constants.StatsIOFields.PCT_OSC_DEFECT: metrics.bin_percentage(
                self.image_rbc_osc_binned, np.array([1])
            ),
            constants.StatsIOFields.PCT_OSC_LOW: metrics.bin_percentage(
                self.image_rbc_osc_binned, np.array([2])
            ),
            constants.StatsIOFields.PCT_OSC_HIGH: metrics.bin_percentage(
                self.image_rbc_osc_binned, np.array([6, 7, 8])
            ),
            constants.StatsIOFields.PCT_OSC_MEAN: metrics.mean_oscillation_percentage(
                self.image_rbc_osc, self.mask_rbc
            ),
            constants.StatsIOFields.PCT_OSC_DEFECTLOW: metrics.bin_percentage(
                self.image_rbc_osc_binned, np.array([1, 2])
            ),
        }

    def generate_figures(self):
        """Export image figures."""
        index_start, index_skip = plot.get_plot_indices(self.mask)
        plot.plot_montage_grey(
            image=np.abs(self.image_gas),
            path="tmp/montage_ven.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_grey(
            image=np.abs(self.image_membrane),
            path="tmp/montage_membrane.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_grey(
            image=np.abs(self.image_rbc),
            path="tmp/montage_rbc.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.map_grey_to_rgb(
                self.image_rbc_osc_binned, constants.CMAP.RBC_OSC_BIN2COLOR
            ),
            path="tmp/montage_rbc_rgb.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_histogram_ventilation(
            data=np.abs(self.image_gas)[self.mask].flatten(), path="tmp/hist_ven.png"
        )
        plot.plot_histogram_rbc_osc(
            data=self.image_rbc_osc[self.mask_rbc],
            path="tmp/hist_rbc_osc.png",
        )
        plot.plot_data_rbc_k0(
            t=np.arange(self.data_rbc_k0.shape[0])
            * self.dict_dis[constants.IOFields.TR],
            data=self.data_rbc_k0,
            path="tmp/data_rbc_k0_proc.png",
            high=self.high_indices,
            low=self.low_indices,
        )
        plot.plot_data_rbc_k0(
            t=np.arange(self.data_rbc_k0.shape[0])
            * self.dict_dis[constants.IOFields.TR],
            data=signal_utils.dixon_decomposition(
                self.data_dissolved, self.rbc_m_ratio
            )[0][:, 0],
            path="tmp/data_rbc_k0.png",
            high=self.high_indices,
            low=self.low_indices,
        )

    def generate_pdf(self):
        """Generate HTML and PDF files."""
        path = os.path.join(self.config.data_dir, "report_clinical.pdf")
        report.clinical(self.stats_dict, path=path)

    def write_stats_to_csv(self):
        """Write statistics to file."""
        io_utils.export_subject_csv(self.stats_dict, path="data/stats_clinical.csv")

    def save_subject_to_mat(self):
        """Save the instance variables into a mat file."""
        path = os.path.join(self.config.data_dir, self.config.subject_id + ".mat")
        io_utils.export_subject_mat(self, path)

    def savefiles(self):
        """Save select images to nifti files and instance variable to mat."""
        io_utils.export_nii(self.image_rbc_osc_binned, "tmp/osc_binned.nii")
        io_utils.export_nii(self.image_rbc_binned, "tmp/rbc_binned.nii")
        io_utils.export_nii(np.abs(self.image_gas), "tmp/gas.nii")
        io_utils.export_nii(np.abs(self.image_rbc), "tmp/rbc.nii")
        io_utils.export_nii(np.abs(self.image_membrane), "tmp/membrane.nii")
        io_utils.export_nii(self.mask.astype(float), "tmp/mask.nii")
        io_utils.export_nii(self.mask_rbc.astype(float), "tmp/mask_rbc.nii")
        io_utils.export_nii(self.image_rbc_osc * self.mask, "tmp/osc.nii")
        io_utils.export_nii(np.abs(self.image_dissolved), "tmp/dissolved.nii")
