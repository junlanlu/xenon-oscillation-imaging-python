"""Module for oscillation imaging subject."""
import datetime
import glob
import logging
import os
import pdb

import nibabel as nib
import numpy as np
import scipy.io as sio
import segmentation
from ml_collections import config_dict

from utils import binning, constants, img_utils, io, metrics


class Subject(object):
    """Module to for processing 2D ventilation.

    Attributes:
        acquisition_time_H: str proton acquisition date
        acquisition_time_Xe: str xenon acquisition date
        bias_key: str bias field correction key
        config: config_dict ml_collections config dict
        data_dir: str output data directory
        data_dir: str output folder path
        FOVdim: float maximum FOV of proton and xenon scans.
        HFOV: float proton field of view.
        manual_reg_dir: str registered proton nifti filepath
        manual_seg_dir: str registered mask nifti filepath
        mask_combined_vesselness: np.ndarray registered mask with vesselness
        mask_combined: np.ndarray inverted mask used to calculate vessel mask
        mask_proton_reg: np.ndarray registered proton mask
        mask_proton_vesselness: np.ndarray
        proton_dicom_dir: str proton dicom directory
        proton_pixel_size: float proton pixel size
        proton_raw: np.ndarray raw proton image
        proton_reg: np.ndarray registered proton image
        proton: np.ndarray scaled proton image
        protonslicethickness: float proton image slice thickness
        registration_key: str registration key
        scan_type: str scan type
        site: str site name (Duke, UVA, etc.)
        stats_dict: dict summary statistics
        subject_id: str subject id
        segmentation_key: str segmentation key
        ventilation: np.nparray scaled ventilation image
        ventilation_biasfield: np.ndarray bias field of ventilation image
        ventilation_binning: np.ndarray binned ventilation image
        ventilation_cor: np.ndarray bias field corrected ventilation image
        ventilation_pixelsize: np.ndarray ventilation image pixel size
        ventilation_raw: np.ndarray raw ventilation image
        vesselmask: np.ndarray mask of vessels
        xefov: float xenon image field of view
        xenon_dicom_dir: str xenon dicom directory
        xenonslicethickness: float xenon image slice thickness

    """

    def __init__(self, config: config_dict.ConfigDict):
        """Init object."""
        logging.info("Initializing 2D ventilation subject.")
        self.acquisition_time_H = ""
        self.acquisition_time_Xe = ""
        self.bias_key = str(config.bias_key)
        self.config = config
        self.data_dir = str(config.data_dir)
        self.FOVdim = 0.0
        self.HFOV = 0.0
        self.proton_dicom_dir = str(config.proton_dicom_dir)
        self.manual_reg_dir = str(config.manual_reg_dir)
        self.manual_seg_dir = str(config.manual_seg_dir)
        self.mask_reg = np.array([0.0])
        self.mask_combined_vesselness = np.array([0.0])
        self.mask_proton_reg = np.array([])
        self.mask_proton_vesselness = np.array([0.0])
        self.mask_reg_vent = np.array([0.0])
        self.proton = np.array([0.0])
        self.proton_pixelsize = 0.0
        self.proton_raw = np.array([0.0])
        self.proton_reg = np.array([0.0])
        self.protonslicethickness = 0.0
        self.registration_key = str(config.registration_key)
        self.scan_type = str(config.scan_type)
        self.segmentation_key = str(config.segmentation_key)
        self.site = str(config.site)
        self.stats_dict = {}
        self.subject_id = str(config.subject_id)
        self.ventilation = np.array([0.0])
        self.ventilation_biasfield = np.array([0.0])
        self.ventilation_binning = np.array([0.0])
        self.ventilation_cor = np.array([0.0])
        self.ventilation_pixelsize = 0.0
        self.ventilation_raw = np.array([0.0])
        self.vesselmask = np.array([0.0])
        self.xefov = 0.0
        self.xenon_dicom_dir = str(config.xenon_dicom_dir)
        self.xenonslicethickness = 0.0

    # Function definitions
    def readinfiles(self):
        """Read in dicom files.

        Also scale each slice to be 128x128.
        """
        # xenon scan
        out_dict = io.importDICOM(path=self.xenon_dicom_dir, scan_type=self.scan_type)
        self.ventilation_raw = out_dict[constants.IOFields.IMAGE]
        self.ventilation_pixelsize = out_dict[constants.IOFields.PIXEL_SIZE]
        self.xenonslicethickness = out_dict[constants.IOFields.SLICE_THICKNESS]
        self.xefov = out_dict[constants.IOFields.FOV]
        self.acquisition_time_Xe = out_dict[constants.IOFields.SCAN_DATE]
        # proton scan
        out_dict = io.importDICOM(path=self.proton_dicom_dir, scan_type=self.scan_type)
        self.proton_raw = out_dict[constants.IOFields.IMAGE]
        self.proton_pixelsize = out_dict[constants.IOFields.PIXEL_SIZE]
        self.protonslicethickness = out_dict[constants.IOFields.SLICE_THICKNESS]
        self.HFOV = out_dict[constants.IOFields.FOV]
        self.acquisition_time_H = out_dict[constants.IOFields.SCAN_DATE]

        self.FOVdim = constants._FOVDIMSCALE * max(self.HFOV, self.xefov)

        self.ventilation = misc.scale2match(
            image=self.ventilation_raw,
            pixelsize=self.ventilation_pixelsize,
            fov=self.FOVdim,
        )
        self.ventilation_raw = misc.scale2match(
            image=self.ventilation_raw,
            pixelsize=self.ventilation_pixelsize,
            fov=self.FOVdim,
        )
        self.proton = misc.scale2match(
            image=self.proton_raw, pixelsize=self.proton_pixelsize, fov=self.FOVdim
        )

    def registration(self):
        """Register moving image to target image.

        Uses ANTs registration to register the proton image to the xenon image.
        """
        if self.registration_key == constants.RegistrationKey.MASK2GAS.value:
            logging.info("Run registration algorithm, vent is fixed, mask is moving")
            self.mask_reg, self.proton_reg = np.abs(
                registration.register_ants(
                    abs(self.ventilation), self.mask_reg, self.proton
                )
            )
        elif self.registration_key == constants.RegistrationKey.PROTON2GAS.value:
            logging.info("Run registration algorithm, vent is fixed, proton is moving")
            self.proton_reg, mask = np.abs(
                registration.register_ants(
                    abs(self.ventilation), self.proton, self.mask_reg
                )
            )
            if (
                self.segmentation_key == constants.SegmentationKey.CNN_PROTON.value
                or self.segmentation_key
                == constants.SegmentationKey.MANUAL_PROTON.value
            ):
                self.mask_reg = mask
        elif self.registration_key == constants.RegistrationKey.MANUAL.value:
            # Load a file specified by the user
            try:
                proton_reg = glob.glob(self.manual_reg_dir)[0]
                self.proton_reg = np.squeeze(np.array(nib.load(proton_reg).get_fdata()))
            except ValueError:
                logging.error("Invalid proton nifti file.")
        elif self.registration_key == constants.RegistrationKey.SKIP.value:
            logging.info("No registration, setting registered proton to proton")
            self.proton_reg = self.proton

    def segmentation(self):
        """Segment the thoracic cavity."""
        if self.segmentation_key == constants.SegmentationKey.CNN_PROTON.value:
            logging.info("Performing neural network segmenation.")
            self.mask_reg = segmentation.evaluate(self.proton)
        elif self.segmentation_key == constants.SegmentationKey.CNN_VENT.value:
            pass
        elif (
            self.segmentation_key == constants.SegmentationKey.MANUAL_VENT.value
            or self.segmentation_key == constants.SegmentationKey.MANUAL_PROTON.value
        ):
            logging.info("loading mask file specified by the user.")
            try:
                mask = glob.glob(self.manual_seg_dir)[0]
                self.mask_reg = np.squeeze(np.array(nib.load(mask).get_fdata()))
            except ValueError:
                logging.error("Invalid mask nifti file.")
        elif self.segmentation_key == constants.SegmentationKey.SKIP.value:
            self.mask_reg = np.ones_like(self.ventilation)
        elif self.segmentation_key == constants.SegmentationKey.THRESHOLD_VENT.value:
            logging.info("segmentation via thresholding.")
            self.mask_reg = (
                self.ventilation
                > np.percentile(
                    self.ventilation, constants._VEN_PERCENTILE_THRESHOLD_SEG
                )
            ).astype(bool)
            self.mask_reg = misc.remove_small_objects(self.mask_reg).astype("float64")
        else:
            raise ValueError("Invalid segmentation key.")

    def biasfield_correction(self):
        """Correct ventilation image for bias field."""
        if self.bias_key == constants.BiasfieldKey.SKIP.value:
            logging.info("skipping bias field correction.")
            self.ventilation_cor = abs(self.ventilation)
            self.ventilation_biasfield = np.ones(self.ventilation.shape)
        elif self.bias_key == constants.BiasfieldKey.N4ITK.value:
            (
                self.ventilation_cor,
                self.ventilation_biasfield,
            ) = biasfield.biasFieldCor(
                image=abs(self.ventilation),
                mask=self.mask_reg.astype(bool),
            )
        else:
            raise ValueError("Invalid bias field correction key.")

    def gas_binning(self):
        """Bin ventilation image to colormap bins."""
        bin_threshold = constants.REFERENCESTATS.REF_BINS_VEN_GRE
        (
            self.ventilation,
            self.ventilation_binning,
            self.mask_reg_vent,
        ) = binning.gasBinning(
            image=abs(self.ventilation_cor),
            bin_threshold=bin_threshold,
            mask=self.mask_reg,
            percentile=constants._VEN_PERCENTILE_RESCALE,
        )

    def generate_statistics(self):
        """Calculate ventilation image statistics."""
        gas_stats = metrics.binStats(
            image=abs(self.ventilation_cor),
            image_bin=self.ventilation_binning,
            image_raw=abs(self.ventilation_cor),
            mask=self.mask_reg.astype(bool),
        )
        self.stats_dict = gas_stats

        # Generate TCV (thoracic cavity volume) metric
        if (
            self.scan_type == constants.ScanType.GRE.value
            or self.scan_type == constants.ScanType.SPIRAL.value
        ):
            inflation_volume = metrics.inflation_volume_2D(
                self.mask_reg, self.xefov, self.xenonslicethickness
            )
            self.stats_dict[constants.STATSIOFields.INFLATION] = inflation_volume
        else:
            raise ValueError("Invalid scan type")
        self.stats_dict[constants.STATSIOFields.SCAN_DATE] = (
            self.acquisition_time_H[0:4]
            + "-"
            + self.acquisition_time_H[4:6]
            + "-"
            + self.acquisition_time_H[6:8]
        )

        current_date = datetime.date.today()
        self.stats_dict[constants.STATSIOFields.PROCESS_DATE] = str(
            current_date.strftime("%Y-%m-%d")
        )

    def generate_figures(self):
        """Export image figures."""
        ## make montage, plot histogram, and generate report
        index2color = constants.BIN2COLORMAP.VENT_BIN2COLOR_MAP
        # Get start/stop intervals
        ind_start, ind_inter = misc.get_start_interval(
            mask=self.mask_reg, scan_type=self.scan_type
        )
        # export montages
        io.export_montage_overlay(
            image_bin=self.ventilation_binning,
            image_background=self.proton_reg,
            path=os.path.join(
                self.data_dir, constants.OutputPaths.VEN_COLOR_MONTAGE_PNG
            ),
            subject_id=self.subject_id,
            index2color=index2color,
            ind_start=ind_start,
            ind_inter=ind_inter,
        )
        io.export_montage_gray(
            image=self.ventilation_raw,
            path=os.path.join(self.data_dir, constants.OutputPaths.VEN_COR_MONTAGE_PNG),
            ind_start=ind_start,
            ind_inter=ind_inter,
            min=np.percentile(np.abs(self.ventilation_raw), 0).astype(float),
            max=np.percentile(
                np.abs(self.ventilation_raw), constants._VEN_PERCENTILE_RESCALE
            ).astype(float),
        )
        io.export_montage_gray(
            image=self.proton_reg,
            path=os.path.join(
                self.data_dir, constants.OutputPaths.PROTON_REG_MONTAGE_PNG
            ),
            ind_start=ind_start,
            ind_inter=ind_inter,
            min=np.percentile(abs(self.proton_reg), 0).astype(float),
            max=np.percentile(
                abs(self.proton_reg), constants._PROTON_PERCENTILE_RESCALE
            ).astype(float),
        )
        io.export_montage_gray(
            image=self.ventilation_cor,
            path=os.path.join(self.data_dir, constants.OutputPaths.VEN_COR_MONTAGE_PNG),
            ind_start=ind_start,
            ind_inter=ind_inter,
            min=np.percentile(abs(self.ventilation_cor), 0).astype(float),
            max=np.percentile(
                abs(self.ventilation_cor), constants._VEN_PERCENTILE_RESCALE
            ).astype(float),
        )

        data = misc.normalize(
            self.ventilation_cor,
            method=constants.NormalizationMethods.PERCENTILE,
            mask=self.mask_reg.astype(bool),
            percentile=constants._VEN_PERCENTILE_RESCALE,
        )[self.mask_reg.astype(bool)]
        io.export_histogram(
            data=data,
            path=os.path.join(self.data_dir, constants.OutputPaths.VEN_HIST_PNG),
            color=constants.VENHISTOGRAMFields.COLOR,
            x_lim=constants.VENHISTOGRAMFields.XLIM,
            y_lim=constants.VENHISTOGRAMFields.YLIM,
            num_bins=constants.VENHISTOGRAMFields.NUMBINS,
            refer_fit=constants.VENHISTOGRAMFields.REFERENCE_FIT,
        )

    def generateHtmlPdf(self):
        """Generate HTML and PDF files."""
        io.export_html_pdf_vent(
            subject_id=self.subject_id,
            data_dir=self.data_dir,
            stats_dict=self.stats_dict,
            scan_type=self.scan_type,
        )

    def generateCSV(self):
        """Generate a CSV file."""
        io.export_csv(
            subject_id=self.subject_id,
            data_dir=self.data_dir,
            stats_dict=self.stats_dict,
        )

    def saveMat(self):
        """Save the instance variables into a mat file."""
        sio.savemat(os.path.join(self.data_dir, self.subject_id + ".mat"), vars(self))

    def savefiles(self):
        """Save select images to nifti files and instance variable to mat."""
        pathOutputcombinedmask = os.path.join(
            self.data_dir, constants.OutputPaths.GRE_MASK_NII
        )
        pathOutputregproton = os.path.join(
            self.data_dir, constants.OutputPaths.GRE_REG_PROTON_NII
        )
        pathOutputvent = os.path.join(
            self.data_dir, constants.OutputPaths.GRE_VENT_RAW_NII
        )
        pathOutputventcor = os.path.join(
            self.data_dir, constants.OutputPaths.GRE_VENT_COR_NII
        )
        pathOutputventbinning = os.path.join(
            self.data_dir, constants.OutputPaths.GRE_VENT_BINNING_NII
        )

        io.export_nii(self.mask_reg, pathOutputcombinedmask)
        io.export_nii(self.proton_reg, pathOutputregproton)
        io.export_nii(self.ventilation_raw, pathOutputvent)
        io.export_nii(self.ventilation_cor, pathOutputventcor)
        io.export_nii(self.ventilation_binning, pathOutputventbinning)
        self.saveMat()
