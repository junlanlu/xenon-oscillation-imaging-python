"""NMR time fit class."""
import logging
import pdb
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


class NMR_Mix:
    """Base Class for curve fitting for spectroscopy.

    Attributes:
        area (np.ndarray): Area of each component.
        freq (np.ndarray): Frequency of each component.
        fwhm (np.ndarray): FWHM of each component.
        phase (np.ndarray): Phase of each component.
        fwhmL (np.ndarray): Lorentzian FWHM of each component.
        fwhmG (np.ndarray): Gaussian FWHM of each component.
        method (str): Method for fitting, either "voigt" or "lorentzian".
        ncomp (int): Number of components.
    """

    def __init__(
        self,
        area: np.ndarray,
        freq: np.ndarray,
        phase: np.ndarray,
        fwhm: np.ndarray = np.array([]),
        fwhmL: np.ndarray = np.array([]),
        fwhmG: np.ndarray = np.array([]),
        method="voigt",
    ):
        """Initialize NMR_Mix class."""
        self.area = np.array([area]).flatten()
        self.freq = np.array([freq]).flatten()
        self.fwhm = np.array([fwhm]).flatten()
        self.phase = np.array([phase]).flatten()
        self.fwhmL = np.array([fwhmL]).flatten()
        self.fwhmG = np.array([fwhmG]).flatten()
        self.method = method

        if self.method == "voigt":
            self.ncomp = 5
        else:
            self.ncomp = 4

    def get_time_function(self, tdata: np.ndarray):
        """Get time function for given time points.

        Available for both voigt and lorentzian fitting.

        Args:
            t (np.ndarray): Time points in seconds.
        """
        n_fre = int(np.size(self.freq))
        time_sig = np.zeros(np.shape(tdata))

        if self.method == "voigt":
            for k in range(0, n_fre):
                time_sig = time_sig + self.area[k] * np.exp(
                    1j * np.pi / 180.0 * self.phase[k]
                    + 1j * 2 * np.pi * tdata * self.freq[k]
                    - np.pi * tdata * self.fwhmL[k]
                    - tdata**2 * 8 * np.log(2) * self.fwhmG[k] ** 2
                )
        else:
            for k in range(0, n_fre):
                time_sig = time_sig + self.area[k] * np.exp(
                    1j * np.pi / 180.0 * self.phase[k]
                    + 1j * 2 * np.pi * tdata * self.freq[k]
                    - np.pi * tdata * self.fwhm[k]
                )
        return time_sig

    def sort_freq(self):
        """Sort components according to resonance frequency in descending order."""
        sort_index = np.argsort(-self.freq)
        self.freq = self.freq[sort_index]
        self.area = self.area[sort_index]
        self.phase = self.phase[sort_index]

        if self.method == "voigt":
            self.fwhmG = self.fwhmG[sort_index]
            self.fwhmL = self.fwhmL[sort_index]
        else:
            self.fwhm = self.fwhm[sort_index]

    def set_components(
        self,
        area: np.ndarray,
        freq: np.ndarray,
        phase: np.ndarray,
        fwhm: np.ndarray = np.array([]),
        fwhmL: np.ndarray = np.array([]),
        fwhmG: np.ndarray = np.array([]),
    ):
        """Set components and sort frequencies in descending order."""
        self.area = np.array([area]).flatten()
        self.freq = np.array([freq]).flatten()
        self.fwhm = np.array([fwhm]).flatten()
        self.phase = np.array([phase]).flatten()
        self.fwhmG = np.array([fwhmG]).flatten()
        self.fwhmL = np.array([fwhmL]).flatten()
        self.sort_freq()


class NMR_TimeFit(NMR_Mix):
    """Class to fit time domain FIDs to a series of exponentially decaying components.

    Attributes:
        ydata (np.ndarray): Time domain data.
        tdata (np.ndarray): Time points in seconds.
        area (np.ndarray): Area of each component.
        freq (np.ndarray): Resonance frequency of each component.
        phase (np.ndarray): Phase of each component.
        fwhm (np.ndarray): FWHM of each component.
        fwhmL (np.ndarray): Lorentzian FWHM of each component.
        fwhmG (np.ndarray): Gaussian FWHM of each component.
        method (str): Method for fitting, either "voigt" or "lorentzian".
        line_broadening (float): Line broadening in Hz.
        zeropad_size (Optional[int]): Zero padding size.
        dwell_time (float): Dwell time in seconds.
        spectral_signal (np.ndarray): Spectral signal.
        f (np.ndarray): Frequency points in Hz.
    """

    def __init__(
        self,
        ydata: np.ndarray,
        tdata: np.ndarray,
        area: np.ndarray,
        freq: np.ndarray,
        phase: np.ndarray,
        fwhm: np.ndarray = np.array([]),
        fwhmL: np.ndarray = np.array([]),
        fwhmG: np.ndarray = np.array([]),
        method: str = "lorentzian",
        line_broadening: float = 0,
        zeropad_size: Optional[int] = None,
    ):
        """Initialize NMR_TimeFit class.

        Args:
            ydata (np.ndarray): Time domain data.
            tdata (np.ndarray): Time points in seconds.
            area (np.ndarray): Area of each component.
            freq (np.ndarray): Resonance frequency of each component.
            phase (np.ndarray): Phase of each component.
            fwhm (np.ndarray): FWHM of each component.
            fwhmL (np.ndarray): Lorentzian FWHM of each component.
            fwhmG (np.ndarray): Gaussian FWHM of each component.
            method (str): Fitting method, either "voigt" or "lorentzian".
            line_broadening (float): Line broadening in Hz.
            zeropad_size (Optional[int]): Zero padding size.
        """
        super().__init__(
            area=area,
            freq=freq,
            fwhm=fwhm,
            fwhmL=fwhmL,
            fwhmG=fwhmG,
            method=method,
            phase=phase,
        )
        self.line_broadening = line_broadening
        self.tdata = tdata
        if not zeropad_size:
            self.zeropad_size = np.size(tdata)
        else:
            self.zeropad_size = zeropad_size
        # apply line broadening on the time domain signal
        self.ydata = np.multiply(
            ydata, np.exp(-np.pi * self.line_broadening * self.tdata)
        )
        # calculate dwell time from delta tdata
        self.dwell_time = self.tdata[1] - self.tdata[0]
        self.spectral_signal = self.dwell_time * np.fft.fftshift(
            np.fft.fft(self.ydata, self.zeropad_size)
        )
        self.f = np.linspace(-0.5, 0.5, self.zeropad_size + 1) / self.dwell_time
        # take out last sample to have the right number of samples
        self.f = self.f[:-1]
        self.sort_freq()

    def calc_time_fit(self, bounds):
        """Fit the time domain signal using least square curve fitting.

        Running trust region reflection algorithm.
        Args:
            bounds (list): Bounds for the fitting parameters.
        """
        fun = self.get_residual_time_function
        if self.method == "voigt":
            x0 = np.array(
                [self.area, self.freq, self.fwhmL, self.fwhmG, self.phase]
            ).flatten()
        else:
            x0 = np.array([self.area, self.freq, self.fwhm, self.phase]).flatten()
        # curve fitting using trust region reflection algorithm
        fit_result = least_squares(
            fun=fun,
            x0=x0,
            jac="3-point",
            bounds=bounds,
            method="lm",
            ftol=1e-15,
            xtol=1e-09,
        )
        # resolving the fitting results
        fit_param = fit_result["x"]
        n_fre = int(np.size(fit_param) / self.ncomp)
        fit_param = np.reshape(fit_param, (self.ncomp, n_fre))
        fit_freq = fit_param[1, :]
        # check for aliased frequency
        halfBW = 0.5 * (np.amax(self.f) - np.amin(self.f))
        alias_index = np.where(abs(fit_freq) > halfBW)
        n_alias = np.size(alias_index)

        while n_alias > 0:
            for k in range(0, n_alias):
                idx = alias_index[k]
                while fit_freq[idx] < -halfBW:
                    fit_freq[idx] = fit_freq[idx] + 2 * halfBW
                while fit_freq[idx] > halfBW:
                    fit_freq[idx] = fit_freq[idx] - 2 * halfBW

            # fit again after the alias frequency is solved
            x0 = fit_param[:]
            x0[1, :] = fit_freq  # replace the frequency to be within the range
            x0 = x0.flatten()

            fit_result = least_squares(
                fun=fun,
                x0=x0,
                jac="3-point",
                bounds=(-np.inf, np.inf),
                method="trf",
            )

            fit_param = fit_result["x"].reshape([5, 3])
            fit_freq = fit_param[1, :]

            # check for aliased frequency
            alias_index = np.where(abs(fit_freq) > halfBW)
            n_alias = np.size(alias_index)

        return fit_param

    def get_residual_time_function(self, x):
        """Calculate the residual of fitting.

        Args:
            x (np.ndarray): Fitting parameters of shape [area, freq, fwhmL, fwhmG,
             phase]
        """
        if self.method == "voigt":
            x = np.reshape(x, (5, int(np.size(x) / 5)))
            tmpNMRMix = NMR_Mix(
                area=x[0, :],
                freq=x[1, :],
                fwhmL=x[2, :],
                fwhmG=x[3, :],
                phase=x[4, :],
                method="voigt",
            )
        else:
            x = np.reshape(x, (4, int(np.size(x) / 4)))
            tmpNMRMix = NMR_Mix(
                area=x[0, :],
                freq=x[1, :],
                fwhm=x[2, :],
                phase=x[3, :],
                method="lorentzian",
            )
        complex_fit_time = tmpNMRMix.get_time_function(self.tdata)
        fit_sig = np.array([np.real(complex_fit_time), np.imag(complex_fit_time)])
        truth_sig = np.array([np.real(self.ydata), np.imag(self.ydata)])
        # residual = (fitruth_sig - fit_sig).flatten()
        residual = (fit_sig - truth_sig).flatten()
        return residual

    def fit_time_signal(
        self,
        bounds: tuple = (-np.inf, np.inf),
    ):
        """Fit the time domain signal using least square curve fitting.

        Also store the fitting results in the class.

        Args:
            bounds (tuple, optional): Bounds for the fitting parameters.
                Defaults to (-np.inf, np.inf).
        """
        fit_param = self.calc_time_fit(bounds)
        # parsing the fitting results
        fit_vec = np.multiply(
            fit_param[0, :], np.exp(1j * np.pi * fit_param[-1, :] / 180.0)
        )
        fit_area = abs(fit_vec)
        fit_freq = fit_param[1, :]
        fit_phase = np.arctan2(np.imag(fit_vec), np.real(fit_vec)) * 180.0 / np.pi

        if self.method == "voigt":
            fit_fwhmL = fit_param[2, :]
            fit_fwhmG = fit_param[3, :]

            self.set_components(
                area=fit_area,
                freq=fit_freq,
                fwhmL=fit_fwhmL,
                fwhmG=fit_fwhmG,
                phase=fit_phase,
            )
        elif self.method == "lorentzian":
            fit_fwhm = fit_param[2, :]
            self.set_components(
                area=fit_area, freq=fit_freq, fwhm=fit_fwhm, phase=fit_phase
            )
        else:
            raise ValueError("Unknown fitting method.")
