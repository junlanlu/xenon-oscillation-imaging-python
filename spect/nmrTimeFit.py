"""NMR time fit class.

Notes: 
    1. method for calculating lsqcurve fitting, used dogleg method for optimization
        https://github.com/nmayorov/bounded-lsq
"""
import pdb

import matplotlib
import numpy as np
import scipy.sparse as sps
from bounded_lsq.least_squares import least_squares
from scipy.stats import norm

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from pylab import *


class NMR_Mix:
    """Base Class for curve fitting for spectroscopy."""

    def __init__(
        self, area, freq, phase, fwhm=[], fwhmL=[], fwhmG=[], method="lorenzian"
    ):
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

    def calc_time_sig(self, t):
        n_pts = np.size(t)
        n_fre = int(np.size(self.freq))  # number of frequency componnets
        time_sig = np.zeros(np.shape(t))

        if self.method == "voigt":
            for k in range(0, n_fre):
                # add up all components to time signal using viogt kernel
                time_sig = time_sig + self.area[k] * np.exp(
                    1j * np.pi / 180.0 * self.phase[k]
                    + 1j * 2 * np.pi * t * self.freq[k]
                    - np.pi * t * self.fwhmL[k]
                    - t**2 * 8 * np.log(2) * self.fwhmG[k] ** 2
                )
        else:
            for k in range(0, n_fre):
                # add up all components to time signal using Lorenstian kernel
                time_sig = time_sig + self.area[k] * np.exp(
                    1j * np.pi / 180.0 * self.phase[k]
                    + 1j * 2 * np.pi * t * self.freq[k]
                    - np.pi * t * self.fwhm[k]
                )

        return time_sig

    def componentTimeDomainSignal(self, t):

        """Calculates the time domain signal from the individual components
        of the NMR mix at the given time points (t is in sec). Note,
        this function returns the time signal for each individual
        component. The overall "mix" signal can be obtained with
        calcTimeDomainSignal"""

        n_pts = np.size(t)
        n_fre = int(np.size(self.freq))  # number of frequency componnets
        componentTimeDomainSignal = np.zeros([n_pts, n_fre], dtype=complex)

        componentTimeDomainSignal[:, 0] = (
            self.area[0]
            * np.exp(
                1j * (np.pi / 180.0) * self.phase[0] + 1j * 2 * np.pi * t * self.freq[0]
            )
            * np.exp(-pi * t * self.fwhmL[0])
        )

        for k in range(1, n_fre):
            componentTimeDomainSignal[:, k] = (
                self.area[k]
                * np.exp(
                    1j * (np.pi / 180.0) * self.phase[k]
                    + 1j * 2 * np.pi * t * self.freq[k]
                )
                * np.exp(-(t**2) * 4 * np.log(2) * (self.fwhmG[k] ** 2))
                * np.exp(-np.pi * t * self.fwhmL[k])
            )

        return componentTimeDomainSignal


class NMR_Fit(NMR_Mix):

    """Using NMR_Mix to fit FIDs to a series of exponentially decaying components"""

    def __init__(
        self,
        time_signal,
        t,
        area,
        freq,
        phase,
        fwhm=[],
        fwhmL=[],
        fwhmG=[],
        method="lorenzian",
        line_boardening=0,
        zeropad_size=[],
    ):

        NMR_Mix.__init__(
            self,
            area=area,
            freq=freq,
            fwhm=fwhm,
            phase=phase,
            fwhmL=fwhmL,
            fwhmG=fwhmG,
            method=method,
        )

        # self.ci_area = np.array([])
        # self.ci_freq = np.array([])
        # self.ci_fwhm = np.array([])
        # self.ci_phase = np.array([])

        self.sort_freq()

        self.line_boardening = line_boardening
        self.t = t
        nPts = np.size(t)

        if zeropad_size == []:
            self.zeropad_size = nPts
        else:
            self.zeropad_size = zeropad_size

        # apply line_boardening on the time domain signal
        self.time_signal = np.multiply(
            time_signal, np.exp(-np.pi * self.line_boardening * self.t)
        )

        self.dwell_time = self.t[1] - self.t[0]

        self.spectral_signal = self.dwell_time * np.fft.fftshift(
            np.fft.fft(self.time_signal, self.zeropad_size)
        )

        self.f = np.linspace(-0.5, 0.5, self.zeropad_size + 1) / self.dwell_time
        self.f = self.f[:-1]  # take out last sample to have the right number of samples

    def sort_freq(self):
        # sort components according to resonance frequency
        sort_index = np.argsort(-self.freq)

        self.freq = self.freq[sort_index]
        self.area = self.area[sort_index]
        self.phase = self.phase[sort_index]

        if self.method == "voigt":
            self.fwhmG = self.fwhmG[sort_index]
            self.fwhmL = self.fwhmL[sort_index]
        else:
            self.fwhm = self.fwhm[sort_index]

        # if(self.ci_area>0):
        #     self.ci_area = self.ci_area[sort_index]
        #     self.ci_freq = self.ci_freq[sort_index]
        #     self.ci_fwhm = self.ci_fwhm[sort_index]
        #     self.ci_phase = self.ci_phase[sort_index]

    def reset_components(self, area, freq, phase, fwhm=[], fwhmL=[], fwhmG=[]):
        self.area = np.array([area]).flatten()
        self.freq = np.array([freq]).flatten()
        self.fwhm = np.array([fwhm]).flatten()
        self.phase = np.array([phase]).flatten()

        self.fwhmG = np.array([fwhmG]).flatten()
        self.fwhmL = np.array([fwhmL]).flatten()

        self.sort_freq()


class NMR_TimeFit(NMR_Fit):

    """Using NMR_Mix to fit time domain FIDs to a series of exponentially decaying components"""

    def __init__(
        self,
        time_signal,
        t,
        area,
        freq,
        phase,
        fwhm=[],
        fwhmL=[],
        fwhmG=[],
        method="lorenzian",
        line_boardening=0,
        zeropad_size=[],
    ):
        NMR_Fit.__init__(
            self,
            time_signal=time_signal,
            t=t,
            area=area,
            freq=freq,
            fwhm=fwhm,
            fwhmL=fwhmL,
            fwhmG=fwhmG,
            method=method,
            phase=phase,
            line_boardening=line_boardening,
            zeropad_size=zeropad_size,
        )

    def fit_time_signal(
        self,
        bounds=(-np.inf, np.inf),
        plot_flag=0,
        data_dir="",
        flag_status="skip",
        data_k0=[],
    ):
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

            self.reset_components(
                area=fit_area,
                freq=fit_freq,
                fwhmL=fit_fwhmL,
                fwhmG=fit_fwhmG,
                phase=fit_phase,
            )
        else:
            fit_fwhm = fit_param[2, :]

            self.reset_components(
                area=fit_area, freq=fit_freq, fwhm=fit_fwhm, phase=fit_phase
            )
        # plot time and spect fitting results
        if plot_flag:
            self.plot_time_spect_fit(
                fit_param=fit_param,
                data_dir=data_dir,
                flag_status=flag_status,
                data_k0=data_k0,
            )

    def calc_time_fit(self, bounds):
        ## function fit the time domain signal using least square curve fitting running trust region reflection algorithm

        # asign function to calculate residuals
        fun = self.calc_residual_time_sig
        # max_nfev = 13000
        ftol = 1e-900
        xtol = 1e-20

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
            jac="2-point",
            bounds=bounds,
            method="dogbox",
            xtol=xtol,
            ftol=ftol,
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
                jac="2-point",
                bounds=(-np.inf, np.inf),
                method="dogbox",
                xtol=xtol,
                ftol=ftol,
            )

            fit_param = fit_result["x"].reshape([5, 3])
            fit_freq = fit_param[1, :]

            # check for aliased frequency
            alias_index = np.where(abs(fit_freq) > halfBW)
            n_alias = np.size(alias_index)

        return fit_param

    def calc_residual_time_sig(self, x):
        # used in fitting to allow constraints for complex fitting
        # calculate the residual of fitting
        # x is the fitting independent variables [area, freq, fwhm, phase]
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
                method="lorenzian",
            )

        complex_fit_time = tmpNMRMix.calc_time_sig(self.t)

        fit_sig = np.array([np.real(complex_fit_time), np.imag(complex_fit_time)])

        truth_sig = np.array([np.real(self.time_signal), np.imag(self.time_signal)])

        residual = (truth_sig - fit_sig).flatten()

        return residual

    def calc_residual_time_sig_final(self, x):
        # return both residual and fitting signal (time domain)
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
                method="lorenzian",
            )

        complex_fit_time = tmpNMRMix.calc_time_sig(self.t)
        fit_sig = np.array([np.real(complex_fit_time), np.imag(complex_fit_time)])
        truth_sig = np.array([np.real(self.time_signal), np.imag(self.time_signal)])
        residual = (fit_sig - truth_sig).flatten()
        return residual, complex_fit_time
