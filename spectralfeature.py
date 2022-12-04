# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:32:24 2019

@author: научный рабочий
"""

import numpy as np

class SpectralFeature () :
    """ Abstract spectral feature, with no x-axis defined
     Order of parameters in array:
     0:    x0 (default 0)
     1:    fwhm (defauld 1)
     2:    asymmetry (default 0)
     3:    Gaussian_share (default 0, i.e. Lorentzian peak)
     4:    voigt_amplitude (~area, not height)
     5:    Baseline slope (k) for linear BL
     6:    Baseline offset (b) for linear BL
     
     For MultiPeak there are not BL slope and BL offset, but instead there is an optional baseline
    """

    def __init__(self) :
        self.specs_array = np.zeros(7)
        self.specs_array[1] = 1 # set default fwhm to 1. Otherwise we can get division by 0

    @property
    def position(self):
        return self.specs_array[0]
    @position.setter
    def position (self, position) :
        self.specs_array[0] = position

    @property
    def fwhm(self):
        return self.specs_array[1]
    @fwhm.setter
    def fwhm (self, fwhm) :
        self.specs_array[1] = fwhm
    
    @property
    def asymmetry(self):
        return self.specs_array[2]
    @asymmetry.setter
    def asymmetry (self, asymmetry) :
        self.specs_array[2] = asymmetry

    @property
    def Gaussian_share(self):
        return self.specs_array[3]
    @Gaussian_share.setter
    def Gaussian_share (self, Gaussian_share) :
        self.specs_array[3] = Gaussian_share

    @property
    def voigt_amplitude(self):
        return self.specs_array[4]
    @voigt_amplitude.setter
    def voigt_amplitude (self, voigt_amplitude) :
        self.specs_array[4] = voigt_amplitude

    @property
    def BL_slope(self):
        return self.specs_array[5]
    @BL_slope.setter
    def BL_slope (self, BL_slope) :
        self.specs_array[5] = BL_slope

    @property
    def BL_offset(self):
        return self.specs_array[6]
    @BL_offset.setter
    def BL_offset (self, BL_offset) :
        self.specs_array[6] = BL_offset



class CalcPeak (SpectralFeature) :
    """ Asymmetric peak calculated on x-asis (a grid of wavenumbers).
    It is possible to set a peak height,
        Changing fwhm keeps area same, while changes height.
        Changing height changes area while keeps fwhm.
    """

    def __init__(self, wn=np.linspace(0, 1, 129)) :
        super().__init__()
        self.wn = wn
        self.specs_array[0] = (wn[-1]-wn[0])/2
        self.baseline = np.zeros_like(wn)

    @property
    def peak_area (self) :
        peak_area = (1 - self.specs_array[3]) * self.specs_array[4] * (1 + 0.69*self.specs_array[2]**2 + 1.35 * self.specs_array[2]**4) + self.specs_array[3] * self.specs_array[4] * (1 + 0.67*self.specs_array[2]**2 + 3.43*self.specs_array[2]**4)
        return peak_area

    @property
    def peak_height (self) :
        amplitudes_L = self.specs_array[4]*2/(np.pi*self.specs_array[1])
        amplitudes_G = self.specs_array[4]*(4*np.log(2)/np.pi)**0.5 / self.specs_array[1]
        peak_height = self.specs_array[3] * amplitudes_G + (1-self.specs_array[3]) * amplitudes_L
        return peak_height
    @peak_height.setter
    def peak_height(self, newheight):
        self.specs_array[4] = newheight / (
                                self.specs_array[3] * (4*np.log(2)/np.pi)**0.5 / self.specs_array[1] + (1-self.specs_array[3]) * 2/(np.pi*self.specs_array[1])
                                )

    @property
    def fwhm_asym (self) :
        """ real fwhm of an asymmetric peak"""
        fwhm_asym = self.fwhm * (1 + 0.4*self.asymmetry**2 + 1.35*self.asymmetry**4)
        return fwhm_asym
    
    @property
    def curve (self) :
        """ Asymmetric pseudo-Voigt funtion as defined in Analyst: 10.1039/C8AN00710A
        """
        curve = self.specs_array[4] * voigt_asym(self.wn-self.specs_array[0], self.specs_array[1], self.specs_array[2], self.specs_array[3])
        return curve

    @property
    def curve_with_BL (self):
        curve_with_BL = self.curve + self.specs_array[5] * (self.wn - self.specs_array[0]) + self.specs_array[6]
        return curve_with_BL
    
    @property
    def save_specs_array_to_txt(self, filename='current_spectral_feature.txt'):
        col1 = 'position'
        col2 = 'fwhm'
        col3 = 'asym'
        col4 = 'gauss'
        col5 = 'area'
        the_header = f'{col1:>8}' + f'{col2:>15}' + f'{col3:>15}' + f'{col4:>15}' + f'{col5:>15}'
        np.savetxt(filename,
                   self.specs_array,
                   fmt='% .7e',
                   header = the_header)



def voigt_asym(x, fwhm, asymmetry, Gausian_share):
    """ returns pseudo-voigt profile composed of Gaussian and Lorentzian,
         which would be normalized by unit area if symmetric
         The funtion as defined in Analyst: 10.1039/C8AN00710A"""
    x_distorted = x*(1 - np.exp(-(x)**2/(2*(2*fwhm)**2))*asymmetry*x/fwhm)
    Lor_asym = fwhm / (x_distorted**2+fwhm**2/4) / (2*np.pi)
    Gauss_asym = (4*np.log(2)/np.pi)**0.5/fwhm * np.exp(-(x_distorted**2*4*np.log(2))/fwhm**2)
    voigt_asym = (1-Gausian_share)*Lor_asym + Gausian_share*Gauss_asym
    return voigt_asym


class MultiPeak ():
    """ Abstract spectral feature, with no x-axis defined
     Order of parameters in array:
     0:    x0 (default 0)
     1:    fwhm (defauld 1)
     2:    asymmetry (default 0)
     3:    Gaussian_share (default 0, i.e. Lorentzian peak)
     4:    voigt_amplitude (~area, not height)

     Asymmetric peaks calculated on x-asis (a grid of wavenumbers).
    It is possible to set a peak height,
        Changing fwhm keeps area same, while changes height.
        Changing height changes area while keeps fwhm.
    """

    def __init__(self, wn=np.linspace(0, 1, 129), number_of_peaks=1) :
        self.specs_array = np.zeros((number_of_peaks, 5))
        self.specs_array[:, 1] = 1 # set default fwhm to 1. Otherwise we can get division by 0
        self.specs_array[:, 0] = (wn[-1]-wn[0])/2
        self.wn = wn
        self.number_of_peaks = number_of_peaks
        self.linear_baseline_scalarpart = np.zeros_like(wn)
        self.linear_baseline_slopepart = np.zeros_like(wn)
        # self.baseline = np.zeros_like(wn)
        self.d2baseline = np.zeros_like(wn)
        # self.linear_baseline_offset = 0
        # self.linear_baseline_slope = 0
        
    @property
    def position(self):
        return self.specs_array[:, 0]
    @position.setter
    def position (self, position) :
        self.specs_array[:, 0] = position

    @property
    def fwhm(self):
        return self.specs_array[:, 1]
    @fwhm.setter
    def fwhm (self, fwhm) :
        self.specs_array[:, 1] = fwhm
    
    @property
    def asymmetry(self):
        return self.specs_array[:, 2]
    @asymmetry.setter
    def asymmetry (self, asymmetry) :
        self.specs_array[:, 2] = asymmetry

    @property
    def Gaussian_share(self):
        return self.specs_array[:, 3]
    @Gaussian_share.setter
    def Gaussian_share (self, Gaussian_share) :
        self.specs_array[:, 3] = Gaussian_share

    @property
    def voigt_amplitude(self):
        return self.specs_array[:, 4]
    @voigt_amplitude.setter
    def voigt_amplitude (self, voigt_amplitude) :
        self.specs_array[:, 4] = voigt_amplitude[:]

    @property
    def peak_area (self) :
        peak_area = (1 - self.Gaussian_share) * self.voigt_amplitude * (1 + 0.69*self.asymmetry**2 + 1.35 * self.asymmetry**4) + self.Gaussian_share * self.voigt_amplitude * (1 + 0.67*self.asymmetry**2 + 3.43*self.asymmetry**4)
        return peak_area

    @property
    def peak_height (self):
        #@Test&Debug # # print('calling getter')
        return self.specs_array[:, 3] * self.specs_array[:, 4]*(4*np.log(2)/np.pi)**0.5 / self.specs_array[:, 1] + (1-self.specs_array[:, 3]) * self.specs_array[:, 4]*2/(np.pi*self.specs_array[:, 1])
    @peak_height.setter # works only with array, not with a single element!
    def peak_height (self, peak_height):
        #@Test&Debug # # print('calling setter')
        self.specs_array[:, 4] = peak_height[:] / (
            self.specs_array[:, 3] * (4*np.log(2)/np.pi)**0.5 / self.specs_array[:, 1] + (1-self.specs_array[:, 3]) * 2/(np.pi*self.specs_array[:, 1])
            )

    @property
    def fwhm_asym (self) :
        """ real fwhm of an asymmetric peak"""
        fwhm_asym = self.fwhm * (1 + 0.4*self.asymmetry**2 + 1.35*self.asymmetry**4)
        return fwhm_asym

    # def linear_baseline_slopepart(self):
    #     """ Returns part of linear baseline.
    #     Coefficients are defined in terms of centered x-axis with unit span.
    #     """
    #     wn_center = (self.wn[0] + self.wn[-1])/2
    #     linear_baseline_slopepart = self.linear_baseline_slope * (self.wn - wn_center) / abs(self.wn[0] - self.wn[-1])
    #     return linear_baseline_slopepart
    
    # def linear_baseline_scalarpart(self):
    #     """ Returns part of linear baseline.
    #     Coefficients are defined in terms of centered x-axis with unit span.
    #     """
    #     return self.linear_baseline_offset

    @property
    def linear_baseline(self):
        """ Returns linear baseline.
        Coefficients are defined in terms of centered x-axis with unit span.
        """
        return self.linear_baseline_scalarpart + self.linear_baseline_slopepart
    
    @property
    def baseline(self):
        return self.linear_baseline + self.d2baseline

    @property
    def curve(self):
        """ Asymmetric pseudo-Voigt funtion as defined in Analyst: 10.1039/C8AN00710A
        """
        curve = np.zeros((len(self.wn), self.number_of_peaks))
        for i in range(self.number_of_peaks):
            curve[:,i] = self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])
        curve = np.sum(curve, axis=1)
        return curve



    @property
    def multicurve (self) :
        """ array of curves
        """
        multicurve = np.zeros((len(self.wn), self.number_of_peaks))
        for i in range(self.number_of_peaks):
            multicurve[:,i] = self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])
        return multicurve

    def save_specs_array_to_txt(self, filename='current_peakparams.txt'):
        col1 = 'position'
        col2 = 'fwhm'
        col3 = 'asym'
        col4 = 'gauss'
        col5 = 'area'
        the_header = f'{col1:>8}' + f'{col2:>15}' + f'{col3:>15}' + f'{col4:>15}' + f'{col5:>15}'
        np.savetxt(filename,
                   self.specs_array,
                   fmt='% .7e',
                   header = the_header)
    
    def write_decomposition_to_txt(self, exp_y='zero', filename='current_decomp.txt'):
        # save decomposition to txt
        the_header = '' 
        col1 = ' x axis'
        col2 = ' raw data'
        col3 = ' baseline'
        the_header += f'{col1:<25}' + f'{col2:<25}' + f'{col3:<25}'
        for current_peak in range(self.number_of_peaks):
            col_n = ' peak at {:.1f}'.format(self.specs_array[current_peak,0])
            the_header += f'{col_n:<25}'
        
        if exp_y=='zero':
            exp_y = np.zeros_like(self.wn)
        current_decomposition = np.column_stack((self.wn, exp_y, self.baseline))
        current_decomposition = np.column_stack((current_decomposition, self.multicurve))
        np.savetxt(filename,
                   current_decomposition,
                   fmt='% .17e',
                   header = the_header)

