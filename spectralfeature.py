# -*- coding: utf-8 -*-
"""
Created

научный рабочий
"""

import numpy as np
from scipy.integrate import quad, dblquad
from copy import deepcopy
from scipy import sparse
from scipy.sparse.linalg import spsolve

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

    def __init__(self, wn=np.linspace(0, 1, 129), number_of_peaks=1, lam=1e5) :
        self.specs_array = np.zeros((number_of_peaks, 5))
        self.specs_array[:, 1] = 1 # set default fwhm to 1. Otherwise we can get division by 0
        self.specs_array[:, 0] = (wn[-1]-wn[0])/2
        self.wn = wn
        self.lam = lam
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

    
    def construct_alpha_matrix(self):
        alpha_matrix = np.zeros((np.shape(self.specs_array)[0], np.shape(self.specs_array)[1],
                                 np.shape(self.specs_array)[0], np.shape(self.specs_array)[1]))
        
        S = sparse.diags([1,-2,1],[-1,0,1], shape=(len(self.wn),len(self.wn)))
        w = np.ones(len(self.wn))
        W = sparse.spdiags(w, 0, len(self.wn), len(self.wn)) # diagonal 1-matrix
        
        LSS1 = S.dot(S.transpose()) * self.lam + W

        F_derivatives_array = self.construct_derivatives_over_params()
        
        X_vectors_array = np.zeros(( len(self.wn), np.shape(self.specs_array)[0], np.shape(self.specs_array)[1] ))
        for i in range(self.number_of_peaks):
             for j in range(np.shape(self.specs_array)[1]):
                 # G_vectors_array[:, i, j] = np.matmul(LSS.dot(LSS.transpose()), F_derivatives_array[:, i, j])
                 H_0j = spsolve(LSS1, F_derivatives_array[:, i, j])
                 H_j = H_0j - F_derivatives_array[:, i, j]
                 X_0j = spsolve(LSS1, H_j)
                 X_vectors_array[:, i, j] = X_0j - H_j

        for alpha_i1 in range(np.shape(self.specs_array)[0]): # over number of peaks
            for alpha_j1 in range(np.shape(self.specs_array)[1]): # over paramteres inside the peak
                for alpha_i2 in range(np.shape(self.specs_array)[0]): # over number of peaks
                    for alpha_j2 in range(np.shape(self.specs_array)[1]): # over paramteres inside the peak
                        alpha_matrix[alpha_i1, alpha_j1, alpha_i2, alpha_j2] = np.sum(X_vectors_array[:, alpha_i1, alpha_j1] * 
                                                                                      F_derivatives_array[:, alpha_i2, alpha_j2])
        return alpha_matrix

    def construct_beta(self):
        F_derivatives_array = self.construct_derivatives_over_params()
        S = sparse.diags([1,-2,1],[-1,0,1], shape=(len(self.wn),len(self.wn)))
        w = np.ones(len(self.wn))
        W = sparse.spdiags(w, 0, len(self.wn), len(self.wn)) # diagonal 1-matrix
        
        LSS1 = S.dot(S.transpose()) * self.lam + W
        
        beta_vectors_array = np.zeros(( np.shape(self.specs_array)[0], np.shape(self.specs_array)[1] ))
        # G_vectors_array[:, i, j] = np.matmul(LSS.dot(LSS.transpose()), F_derivatives_array[:, i, j])
        H_0j = spsolve(LSS1, self.baseline)
        H_j = H_0j - self.baseline
        # H_0j = spsolve(LSS1, self.d2baseline)
        # H_j = H_0j - self.d2baseline
        
        # import matplotlib.pyplot as plt
        # import matplotlib as mpl
        # mpl.rcParams['figure.dpi'] = 300
        # mpl.rcParams['figure.figsize'] = [6.0, 3.2]
        # plt.style.use('ggplot')
        # plt.plot(self.wn, H_j); plt.show()
        # plt.plot(self.wn, H_0j); plt.show()

        for i in range(self.number_of_peaks):
             for j in range(np.shape(self.specs_array)[1]):
                 beta_vectors_array[i, j] = np.sum(H_j * F_derivatives_array[:, i, j])
        return beta_vectors_array


    def construct_corrections(self, corrections='main3'):
        """
        main 2:
            corrections to fwhms and amplitudes
        main3:
            also positions """
        alpha_matrix_matrix = self.construct_alpha_matrix()
        if corrections=='main2' or corrections=='main3':
            if corrections=='main2':
                # discard derivatives over positions:
                alpha_matrix_matrix[:, 0, :, :] = 0
                alpha_matrix_matrix[:, :, :, 0] = 0
            # discard derivatives over asym
            alpha_matrix_matrix[:, 2, :, :] = 0
            alpha_matrix_matrix[:, :, :, 2] = 0
            # discard derivatives over gaussian share
            alpha_matrix_matrix[:, 3, :, :] = 0
            alpha_matrix_matrix[:, :, :, 3] = 0

        beta_matrix_vector = self.construct_beta()
        
        npeaks = np.shape(self.specs_array)[0]
        nparams = np.shape(self.specs_array)[1] # 5
        
        alpha_matrix_matrix = alpha_matrix_matrix.reshape(npeaks, nparams, npeaks*nparams)
        alpha_matrix_matrix = alpha_matrix_matrix.reshape(npeaks*nparams, npeaks*nparams)
        
        beta_matrix_vector = beta_matrix_vector.reshape(npeaks*nparams)
        
        # params_corrections_vector = np.linalg.solve(alpha_matrix_matrix, beta_matrix_vector)
        params_corrections_vector,_,_,_ = np.linalg.lstsq(alpha_matrix_matrix, beta_matrix_vector)
        params_corrections_array = params_corrections_vector.reshape(npeaks, nparams)
        return params_corrections_array

    
    def construct_derivatives_over_params(self, shift_over_param=128):
        """ Shift over params by default = 1/128.
            To make it smaller, set, say, shift_over_param=512
            """
        derivatives_array = np.zeros(( len(self.wn), np.shape(self.specs_array)[0], np.shape(self.specs_array)[1] ))
     # 0:    x0 (default 0)
     # 1:    fwhm (defauld 1)
     # 2:    asymmetry (default 0)
     # 3:    Gaussian_share (default 0, i.e. Lorentzian peak)
     # 4:    voigt_amplitude (~area, not height)

        for i in range(self.number_of_peaks):
             for j in range(np.shape(self.specs_array)[1]):
                 derivatives_array[:, i, j] = np.zeros_like(self.wn)
                 if j == 0: # derivative over position:
                     position_shift = (1/shift_over_param) * abs((self.wn[-1]-self.wn[0]) / (len(self.wn)-1)) # 1/shift_over_param of interpoint distance
                     derivatives_array[:, i, j] += self.voigt_amplitude[i] * voigt_asym(self.wn-(self.position[i]+position_shift), self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])
                     derivatives_array[:, i, j] -= self.voigt_amplitude[i] * voigt_asym(self.wn-(self.position[i]-position_shift), self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])
                     derivatives_array[:, i, j] /= (2*position_shift)
                     
                 if j == 1: # derivative over fwhm:
                     fwhm_shift = (1/shift_over_param) * abs(self.fwhm[i]) # 1/shift_over_param of current fwhm
                     derivatives_array[:, i, j] += self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i]+fwhm_shift, self.asymmetry[i], self.Gaussian_share[i])
                     derivatives_array[:, i, j] -= self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i]-fwhm_shift, self.asymmetry[i], self.Gaussian_share[i])
                     derivatives_array[:, i, j] /= (2*fwhm_shift)
                 
                 if j == 2: # derivative over asymmetry:
                     derivatives_array[:, i, j] += self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i]+1/shift_over_param, self.Gaussian_share[i])
                     derivatives_array[:, i, j] -= self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i]-1/shift_over_param, self.Gaussian_share[i])
                     derivatives_array[:, i, j] /= (2*shift_over_param)
                     
                 if j == 3: # derivative over Gaussian share:
                     # shift only one side
                     gs_shift = (1/shift_over_param) * np.sign (0.5-self.Gaussian_share[i])
                     if gs_shift == 0: # in case if Gaussian_share was 0.5 eggzakktly
                         gs_shift = 1/shift_over_param
                     derivatives_array[:, i, j] += self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i]+gs_shift)
                     derivatives_array[:, i, j] -= self.voigt_amplitude[i] * voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])
                     derivatives_array[:, i, j] /= gs_shift

                 if j == 4: # derivative over amplitude is the peak shape divided by the amplitude:
                     derivatives_array[:, i, j] += voigt_asym(self.wn-self.position[i], self.fwhm[i], self.asymmetry[i], self.Gaussian_share[i])

        return derivatives_array
    

    def apply_corrections_to_BL_and_params(self):
        params_corrections_array = self.construct_corrections()
        self.specs_array += params_corrections_array
        