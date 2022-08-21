#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:06:53 2020

@author: korepashki

"""

# settings:
max_number_baseline_iterations = 16
 # The number of iterations in the baseline search


import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import detrend
import matplotlib as mpl
from copy import deepcopy
mpl.rcParams['figure.dpi'] = 300 # default resolution of the plot


class ExpSpec ():
    def __init__(self, full_x, full_y) :
        self.full_x = full_x
        self.full_y = full_y
        self.xrange = (np.min(full_x), np.max(full_x))
        self.x = full_x
        self.y = full_y
        
    @property
    def working_range(self):
        return self.xrange
    @working_range.setter
    def working_range(self, xrange):
        self.xrange = (np.maximum(np.min(xrange), np.min(self.full_x)), np.minimum(np.max(xrange), np.max(self.full_x)))
        self.x = self.full_x[np.where(np.logical_and(self.full_x>=np.amin(xrange), self.full_x<=np.amax(xrange)))]
        self.y = self.full_y[np.where(np.logical_and(self.full_x>=np.amin(xrange), self.full_x<=np.amax(xrange)))]
    
    @property
    def amplitude(self):
        """ returns the span of de-trended spectrum over y-axis"""
        return (np.max(self.y-detrend(self.y)) - np.min(self.y-detrend(self.y)))
        


def baseline_als(x, y, display=2, als_lambda=5e6, als_p_weight=3e-6):
    """ asymmetric baseline correction
    Code by Rustam Guliev ~~ https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    parameters which can be manipulated:
    als_lambda  ~ 5e6
    als_p_weight ~ 3e-6
    (found from optimization with random smooth BL)
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = als_lambda * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(max_number_baseline_iterations):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    #@Test&Debug: # 
    if display > 1:
        plt.plot(x, y - z, 'r', # subtracted spectrum
                  x, y, 'k',    # original spectrum
                  x, baseline, 'b');
        plot_annotation = 'ALS baseline';
        plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes);
        plt.show()
    return baseline   
    
def derpsalsa_baseline(x, y, display=2, als_lambda=5e7, als_p_weight=1.5e-3):
    """ asymmetric baseline correction
    Algorithm by Sergio Oller-Moreno et al.
    Parameters which can be manipulated:
    als_lambda  ~ 5e7
    als_p_weight ~ 1.5e-3
    (found from optimization with random 5-point BL)
    """

    # 0: smooth the spectrum 16 times
    #    with the element of 1/100 of the spectral length:
    zero_step_struct_el = int(2*np.round(len(y)/200) + 1)
    y_sm = molification_smoothing(y, zero_step_struct_el, 16)
    # compute the derivatives:
    y_sm_1d = np.gradient(y_sm)
    y_sm_2d = np.gradient(y_sm_1d)
    # weighting function for the 2nd der:
    y_sm_2d_decay = (np.mean(y_sm_2d**2))**0.5
    weifunc2D = np.exp(-y_sm_2d**2/2/y_sm_2d_decay**2)
    # weighting function for the 1st der:
    y_sm_1d_decay = (np.mean((y_sm_1d-np.mean(y_sm_1d))**2))**0.5
    weifunc1D = np.exp(-(y_sm_1d-np.mean(y_sm_1d))**2/2/y_sm_1d_decay**2)
    
    weifunc = weifunc1D*weifunc2D

    # exclude from screening the edges of the spectrum (optional)
    weifunc[0:zero_step_struct_el] = 1; weifunc[-zero_step_struct_el:] = 1

    # estimate the peak height
    peakscreen_amplitude = (np.max(detrend(y)) - np.min(detrend(y)))/8 # /8 is good, because this is a characteristic height of a tail
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = als_lambda * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    # k = 10 * morphological_noise(y) # above this height the peaks are rejected
    for i in range(max_number_baseline_iterations):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * weifunc * np.exp(-((y-z)/peakscreen_amplitude)**2/2) * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    #@Test&Debug: # 
    if display > 1:
        plt.plot( # x, y - z, 'r',
                  x, y, 'k',
                  x, baseline, 'b');
        plot_annotation = 'derpsalsa baseline';
        plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes);
        plt.show()
    return baseline


def molification_smoothing (rawspectrum, struct_el=5, number_of_molifications=1):
    """ Molifier kernel here is defined as in the work of Koch et al.:
        http://doi.wiley.com/10.1002/jrs.5010
        The structure element is in pixels, not in wn!
        struct_el should be odd integer >= 3
    """
    molifier_kernel = np.linspace(-1, 1, num=struct_el)
    molifier_kernel[1:-1] = np.exp(-1/(1-molifier_kernel[1:-1]**2))
    molifier_kernel[0] = 0; molifier_kernel[-1] = 0
    molifier_kernel = molifier_kernel/np.sum(molifier_kernel)
    denominormtor = np.convolve(np.ones_like(rawspectrum), molifier_kernel, 'same')
    smoothline = rawspectrum
    # i = 0
    for i in range (number_of_molifications) :
        smoothline = np.convolve(smoothline, molifier_kernel, 'same') / denominormtor
        # i += 1
    return smoothline


# def morphological_noise(rawspectrum, min_struct_el=7):
#     thenoise = rawspectrum
#     thenoise = (thenoise - np.roll(thenoise, min_struct_el-2))**2
#     # Remove largest value of thenoise**2
#     #   to take into account the shift (roll),
#     #   and, to keep it symmetrical, also a couple of smallest ones:
#     for i in range (min_struct_el+1) :
#         thenoise = np.delete(thenoise, np.argmax(thenoise))
#         thenoise = np.delete(thenoise, np.argmin(thenoise))
#         thenoise = np.delete(thenoise, np.argmin(thenoise))
#     rmsnoise = (np.average(thenoise))**0.5
#     #@Test&Debug: #
#     print ('morphological noise is ', rmsnoise)
#     return rmsnoise

def morphological_noise(rawspectrum, mollification_width='auto'):
    """ computes mean square deviation of the (spectrum - mollified_spectrum)
    mollification_width by default is ~1/16th of the number-of-ponts"""
    thenoise = (rawspectrum - moving_average_molification(rawspectrum, mollification_width))
    rmsnoise = np.std(thenoise)
    #@Test&Debug: #  
    print ('morphological noise is ', rmsnoise)
    return rmsnoise


def moving_average_molification (rawspectrum, mollification_width='auto', number_of_molifications=1):
    """Moving average mollification
        mollification_width by default is 1/16th of the number-of-ponts;
            mollification_width should be odd!
    """
    if mollification_width=='auto':
        mollification_width = int(2*np.round(len(rawspectrum)/32) + 1)
        #@Test&Debug: #  print('mollification_width by default is', mollification_width )
    molifier_kernel = np.ones(mollification_width)/mollification_width
    denominormtor = np.convolve(np.ones_like(rawspectrum), molifier_kernel, 'same')
    smoothline = deepcopy(rawspectrum)
    for i in range (number_of_molifications) :
        smoothline = np.convolve(smoothline, molifier_kernel, 'same') / denominormtor
    return smoothline
