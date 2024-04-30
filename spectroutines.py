#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Technical procedures

"""

import numpy as np
from copy import deepcopy
import colorsys
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.signal import find_peaks
from scipy.signal import detrend
from expspec import *

def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def is_number(n):
    """
    'nan' and np.nan are considered as numbers !
    """
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True


def molification_smoothing (rawspectrum, struct_el=5, number_of_molifications=1):
    """ Molifier kernel here is defined as in the work of Koch et al.:
        http://doi.wiley.com/10.1002/jrs.5010
        The structure element is in pixels, not in wn!
            struct_el should be odd integer >= 3
        rawspectrum is a vector (y-component of a spectrum)
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

def morphological_noise(rawspectrum, mollification_width='auto'):
    """ computes mean square deviation of the (spectrum - mollified_spectrum)
    mollification_width by default is ~1/16th of the number-of-ponts"""
    thenoise = (rawspectrum - moving_average_molification(rawspectrum, mollification_width))
    rmsnoise = np.std(thenoise)
    #@Test&Debug: #  
    # print ('morphological noise is ', rmsnoise)
    # EndOf @Test&Debug: #  
    return rmsnoise


def moving_average_molification (rawspectrum, mollification_width='auto', number_of_molifications=1):
    """Moving average mollification
        mollification_width by default is 1/16th of the number-of-ponts;
            mollification_width should be odd!
       rawspectrum is a vector (y-component of a spectrum)
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


def das_baseline(thespectrum, # , x_axis=None,
                 display=0,
                 als_lambda='auto0',
                 als_p_weight=1.5e-3,
                 max_number_baseline_iterations=16,
                 return_weights=False):
    """
    das_baseline = double-auto spectral baseline
    
    """

    if als_lambda=='auto0':
        als_lambda = (len(thespectrum.y)/16)**4

    # 0: smooth the spectrum 16 times
    #    with the element of 1/64 of the spectral length:
    zero_step_struct_el = int(2*np.round(len(thespectrum.y)/128) + 1)
    y_sm = moving_average_molification(thespectrum.y, zero_step_struct_el, 16) #

    # compute the derivatives:
    y_sm_1d = np.gradient(y_sm)
    y_sm_2d = np.gradient(y_sm_1d)

    # weighting function for the 2nd derivative:
    # y_sm_2d_decay = np.std(y_sm_2d) / 2
    y_sm_2d_decay = ( np.mean(y_sm_2d**4) )**0.25 / 4
    weifunc2D = np.exp(-y_sm_2d**2/2/y_sm_2d_decay**2)
    # weighting function for the 1st derivative:
    # y_sm_1d_decay = np.std(y_sm_1d) / 2
    y_sm_1d_decay = ( np.mean((y_sm_1d-np.mean(y_sm_1d))**4) )**0.25 / 4
    weifunc1D = np.exp(-(y_sm_1d-np.mean(y_sm_1d))**2/2/y_sm_1d_decay**2)
    
    # weighting function for both derivatives:
    weifunc = weifunc1D*weifunc2D

    # optional: exclude from screening the edges of the spectrum
    # weifunc[0:zero_step_struct_el] = 1; weifunc[-zero_step_struct_el:] = 1

    # estimate the peak height for the screening by the amplitude
    peakscreen_amplitude = np.std(thespectrum.y-y_sm) / 2

    L = len(thespectrum.y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = als_lambda * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(max_number_baseline_iterations):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*thespectrum.y)
        w = als_p_weight * weifunc * np.exp(-((thespectrum.y-z)/peakscreen_amplitude)**2/2) * (thespectrum.y > z) + (1-als_p_weight) * (thespectrum.y < z)
        w /= np.sqrt(np.mean(w**2))
        # peakscreen_amplitude = np.std(thespectrum.y-z) / 2
        peakscreen_amplitude = ( np.mean((thespectrum.y-z)**4 ) )**0.25 / 4
        y_sm = moving_average_molification(thespectrum.y-z, zero_step_struct_el, 16) #
        y_sm_1d = np.gradient(y_sm)
        y_sm_2d = np.gradient(y_sm_1d)
        # y_sm_2d_decay = np.std(y_sm_2d) / 2
        y_sm_2d_decay = ( np.mean(y_sm_2d**4) )**0.25 / 4
        weifunc2D = np.exp(-y_sm_2d**2/2/y_sm_2d_decay**2)
        # y_sm_1d_decay = np.std(y_sm_1d) / 2
        y_sm_1d_decay = ( np.mean((y_sm_1d-np.mean(y_sm_1d))**4) )**0.25 / 4
        weifunc1D = np.exp(-(y_sm_1d-np.mean(y_sm_1d))**2/2/y_sm_1d_decay**2)
        weifunc = weifunc1D*weifunc2D
        # optional: exclude from screening the edges of the spectrum
        # weifunc[0:zero_step_struct_el] = 1; weifunc[-zero_step_struct_el:] = 1
    baseline = z
    datweight = weifunc * np.exp(-((thespectrum.y-z)/peakscreen_amplitude)**2/2)
    #@Test&Debug: # 
    if display > 1:
        plt.plot(thespectrum.x, thespectrum.y, 'k', linewidth=1, label='the spectrum')
        plt.plot(thespectrum.x, baseline, 'b', linewidth=1, label='baseline')
        plt.plot(thespectrum.x, weifunc*(np.max(thespectrum.y)-np.min(thespectrum.y))+np.min(thespectrum.y), ':r', linewidth=0.5, label='w for derivatives')
        plt.plot(thespectrum.x, datweight*(np.max(thespectrum.y)-np.min(thespectrum.y))+np.min(thespectrum.y), '0.5', linewidth=1, label='w total')
        plt.title('d.a.s. baseline,  ' + r'$\lambda$' + ' = {:.2e}'.format(als_lambda) + ', p = {:.2e}'.format(als_p_weight), fontsize=8)
        plt.legend(fontsize=8, framealpha=0.09)
        plt.tick_params(labelleft=False)
        save_it_path = 'current_output/baseline'
        Path(save_it_path).mkdir(parents=True, exist_ok=True)
        plt.grid(color='0.4', linestyle=':', linewidth=0.5)
        plt.xlabel('wavenumber  / cm$^{-1}$')
        plt.ylabel('intensity')
        plt.savefig(save_it_path + '/baseline.png', bbox_inches='tight', transparent=True)
        plt.savefig(save_it_path + '/baseline.eps', bbox_inches='tight', transparent=True)
        plt.show()

    thespectrum.baseline = baseline
    if return_weights:
        return baseline, datweight
    else:
        return baseline


def spectral_powers(thespectrum,
                    return_only_d2bl=True,
                    check_if_apodized=True,
                    gridlength=129,
                    display=0):
    """
    If you need a detailed output, set return_only_d2bl=False
    
    """
    
    if display > 1:
        return_only_d2bl=False
    
    if not return_only_d2bl:
        save_it_path = 'current_output/spectral_powers'
        Path(save_it_path).mkdir(parents=True, exist_ok=True)

    # compute grid for wavelet:
    scalegrid_max = np.log(len(thespectrum.y)-1)
    scalegrid_adv = np.linspace(-0.2, scalegrid_max, gridlength)
    scalegrid_adv = np.exp(scalegrid_adv)

    # subtract quasi-linear background from the spectrum:
    thespectrum_detrended = thespectrum.y - das_baseline(thespectrum,
                                                  als_lambda = (len(thespectrum.y)/4)**4,
                                                  als_p_weight=0.5,
                                                  display=display)

    # compute wavelet transform:
    coef = signal.cwt(thespectrum_detrended, signal.ricker, scalegrid_adv)


    # integrate the reconstruction wrt scalegrid_adv (y-axis):
    reconstruction_factor = np.ones_like(scalegrid_adv) / scalegrid_adv**0.5
    spectral_powers = np.trapz((coef*reconstruction_factor[:, np.newaxis])**2, axis=1)
    reconstructed_spectrum = np.trapz((coef*reconstruction_factor[:, np.newaxis]), axis=0)
    
    scalefactor = np.linalg.norm(thespectrum_detrended) / np.linalg.norm(reconstructed_spectrum)
    reconstructed_spectrum *= scalefactor

    # find the cutoff for the baseline:
    #  it gatta be something like the maximum negative gradient of the spectral powers
    spectral_power_gradient = np.gradient(spectral_powers) / np.gradient(np.log(scalegrid_adv)) # <<<<<<<<<<< this one !11
    # for the baseline, we are intereted in the scale > 1/16 of the spectral range    
    valid_idx = np.where(scalegrid_adv > scalegrid_adv[-1] / 16)[0]
    baseline_cutoff_index  = valid_idx[spectral_power_gradient[valid_idx].argmin()]
    # alternative criterion:
    # baseline_cutoff_index  = valid_idx[spectral_powers[valid_idx].argmax()]

    # find the cutoff for the noise:
    #  it gatta be something like 1/4 of the maximum positive gradient of the spectral powers
    noise_cutoff_index_from_wavelets = np.argmax(np.gradient(spectral_powers)) / 4
    noise_cutoff_index_from_wavelets = int(np.ceil(noise_cutoff_index_from_wavelets))

    if check_if_apodized:
        if len(thespectrum.y) > 2048:
            # set the noise cutoff to 3 pixels of some average 1024-pixel CCD:
            possible_noise_cutoff_index = np.argmin(abs(len(thespectrum.y)/1024*3 - scalegrid_adv))
            if not return_only_d2bl:
                print('the spectrum seems apodized, adjusting cutoff')
        else:
            possible_noise_cutoff_index = np.argmin(abs(3 - scalegrid_adv))
            if not return_only_d2bl:
                print('the spectrum seems not apodized')
    else:
        # also set it to ~3 pixels
        possible_noise_cutoff_index = np.argmin(abs(3 - scalegrid_adv))
        
    noise_cutoff_index = np.max((noise_cutoff_index_from_wavelets, possible_noise_cutoff_index))

    baseline_reconstruction_range = baseline_cutoff_index, len(scalegrid_adv)
    baseline_reconstructed = np.trapz(coef[baseline_reconstruction_range[0]:baseline_reconstruction_range[1],:] *
                                      reconstruction_factor[baseline_reconstruction_range[0]:baseline_reconstruction_range[1], np.newaxis],
                                      axis=0)
    baseline_reconstructed *= scalefactor
    
    signal_reconstruction_range = noise_cutoff_index, baseline_cutoff_index+1
    signal_reconstructed = np.trapz(coef[signal_reconstruction_range[0]:signal_reconstruction_range[1],:] *
                                      reconstruction_factor[signal_reconstruction_range[0]:signal_reconstruction_range[1], np.newaxis],
                                      axis=0)
    signal_reconstructed *= scalefactor
    
    noise_reconstructed = thespectrum_detrended - signal_reconstructed - baseline_reconstructed

    # polyfy the reconstructed baseline:
    x = np.linspace(-1, 1, num=len(thespectrum.y))
    bl_polycoeff = polyfit(x, baseline_reconstructed, 15)
    bl_poly = polyval(x, bl_polycoeff)
    D2BL = (np.mean(np.diff(bl_poly, 2)**2))**0.5
    
    if not return_only_d2bl:
        print('noise reconstruction cutoff =', scalegrid_adv[noise_cutoff_index])
        print('baseline reconstruction cutoff =', scalegrid_adv[baseline_cutoff_index])
        print('rms 2nd derivative of BL: {:.2e}'.format(D2BL))
        
        # display wavelet coefficients
        theaspect = len(thespectrum.y)/len(scalegrid_adv)/1.875
        plt.matshow(coef, cmap=plt.cm.nipy_spectral_r, aspect = theaspect)
        # plt.title('wavelet coefficients', fontsize=7)
        plt.ylabel('wavelet scale / px', fontsize=7)
        plt.xlabel('pixel number', fontsize=7)
        locs, _ = plt.yticks()
        locs = locs[locs <= len(scalegrid_adv)]
        locs = locs[locs >= 0]
        labels = scalegrid_adv[locs.astype(int)]
        labels = (np.floor(labels)).astype(int)
        plt.yticks(locs, labels)
        plt.yticks(fontsize=7)
        plt.axhline(noise_cutoff_index, ls=':', color='white', lw=2)
        plt.axhline(baseline_cutoff_index, ls=':', color='white', lw=2)
        plt.grid(color='0.4', linestyle=':', linewidth=0.5)
        ax = plt.gca()
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=False)
        plt.xticks(fontsize=7)
        plt.savefig(save_it_path + '/wavelet_coefficients.png', bbox_inches='tight', transparent=True)
        plt.show()


        plt.semilogx(scalegrid_adv, spectral_powers, 'o', mfc='none', ms = 4, mec='k')
        plt.axvline(scalegrid_adv[noise_cutoff_index], ymin=0.04, ymax=0.988, ls=':', c='r', lw=2)
        plt.axvline(scalegrid_adv[baseline_cutoff_index], ymin=0.04, ymax=0.988, ls=':', c='r', lw=2)
        plt.tick_params(labelleft=False)
        # plt.title('spectral powers', fontsize=8)
        plt.xlabel('wavelet scale / px')
        plt.ylabel('spectral power')
        plt.grid(color='0.4', linestyle=':', linewidth=0.5)
        plt.savefig(save_it_path + '/spectral_powers.png', bbox_inches='tight', transparent=True)
        plt.show()

        plt.semilogx(scalegrid_adv, np.gradient(spectral_powers), 'o', mfc='none', ms = 4, mec='k')
        plt.axvline(scalegrid_adv[noise_cutoff_index], ymin=0.04, ymax=0.988, ls=':', c='r', lw=2)
        plt.axvline(scalegrid_adv[baseline_cutoff_index], ymin=0.04, ymax=0.988, ls=':', c='r', lw=2)
        plt.tick_params(labelleft=False)
        # plt.title('spectral power gradient', fontsize=8)
        plt.xlabel('wavelet scale / px')
        plt.ylabel('gradient')
        plt.grid(color='0.4', linestyle=':', linewidth=0.5)
        plt.savefig(save_it_path + '/spectral_power_gradient.png', bbox_inches='tight', transparent=True)
        plt.show()
        
        plt.plot(thespectrum.x, thespectrum_detrended, '0.5', label='raw')
        plt.plot(thespectrum.x, baseline_reconstructed, 'g', linewidth=1, label='baseline reconstructed')
        plt.plot(thespectrum.x, signal_reconstructed, 'b', linewidth=1, label='signal reconstructed')
        plt.plot(thespectrum.x, noise_reconstructed, 'r', linewidth=1, label='noise reconstructed')
        plt.legend(fontsize=8, framealpha=0.09)
        plt.tick_params(labelleft=False)
        # plt.title('wavelet decomposition', fontsize=8)
        # plt.ylim(-80, 640) # for HR of water
        plt.xlabel('wavenumber  / cm$^{-1}$')
        plt.ylabel('intensity')
        plt.grid(color='0.4', linestyle=':', linewidth=0.5)
        plt.savefig(save_it_path + '/wavelet_decomposition.png', bbox_inches='tight', transparent=True)
        plt.show()


    total_spectral_power = np.sum(noise_reconstructed**2) + np.sum(signal_reconstructed**2) + np.sum(baseline_reconstructed**2) # np.sum(detrend(thespectrum)**2)
    noise_power = np.sum(noise_reconstructed**2) / total_spectral_power
    signal_power = np.sum(signal_reconstructed**2) / total_spectral_power
    bl_power = np.sum(baseline_reconstructed**2) / total_spectral_power
    
    if not return_only_d2bl:
        print('noise: {:.3f}, signal: {:.3f}, baseline: {:.3f}'.format(noise_power, signal_power, bl_power))
        print('p = {:.4f}'.format(0.5*noise_power**0.5))
        print('p_e = {:.4f}'.format(0.5*(1-np.exp(-noise_power**2))))

    thespectrum.d2bl = D2BL
    return D2BL


def find_both_optimal_parameters(testspec, display=2):
    d2bl = spectral_powers(testspec, display=display) # auto_D2BL(testspec.y, display=display)

    def func2min(lam_and_p): # params = lam, p
        lam_and_p = np.exp(lam_and_p)
        current_bl, current_weights = das_baseline(testspec,
                                                   als_lambda=lam_and_p[0],
                                                   als_p_weight=lam_and_p[1],
                                                   display=0,
                                                   return_weights=True)
        current_residuals = testspec.y - current_bl
        current_residuals_weighted = current_residuals * current_weights
        negativeresiduals = current_residuals_weighted * (testspec.y < current_bl)
        d2bl_current = (np.mean(np.diff(current_bl, 2)**2))**0.5
        neg_residuals_without_positive = negativeresiduals[np.where(testspec.y < current_bl)]
        weighted_positive_residuals_without_negative = current_residuals_weighted[np.where(testspec.y > current_bl)]
        positive_median = np.median(weighted_positive_residuals_without_negative)
        # positive_median = np.mean(weighted_positive_residuals_without_negative)
        m4lvl_of_negativeresiduals_good = ( np.mean(neg_residuals_without_positive**4) )**0.25
        thefuncamential_blpart = 2 * (d2bl-d2bl_current)**2 / (d2bl**2 + d2bl_current**2)
        thefuncamential_ppart = 2 * (positive_median-m4lvl_of_negativeresiduals_good)**2 / (positive_median**2+m4lvl_of_negativeresiduals_good**2)
        thefuncamential = thefuncamential_blpart + thefuncamential_ppart
        
        # # @test&debug:
        # if display > 1:
        #     print('positive_median = {:.3e}'.format(positive_median))
        #     print('positive_mean = {:.3e}'.format(np.mean(weighted_positive_residuals_without_negative)))
        #     print('m4lvl_of_negativeresiduals_good = {:.3e}'.format(m4lvl_of_negativeresiduals_good))
        
        return thefuncamential


    startingpoint_for_lambda = (len(testspec.x)/8)**4
    startingpoint_for_p = 0.1
    startingpoint = startingpoint_for_lambda, startingpoint_for_p
    print('startingpoint = {:.3e}, {:.3e}'.format(startingpoint_for_lambda, startingpoint_for_p))
    bounds_for_lambda = (len(testspec.x)/1000)**4, (len(testspec.x)/4)**4
    bounds_for_p = 1e-5, 0.49
    startingpoint = np.log(startingpoint)
    bounds_for_lambda = np.log(bounds_for_lambda)
    bounds_for_p  = np.log(bounds_for_p)
    bothbounds = Bounds([bounds_for_lambda[0], bounds_for_p[0]], [bounds_for_lambda[1], bounds_for_p[1]])        
    
    solution = minimize(func2min,
                        startingpoint,
                        method = 'Powell',
                        bounds=bothbounds,
                        options={'disp':1,
                                 'ftol':1e-4,
                                 'xtol': 0.001})
    optimal_params = solution.x
    optimal_params = np.exp(optimal_params)
    if display > 1:
        current_bl = das_baseline(testspec,
                                  als_lambda=optimal_params[0],
                                  als_p_weight=optimal_params[1],
                                  display=2)
        print('optimization converged \n lam={:.2e}, p={:.2e}'.format(optimal_params[0], optimal_params[1]))
        print('current d2bl = {:.3e}'.format((np.mean(np.diff(current_bl, 2)**2))**0.5))

    return optimal_params # lam, p



def longpass_wavelet_filter(thespectrum, cutoff='auto', polyfy=False, display=1):
    """
    cutoff is in pixels.
    auto cutoff is 1/16 of the length
    returns 4th derivative and smooth line
    """
    
    thespectrum = detrend(thespectrum)
    if cutoff=='auto':
        cutoff = int(len(thespectrum)/16)
    if display > 1:
        print('cutoff = {}'.format(cutoff))

    # compute grid for wavelet:
    scalegrid_max = np.log(len(thespectrum)-1)
    scalegrid_adv = np.linspace(0, scalegrid_max, 33)
    scalegrid_adv = np.exp(scalegrid_adv)
    scalegrid_adv = scalegrid_adv.astype(int)
    scalegrid_adv = np.unique(scalegrid_adv)
    if display > 1:
        print('wavelet grid:\n', scalegrid_adv)

    # compute wavelet transform:
    # coef, freqs=pywt.cwt(detrend(thespectrum), scalegrid_adv, 'mexh')
    coef = signal.cwt(thespectrum, signal.ricker, scalegrid_adv)

    if display >0:
        theaspect = len(thespectrum)/len(scalegrid_adv)/1.875
        #with plt.rc_context(rc={rc('font', size=SMALL_SIZE)}):
        # with plt.rc('font', size=8):
        with plt.rc_context(rc={'font.size': 7}):
            plt.matshow(coef, cmap=plt.cm.nipy_spectral_r, aspect = theaspect)
            plt.title('longpass wavelet filter', fontsize=8)
            plt.show()

    # compute reconstruction range:
    cutoff_index = np.where(scalegrid_adv>=cutoff)[0][0]
    reconstruction_range = cutoff_index,len(scalegrid_adv)

    # reconstruct it!
    icwt_mh_reconstructed = icwt_mexhat(coef[reconstruction_range[0]:reconstruction_range[1],:], scalegrid_adv[reconstruction_range[0]:reconstruction_range[1]])

    if polyfy:
        x = np.linspace(-1, 1, num=len(thespectrum))
        thesp_polycoeff = polyfit(x, icwt_mh_reconstructed, 15)
        thesp_poly = polyval(x, thesp_polycoeff)

    if display >0:
        # plt.plot(detrend(thespectrum), 'k', label='raw')
        plt.plot(thespectrum, 'k', label='raw')
        plt.plot(icwt_mh_reconstructed, 'r', linewidth=2.5, label='low-pass filtered')
        plt.title('longpass wavelet filter: estimate 4th derivative of BL', fontsize=10)
        if polyfy:
            plt.plot(thesp_poly, 'g', linewidth=1.5, label='polyfied')
        plt.legend()
        plt.show()
    if polyfy:
        F4_raw = (np.mean(np.diff(thespectrum, 4)**2))**0.5
        F4_filtered = (np.mean(np.diff(icwt_mh_reconstructed, 4)**2))**0.5
        F4_filtered_pol = (np.mean(np.diff(thesp_poly, 4)**2))**0.5
        print('rms 4th derivatives: \n raw = {:.2e}, \n wavelet-filtered = {:.2e},\n wavelet-filtered with polyfit = {:.2e}'.format(F4_raw, F4_filtered, F4_filtered_pol))
        smoothline = thesp_poly
        F4 = F4_filtered_pol
    if not polyfy:
        smoothline = icwt_mh_reconstructed
        F4 = (np.mean(np.diff(icwt_mh_reconstructed, 4)**2))**0.5

    # return 4th derivative and smooth line
    return F4, smoothline # coef, scalegrid_adv   



def icwt_mexhat(coeffs, scalegrid):
    """ inverse wavelet transform for Mexhat
    Parameters
    ----------
    coeffs : array of floats
        Result from pywt.cwt
    scalegrid : array of ints
        Grid for pywt.cwt
    Returns reconstructed spectrum
    """
    dt = 1
    # reconstruction_factor for MexHat (aka Marr) taken from
    #   Christopher Torrence and Gilbert P. Compo
    #   A Practical Guide to Wavelet Analysis / BAMS 1998
    reconstruction_factor = 1.4 * dt**0.5 / (3.451*0.867) * np.ones_like(scalegrid) / scalegrid**0.5 * 0.25
    # reconstructed_spectrum = np.trapz(coeffs, scalegrid**0.5, axis=0)
    
    reconstructed_spectrum = np.trapz(coeffs*reconstruction_factor[:, np.newaxis], axis=0)
    # reconstructed_spectrum *= reconstruction_factor
    return reconstructed_spectrum



def auto_lambda(rawspectrum, fwhm_to_pixelwidth_ratio='auto', f4th_derivative='wavelet', pixelwidth=1):
    """ rawspectrum here is a 1D vector, i.e. only y-component of the spectrum
    """
    if fwhm_to_pixelwidth_ratio=='auto':
        fwhm_to_pixelwidth_ratio = 3
    
    if f4th_derivative=='mollification':
        mwidth = int(2*(len(rawspectrum)/6)+1)
        F4 = moving_average_molification(rawspectrum, mollification_width=mwidth, number_of_molifications=16)
        F4 = (np.mean(np.diff(F4, 4)**2))**0.5
        print('auto 4th derivative of baseline from 16x mollification = {:.3e}'.format(F4))
        f4th_derivative = F4
    elif f4th_derivative=='wavelet':
        F4, _ = longpass_wavelet_filter(rawspectrum, polyfy=True)
        f4th_derivative = F4
        print('4th derivative estimated from wavelet reconstruction')
        print(' 4th derivative = {:.2e}'.format(F4))
    else:
        print(' 4th derivative of baseline from input = {:.3e}', f4th_derivative)

    # 0: find sigma_rm
    sigma0 = morphological_noise(rawspectrum)
    alpha = np.pi * fwhm_to_pixelwidth_ratio
    sigma_rm = sigma0 * (
        np.pi**2 * (2*alpha + np.sinh(2*alpha))/
        (alpha * (np.sinh(alpha))**2) )**0.5
    #@Test&Debug:
    print(' sigma0 = {:.3e}'.format(sigma0),
          '\n sigma_rm = {:.3e}'.format(sigma_rm))

    # 1: find parameter 'a'
    #   rms 4th derivative:

    parameter_a = (2 * sigma_rm / np.pi) / (np.mean(f4th_derivative**2))**0.5
    
    # 2: lambda
    auto_lambda = (np.pi**0.5 * parameter_a / 2)**(8/9)
    
    #@Test&Debug:
    print(' auto_lambda = {:.3e}'.format(auto_lambda))
    
    return auto_lambda



def find_da_peaks(derspec, signal2noise=5, display=0):
    """ 
        This function finds peaks using find_peaks from scipy.signal
        Input:
            the ExpSpec object
        signal2noise
            is the threshold above which the peak detection looks for peaks
            Default: 5 (Rose criterion)
        It is assumed that derxpec.x is ordered ascending"""
    
    # peak detection
    
    # adding edge elements to make it possible to detect peaks at the edges:
    y_4detection = deepcopy(derspec.y-moving_average_molification(derspec.y, number_of_molifications=16))
    y_4detection = np.append(y_4detection, 0)
    y_4detection = np.insert(y_4detection, 0, 0)
    
    morph_noise_level = morphological_noise(derspec.y)
    peak_indexes, _= find_peaks(y_4detection, prominence=signal2noise*morph_noise_level)
    
    # now making adjustment for the extra edge elements
    peak_indexes -= 1
    
    if display > 0:
        plt.plot(derspec.x, derspec.y, 'k')
        plt.plot(derspec.x[peak_indexes], derspec.y[peak_indexes], 'o', mfc='none', ms = 8, mec='r')
        plt.title('peak detection by morphological algorithm', fontsize=10)
        plt.show()
    
    return derspec.x[peak_indexes]


def find_startingpoint_for_fit_with_baseline(derspec, fitrange=None):
    """ this function detects peaks in the spectrum
          and writes the detected peak parameters
          to the file named 'auto_generated_startingpoint.txt'
          """
    spectrum_at_detection = ExpSpec(derspec.x,
                                    derspec.y-moving_average_molification(derspec.y, number_of_molifications=16))
    interpoint_distance = (derspec.x[-1]-derspec.x[0]) / (len(derspec.x)-1)
    
    # detect peaks and find parameters
    dermultipeak = multipeak_fit(spectrum_at_detection, fitrange)
    
    startingpoint = np.full((dermultipeak.number_of_peaks, 15), np.nan)
    
    # 1: positions
    #       since the peak positions are well defined,
    #       they will be constrained to ~3 interpoint distances on each side
    startingpoint[:,0] = dermultipeak.specs_array[:,0]
    startingpoint[:,1] = dermultipeak.specs_array[:,0] - 3*interpoint_distance
    startingpoint[:,2] = dermultipeak.specs_array[:,0] + 3*interpoint_distance
    
    # 2: FWHM
    #       fwhm from detection should be under-estimated, so the range will be
    #       [0.8*fwhm0, 2*fwhm0]
    startingpoint[:,3] = dermultipeak.specs_array[:,1]
    startingpoint[:,4] = dermultipeak.specs_array[:,1] * 0.8
    startingpoint[:,5] = dermultipeak.specs_array[:,1] * 2.0
    
    # asym will be not constrained
    startingpoint[:,6] = dermultipeak.specs_array[:,2]
    
    # Gauss contributions will be not constrained
    startingpoint[:,9] = 0.5
    
    # Amplitudes will be not constrained
    startingpoint[:,12] = dermultipeak.specs_array[:,4]

    write_startingpoint_to_file(startingpoint)
    
    return startingpoint
        
