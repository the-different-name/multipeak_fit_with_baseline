#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created

научный рабочий
"""

import numpy as np
from copy import deepcopy
from scipy.optimize import least_squares, minimize, Bounds, fmin_l_bfgs_b
from scipy.signal import find_peaks
from datetime import datetime
import colorsys
from itertools import cycle
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pathlib import Path
from scipy import signal
from numpy.polynomial.polynomial import polyfit, polyval

import warnings
warnings.simplefilter('ignore',sparse.SparseEfficiencyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [6.0, 3.2]
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from expspec import *
from spectralfeature import voigt_asym, MultiPeak, CalcPeak


def read_startingpoint_from_txt(filename):
    """ ???
    """
    
    number_of_peaks = 0
    empty_row = np.full((1, 15), np.nan)
    startingpoint = np.empty((0, 15))

    with open(filename) as input_data:
        
        # locate the beginning of the startingpoint:
        while True:
            line = input_data.readline()
            if line.lower().strip().startswith('#') == True:
                continue
            else:
                break

        number_of_peaks = 0
        
        while True:
            line = ' '.join(line.split())
            line = line.replace(',', '')
            line = line.replace(';', '')
            line = line.replace('( ', '(')
            line = line.replace(' )', ')')
            currentline = line.split()
            if line.strip() == '':
                break
            if line.lower().strip().startswith('#') == True:
                line = input_data.readline()
                continue
            else:
                number_of_peaks+= 1
                startingpoint = np.append(startingpoint, empty_row, axis=0)

                string_number = 0
                for i in range(5):
                    try:
                        tmpv = currentline[string_number]
                        tmpv = np.asarray(tmpv, dtype=float)
                        startingpoint[number_of_peaks-1, 3*i] = deepcopy(tmpv)
                        string_number += 1
                    except IndexError:
                        break

                    try:
                        if currentline[string_number].startswith('('):
                            constraint_L = currentline[string_number]
                            constraint_L = constraint_L.replace('(', '')
                            constraint_L = np.asarray(constraint_L, dtype=float)
                            string_number += 1
                            
                            constraint_U = currentline[string_number]
                            constraint_U = constraint_U.replace(')', '')
                            constraint_U = np.asarray(constraint_U, dtype=float)
                            startingpoint[number_of_peaks-1, 3*i+1] = deepcopy(constraint_L)
                            startingpoint[number_of_peaks-1, 3*i+2] = deepcopy(constraint_U)
                            string_number += 1

                    except IndexError:
                        break

            line = input_data.readline()
            if not line:
                # print('reading finished')
                break

    # check if LB <= UB
    for i in range(number_of_peaks):
        for parameter_number in range(5):
            if startingpoint[i, parameter_number*3+1] > startingpoint[i, parameter_number*3+2]:
                print('Error! \n LB > UB')
                print (' Peak number ', i+1, ', \N{greek small letter omega} = ', startingpoint[i, 0], ' cm\N{superscript minus}\N{superscript one}', sep = '')
                raise ValueError
                return

    # check if Starting value of a parameter is within the [LB, UB] interval
    for i in range(number_of_peaks):
        for parameter_number in range(5):
            if startingpoint[i, parameter_number*3] < startingpoint[i, parameter_number*3+1] or startingpoint[i, parameter_number*3+2] < startingpoint[i, parameter_number*3]:
                print('Error! \n Starting value of a parameter is not within the [LB, UB] interval')
                print (' Peak number ', i+1, ', \N{greek small letter omega} = ', startingpoint[i, 0], ' cm\N{superscript minus}\N{superscript one}', sep = '')
                raise ValueError
                return


    # print('number of peaks = {}'.format(number_of_peaks))
    print(startingpoint)
    return startingpoint




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


def _get_colors(num_colors):
    """
    this is supplementary function to use while plotting.
    It makes a list of [significantly] different colors.
    """
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def multipeak_fit(derspec,
                  fitrange=None,
                  display=1,
                  startingpoint='auto_generate',
                  filenameprefix = 'multipeak_fit',
                  saveresults = False,
                  x_axis_title = 'wavenumber / cm$^{-1}$',
                  y_axis_title = 'intensity',
                  supress_negative_peaks = False,
                  labels_on_plot = True,
                  plot_residuals = False,
                  exp4sign=0.0):
    """ 
    Multipeak fit with no baseline.
    Returns class multipeak
    Input:
     derspec: class expspec
     staringpoint:
         either numpy array,
         or text file,
         or let it auto generate
     fitrange: (x1, x2) is the fitting range
     Display:    0 for nothing,
                 1 to save the figure of fit results + final print-out
                 2 for test-and-debug
    exp4sign - parameter for the asymmetry of (y-yfit):
                 default is 0, but you may try np.exp(-1) as a good value
     """

    # capture the original working range, which has to be restored later:
    original_working_range = derspec.working_range
    
    # step 1: check if we need to set fitting range
    if fitrange != None: 
        derspec.working_range = fitrange
        #@Test&Debug # 
        if display > 1:
            print('fitting range from input: ', derspec.working_range)

    # step 2: load startingpoint
    if startingpoint=='auto_generate':
        # generate from find_peaks !!
        peak_positions = find_da_peaks(derspec, display=display)
        startingpoint = np.full((np.shape(peak_positions)[0], 15), np.nan)
        startingpoint[:,0] = peak_positions
    elif type(startingpoint)==str:
        print('startingpoint file = {}, reading from file'.format(startingpoint))
        startingpoint = read_startingpoint_from_txt(startingpoint)
        print('(nan okay here)')
    else:
        print('startingpoint from input')


    number_of_peaks = np.shape(startingpoint)[0]
    bounds_high = startingpoint[:, 2::3]
    bounds_low = startingpoint[:, 1::3]
    startingpoint = startingpoint[:, 0::3]
    interpoint_distance = (derspec.x[-1]-derspec.x[0]) / (len(derspec.x)-1)

    if display > 1:
            print('number_of_peaks: ', number_of_peaks)

    # 1(b): fwhm startingpoint and bounds
    # if bounds not set,
    #   default for fwhm is from interpoint_distance to 1/3 of the spectral range
    #   default for asym parameter is [-0.36 -:- 0.36]
    #   default for Gaussian share is [0 -:- 1]
    #   default for amplitudes is [0 -:- inf]
    #   amplitude bounds are not used in the script !
    j = 1
    bounds_low[np.isnan(bounds_low[:,j]), j] = abs(interpoint_distance)
    bounds_high[np.isnan(bounds_high[:,j]), j] = abs((derspec.x[-1]-derspec.x[0]))/3 # 1/3 of the spectral range
    startingpoint[np.isnan(startingpoint[:,j]), j] = bounds_low[np.isnan(startingpoint[:,j]), j]*0.75 + bounds_high[np.isnan(startingpoint[:,j]), j]*0.25
    # 2(c): asym
    j = 2
    bounds_low[np.isnan(bounds_low[:,j]), j] = -0.36
    bounds_high[np.isnan(bounds_high[:,j]), j] = 0.36
    startingpoint[np.isnan(startingpoint[:,j]), j] = 0.5*(bounds_low[np.isnan(startingpoint[:,j]), j] + bounds_high[np.isnan(startingpoint[:,j]), j])
    # 3(d): Gaussian share
    j = 3
    bounds_low[np.isnan(bounds_low[:,j]), j] = 0
    bounds_high[np.isnan(bounds_high[:,j]), j] = 1
    startingpoint[np.isnan(startingpoint[:,j]), j] = 0.5*(bounds_low[np.isnan(startingpoint[:,j]), j] + bounds_high[np.isnan(startingpoint[:,j]), j])
    # 4(e): Voigt amplitudes
    j = 4
    bounds_low[:, j] = 0
    bounds_high[:, j] = np.inf
    startingpoint[:, j] = 1
    # 0(a): peak positions.
    #   if bounds_high are absent, set to position + bounds_high of fwhm
    j = 0
    bounds_low[np.isnan(bounds_low[:,j]), j] = startingpoint[np.isnan(bounds_low[:,j]), j] - bounds_high[np.isnan(bounds_low[:,j]), 1]
    bounds_high[np.isnan(bounds_high[:,j]), j] = startingpoint[np.isnan(bounds_high[:,j]), j] + bounds_high[np.isnan(bounds_high[:,j]), 1]

    if display > 1:
        print('startingpoint loaded. \n Startingpoint: \n', startingpoint,
              '\n bounds_low: \n', bounds_low,
              '\n bounds_high: \n', bounds_high,)

    # check if LB <= UB
    for i in range(number_of_peaks):
        for parameter_number in range(5):
            if bounds_low[i, parameter_number] > bounds_high[i, parameter_number]:
                print('Error! \n LB > UB')
                print (' Peak number ', i+1, ', \N{greek small letter omega} = ', startingpoint[i, 0], ' cm\N{superscript minus}\N{superscript one}', sep = '')
                raise ValueError
                return

    # check if Starting value of a parameter is within the [LB, UB] interval
    for i in range(number_of_peaks):
        for parameter_number in range(5):
            if startingpoint[i, parameter_number] < bounds_low[i, parameter_number] or startingpoint[i, parameter_number] > bounds_high[i, parameter_number]:
                print('Error! \n Starting value of a parameter is not within the [LB, UB] interval')
                print (' Peak number ', i+1, ', \N{greek small letter omega} = ', startingpoint[i, 0], ' cm\N{superscript minus}\N{superscript one}', sep = '')
                raise ValueError
                return
    if display > 1:
        print('startingpoint check okay')


    # step 3: initialize the multipeak (number_of_peaks)
    dermultipeak = MultiPeak(derspec.x, number_of_peaks)
    dermultipeak.specs_array = startingpoint
    #@Test&Debug # 
    # print('dermultipeak.peak_height before setting = ', dermultipeak.peak_height)

    # step 4: Set starting values for all peak_heights
    starting_peakindexes = np.zeros_like(dermultipeak.specs_array[:, 0])
    for i in range(number_of_peaks):
        starting_peakindexes[i] = (np.abs(derspec.x-dermultipeak.specs_array[i, 0])).argmin() # find point number for a closest to x0 point:

        #@Test&Debug # 
        # print('x0 of peak ', i, ' at ', derspec.x[int(starting_peakindexes[i])], ' is point number ', starting_peakindexes[i])
        # print('starting_peakindexes', starting_peakindexes)
    peakheights = derspec.y[starting_peakindexes.astype(int)]
    peakheights = peakheights * (peakheights >= 0) + 0 * (peakheights < 0)
    dermultipeak.peak_height  = peakheights
    #@Test&Debug # 
    # if display > 1:
    #     print('dermultipeak.peak_height after setting = ', dermultipeak.peak_height)
    
    if display > 1:
        plt.plot(derspec.x, derspec.y, 'k',
                 dermultipeak.wn, dermultipeak.curve, 'b');
        plot_annotation = 'starting point'
        plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.show()


    def func2min (peakparams) :
        """ Asymmetric pseudo-Voigt funtion as defined in Analyst: 10.1039/C8AN00710A
        """
        curve = np.zeros((len(dermultipeak.wn), number_of_peaks))
        for i in range(number_of_peaks):
            curve[:,i] = peakparams[i*5+4] * voigt_asym(dermultipeak.wn-peakparams[i*5+0], peakparams[i*5+1], peakparams[i*5+2], peakparams[i*5+3])
        thediff = derspec.y - np.sum(curve, axis=1)
        derfunc = (thediff * np.exp(exp4sign*(1 - np.sign(thediff))**2))
        return derfunc



    while True :
        # convert startingpoint and bounds to vectors:
        startingpoint1D = np.reshape(startingpoint, 5*number_of_peaks)
        bounds_low1D = np.reshape(bounds_low, 5*number_of_peaks)
        bounds_high1D = np.reshape(bounds_high, 5*number_of_peaks)
        try:
            solution = least_squares(func2min, startingpoint1D, bounds=[bounds_low1D, bounds_high1D], verbose=0) # , max_nfev=1e4)
            converged_parameters = np.reshape(solution.x, (number_of_peaks, 5))
            dermultipeak.specs_array = converged_parameters
            #     #@Test&Debug #
            if display > 1:
                print('least_squares converged')
            break
        except RuntimeError: # (RuntimeError, TypeError, NameError):
            #     #@Test&Debug #
            if display > 1:
                print('least_squares optimization error, expanding the fitting range')
            converged_parameters = np.reshape(solution.x, (number_of_peaks, 5))
            dermultipeak.specs_array = converged_parameters
            continue

    # #@Test&Debug 
    # tmpv = input("""press 'k Enter' to stop or 'Enter' to continue """)
    # if tmpv == 'k':
    #     print('''okay, let's stop here''')
    #     break
    

    if saveresults:
        Path("fit_results/txt").mkdir(parents=True, exist_ok=True)
        Path("fit_results/pic").mkdir(parents=True, exist_ok=True)

    if display > 0 or saveresults:
        if display > 0:
            color_list = _get_colors(1+number_of_peaks)
            cycol = cycle(color_list)
            plt.plot(derspec.x, derspec.y, 'o', mfc='none', ms = 4, mec=(0.1, 0.1, 0.1, 0.5),
                     label='raw data')
            plt.plot(dermultipeak.wn, dermultipeak.baseline + dermultipeak.curve,
                     color=next(cycol), label='fit envelope', linewidth=1);
            for current_peak in range(number_of_peaks):
                plt.plot(dermultipeak.wn, dermultipeak.baseline + dermultipeak.multicurve[:,current_peak],
                         ':', color=next(cycol),
                         label='peak at {:.1f}'.format(dermultipeak.specs_array[current_peak,0]),
                         linewidth=1.5)
            # plt.tick_params(labelleft=False)
            # plt.grid(False)
            # plt.title('lam = {:.2e}'.format(the_lambda), fontsize=10)
            plt.xlabel(x_axis_title)
            plt.ylabel(y_axis_title)
            if labels_on_plot:
                plt.legend()
            if saveresults:
               now = datetime.now()
               filename = filenameprefix + '_multipeak_conv_' + now.strftime("%H%M%S") + '.png'
               plt.savefig('fit_results/pic/' + filename, bbox_inches='tight', transparent=True)
            plt.show()

            if plot_residuals:        
                plt.plot(derspec.x, derspec.y-dermultipeak.curve,
                         'o', mfc='none', ms = 4, mec=(0.1, 0.1, 0.1, 0.5),
                         label='fit residuals')
                plt.legend()
                plt.show()

        if saveresults:
        # save peak parameters
            dermultipeak.save_specs_array_to_txt(filename='fit_results/txt/'+filenameprefix + '_peakparams_' + now.strftime("%H%M%S") + '.txt')
        # save decomposition to txt
            dermultipeak.write_decomposition_to_txt(exp_y=derspec.y,
                                                    filename='fit_results/txt/' + filenameprefix + '_decomp_' + now.strftime("%H%M%S") + '.txt')

        
        # print results
        results_header = ' '
        col1 = 'position'
        col2 = 'fwhm'
        col3 = 'asym'
        col4 = 'gauss'
        col5 = 'area'
        results_header += f'{col1:>12}' + f'{col2:>12}' + f'{col3:>12}' + f'{col4:>12}' + f'{col5:>12}'
        print('\n      FIT RESULTS:\n')
        print(results_header)
        with np.printoptions(formatter={'float': '{: 11.3f}'.format}):
            print(dermultipeak.specs_array)

    # restoring the working range of the ExpSpec class:
    derspec.working_range = original_working_range
    return dermultipeak


def write_startingpoint_to_file(startingpoint, filename='auto_generated_startingpoint.txt'):
    with open(filename, 'w', encoding='utf-8') as the_file:
        startingpoint_header = ' # auto-generated startingpoint \n'
        startingpoint_header +=  '# nan (not-a-number) are okay here \n'
        startingpoint_header += '# positions' + 23*' '
        startingpoint_header += ' fwhms' + 27*' '
        startingpoint_header += ' asyms' + 27*' '
        startingpoint_header += 'Gauss_shares'  + 18*' '
        startingpoint_header += 'amplitudes' + 20*' '
        startingpoint_header += '\n'
        the_file.write(startingpoint_header)
        for i in range(np.shape(startingpoint)[0]):
            current_line  = ''
            for j in range(5):
                current_line +=  '{:8.2f}  '.format(startingpoint[i, 3*j])
                current_line += '({:8.2f}, '.format(startingpoint[i, 3*j+1])
                current_line += '{:8.2f})  '.format(startingpoint[i, 3*j+2])
            the_file.write(current_line + '\n')



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
        
    


def multipeak_fit_with_BL(derspec,
                          fitrange=None,
                          display=1,
                          startingpoint='auto_generate',
                          the_lambda = None,
                          saveresults = False,
                          filenameprefix = 'multipeak_fit',
                          x_axis_title = 'wavenumber / cm$^{-1}$',
                          y_axis_title = 'intensity',
                          supress_negative_peaks = False,
                          labels_on_plot = True,
                          apply_corrections = False):
    """ 
     fitrange: (x1, x2) is the fitting range
     Display:    0 for nothing,
                 1 to save the figure of fit results + final print-out
                 2 for test-and-debug
     Returns:
                derspec: class multipeak
     """

    # capture the original working range, which has to be restored later:
    original_working_range = derspec.working_range

    # step 0: load startingpoint and check it
    if startingpoint=='auto_generate':
        # generate from find_peaks !!
        peak_positions = find_da_peaks(derspec, display=display)
        startingpoint = np.full((np.shape(peak_positions)[0], 15), np.nan)
        startingpoint[:,0] = peak_positions
    elif type(startingpoint)==str:
        if display > 0:
            print('startingpoint file = {}, reading from file'.format(startingpoint))
        startingpoint = read_startingpoint_from_txt(startingpoint)
        print('(nan okay here)')
    else:
        if display > 0:
            print('startingpoint from input array')

    # @Test&Debug: 
        # optional: write current starting point to file
    write_startingpoint_to_file(startingpoint, filename='current_startingpoint.txt')

    number_of_peaks = np.shape(startingpoint)[0]
    bounds_high = startingpoint[:, 2::3]
    bounds_low = startingpoint[:, 1::3]
    startingpoint = startingpoint[:, 0::3]
    # discard amplitude from the starting point:
    startingpoint = startingpoint[:,:4]
    bounds_high = bounds_high[:,:4]
    bounds_low = bounds_low[:,:4]
    
    interpoint_distance = (derspec.x[-1]-derspec.x[0]) / (len(derspec.x)-1)

    if display > 1:
            print('number_of_peaks: ', number_of_peaks)
            
    # 1(b): fwhm startingpoint and bounds
    j = 1
    bounds_low[np.isnan(bounds_low[:,j]), j] = abs(interpoint_distance)
    bounds_high[np.isnan(bounds_high[:,j]), j] = abs((derspec.x[-1]-derspec.x[0]))/3 # 1/3 of the spectral range
    startingpoint[np.isnan(startingpoint[:,j]), j] = bounds_low[np.isnan(startingpoint[:,j]), j]*0.75 + bounds_high[np.isnan(startingpoint[:,j]), j]*0.25
    # 2(c): asym
    j = 2
    bounds_low[np.isnan(bounds_low[:,j]), j] = -3.6e-1
    bounds_high[np.isnan(bounds_high[:,j]), j] = 3.6e-1
    startingpoint[np.isnan(startingpoint[:,j]), j] = 0.5*(bounds_low[np.isnan(startingpoint[:,j]), j] + bounds_high[np.isnan(startingpoint[:,j]), j])
    # 3(d): Gaussian share
    j = 3
    bounds_low[np.isnan(bounds_low[:,j]), j] = 0
    bounds_high[np.isnan(bounds_high[:,j]), j] =  1
    startingpoint[np.isnan(startingpoint[:,j]), j] = 0.5*(bounds_low[np.isnan(startingpoint[:,j]), j] + bounds_high[np.isnan(startingpoint[:,j]), j])

    # 0(a): peak positions.
    #   if bounds_high are absent, set to [position + bounds_high of fwhm]
    j = 0
    bounds_low[np.isnan(bounds_low[:,j]), j] = startingpoint[np.isnan(bounds_low[:,j]), j] - bounds_high[np.isnan(bounds_low[:,j]), 1]
    bounds_high[np.isnan(bounds_high[:,j]), j] = startingpoint[np.isnan(bounds_high[:,j]), j] + bounds_high[np.isnan(bounds_high[:,j]), 1]

    if display > 1:
        print('\n Startingpoint: \n', startingpoint,
              '\n bounds_low: \n', bounds_low,
              '\n bounds_high: \n', bounds_high,)

    # check if LB <= UB
    for i in range(number_of_peaks):
        for parameter_number in range(4):
            if bounds_low[i, parameter_number] > bounds_high[i, parameter_number]:
                print (' Peak number ', i+1, ', \N{greek small letter omega} = ', startingpoint[i, 0], ' cm\N{superscript minus}\N{superscript one}', sep = '')
                raise SystemExit('Error! \n LB > UB')
                return

    # check if Starting value of a parameter is within the [LB, UB] interval
    for i in range(number_of_peaks):
        for parameter_number in range(4):
            if startingpoint[i, parameter_number] < bounds_low[i, parameter_number] or startingpoint[i, parameter_number] > bounds_high[i, parameter_number]:
                print (' Peak number ', i+1, ', \N{greek small letter omega} = ', startingpoint[i, 0], ' cm\N{superscript minus}\N{superscript one}', sep = '')
                raise SystemExit('Error! \n Starting value of a parameter is not within the [LB, UB] interval')
                return
    if display > 1:
        print('startingpoint check okay')


    # step 1: check if we need to set fitting range
    if fitrange != None: 
        derspec.working_range = fitrange
        #@Test&Debug # 
        if display > 1:
            print('fitting range from input: ', derspec.working_range)

    # step 2: compute lambda
    if not the_lambda:
        the_lambda = auto_lambda(derspec.y, fwhm_to_pixelwidth_ratio='auto', pixelwidth=interpoint_distance)

    if display > 0:
        print('in this run set the_lambda to {:.3e}'.format(the_lambda) )

    # step 3: initialize the multipeak (number_of_peaks)
    dermultipeak = MultiPeak(derspec.x, number_of_peaks)
    dermultipeak.specs_array[:,0:4] = startingpoint
    dermultipeak.specs_array[:,4] = 1
    dermultipeak.lam = the_lambda

    # current_multipeak would be updated at each iteration:
    current_multipeak = MultiPeak(derspec.x, number_of_peaks)
    current_multipeak.lam = the_lambda
    
    # step 4: initialize the sparse matrix:
    # these matrixes gatta be set outside the fitting function:    
    L = len(derspec.y)
    D2 = sparse.diags([1,-2,1],[-1,0,1], shape=(L,L))
    D4 = D2.dot(D2.transpose())
    D = deepcopy(D4) * the_lambda
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    D += W
    D = sparse.hstack([D, np.zeros((L, number_of_peaks+2))])
    D = sparse.vstack([D, np.zeros((number_of_peaks+2, L+number_of_peaks+2))])
    D = sparse.csr_matrix(D)
    b_vector = np.zeros(L+number_of_peaks+2)
    b_vector[0:L] = derspec.y
    
    def find_baseline_and_amplitudes(peakparams, display=2):
        """ This function calculates the norm of the deviation
            And updates the baseline and amplitudes to match the given peak parameters.
            Achtung! Peak parameters here are given as a vector, not array!
                (This vector is 1D, because the minimization takes a 1D vector)
            """        
        # calculate peak shapes:
        current_multipeak.specs_array[:,0:4] = np.reshape(peakparams, (number_of_peaks, 4))
        current_multipeak.specs_array[:,4] = 1 # set amplitudes to 1 to find them later !
        peak_shapes = current_multipeak.multicurve
        
        # add linear offset here:
        current_multipeak.linear_baseline_scalarpart = np.ones_like(derspec.y) / len(derspec.y)
        current_multipeak.linear_baseline_slopepart = np.linspace(-1, 1, len(derspec.y)) / len(derspec.y)
        peak_shapes = np.column_stack((peak_shapes , current_multipeak.linear_baseline_scalarpart))
        peak_shapes = np.column_stack((peak_shapes , current_multipeak.linear_baseline_slopepart))
        
        # now set the values for the last rows and columns
        for current_peak in np.arange(number_of_peaks+2):
            # construct it for off-diagonal also:
            # off-diagonal terms should be like
                # L1*L2 (scalar multiplication)
            for current_peak_2 in np.arange(number_of_peaks+2):
                D[L+current_peak, L+current_peak_2] = np.sum(peak_shapes[:, current_peak] *
                                                             peak_shapes[:, current_peak_2])
            # achtung: these lines sometimes don't work:
                # D[0:L, L+current_peak] = peak_shapes[:, current_peak]
                # D[L+current_peak, 0:L] = peak_shapes[:, current_peak]
            #  so I added 'reshape'
            D[0:L, L+current_peak] = np.reshape(peak_shapes[:, current_peak], (L, 1))
            D[L+current_peak, 0:L] = np.reshape(peak_shapes[:, current_peak], (1, L))
            
            b_vector[L+current_peak] = np.sum((derspec.y)*peak_shapes[:,current_peak])
    
        full_solution = spsolve(D, b_vector)
        current_multipeak.d2baseline = full_solution[0:L]
        the_amplitudes = full_solution[L:-2] # **2 # * peak_fwhms
        current_multipeak.linear_baseline_scalarpart *= full_solution[-2]
        current_multipeak.linear_baseline_slopepart *= full_solution[-1]

        #@Test&Debug # we can plot at each iteration:        
        # if display>1:
        #     color_list = _get_colors(1+number_of_peaks)
        #     cycol = cycle(color_list)
    
        #     plt.plot(derspec.x, derspec.y, 'o', mfc='none', ms = 4, mec=(0.1, 0.1, 0.1, 0.5),
        #               label='raw data')
        #     plt.plot(derspec.x, current_multipeak.baseline,
        #              color=next(cycol), label='baseline')
        #     for current_peak in np.arange(number_of_peaks):
        #         plt.plot(derspec.x, current_multipeak.baseline + current_multipeak.multicurve[:,current_peak],
        #                   ':', color=next(cycol), label='BL+peak '+str(current_peak))
        #     plt.legend()
        #     plt.show()
        
        current_multipeak.specs_array[:,4] = the_amplitudes

        norm_of_deviation = np.sum( (np.matmul(D2.toarray(), current_multipeak.d2baseline))**2 * the_lambda +
                                    (derspec.y - 
                                    current_multipeak.d2baseline -
                                    current_multipeak.linear_baseline -
                                    current_multipeak.curve)**2 )

        if supress_negative_peaks:
            if np.min(the_amplitudes) < 0:
                # print('suppressing negative peaks: np.min(the_amplitudes) =', np.min(the_amplitudes))
                norm_of_deviation += 1
                norm_of_deviation *= 1e4

        return norm_of_deviation


    while True :
        # convert startingpoint and bounds to vectors:
        startingpoint1D = np.reshape(startingpoint, 4*number_of_peaks)
        bounds_low1D = np.reshape(bounds_low, 4*number_of_peaks)
        bounds_high1D = np.reshape(bounds_high, 4*number_of_peaks)
        
        try:
            solution = minimize(find_baseline_and_amplitudes,
                        startingpoint1D,
                        method = 'L-BFGS-B', # 'Nelder-Mead'
                        bounds=Bounds(bounds_low1D, bounds_high1D),
                        options={'maxcor':8,
                                 'ftol':1e-7,
                                 'gtol': 4e-05,
                                 'eps': 1e-07,
                                 'maxiter':128,
                                 'maxfun':4096,
                                 # 'iprint':8,
                                 'maxls':16}
                        # options={'disp':2}
                        )

            # solution = fmin_l_bfgs_b(find_baseline_and_amplitudes,
            #                     startingpoint1D,
            #                     approx_grad=True,
            #                     bounds=(bounds_low1D, bounds_high1D),
            #                     m=8,
            #                     factr=1e7,
            #                     pgtol=4e-05,
            #                     epsilon=1e-07,
            #                     iprint=1,
            #                     maxfun=1e3,
            #                     maxiter=2e3,
            #                     maxls=16)
            # final evaluation of 'find_baseline_and_amplitudes'
            #   to make sure that the optimized parameters are written to the results:
            optimized_norm_of_deviation = find_baseline_and_amplitudes(solution.x)
            if display > 1:
                print('optimized_norm_of_deviation = ', optimized_norm_of_deviation)
            
            converged_linear_offset = current_multipeak.linear_baseline
            converged_parameters = np.reshape(solution.x, (number_of_peaks, 4))
            dermultipeak = current_multipeak
            #     #@Test&Debug #
            if display > 1:
                print('least_squares converged')
            break
        except RuntimeError: # (RuntimeError, TypeError, NameError):
            #     #@Test&Debug #
            if display > 0:
                print('least_squares optimization error, expanding the fitting range')
            converged_parameters = np.reshape(solution.x, (number_of_peaks, 4))
            dermultipeak.specs_array[:,0:4] = converged_parameters
            continue
        
    if saveresults:
        Path("fit_results/txt").mkdir(parents=True, exist_ok=True)
        Path("fit_results/pic").mkdir(parents=True, exist_ok=True)
        
    if apply_corrections:
        params_corrections_array = dermultipeak.construct_corrections()
        uncorrected_specs_array = deepcopy(dermultipeak.specs_array)
        dermultipeak.specs_array += params_corrections_array
        baseline_before_corrections = deepcopy(dermultipeak.baseline)
        D2 = sparse.diags([1,-2,1],[-1,0,1], shape=(L,L))
        D4 = D2.dot(D2.transpose())
        D_short = deepcopy(D4) * the_lambda
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        D_short += W
        D_short = sparse.csr_matrix(D_short)
        b_vector_short = derspec.y - dermultipeak.curve - dermultipeak.linear_baseline

        corrected_baseline = spsolve(D_short, b_vector_short)
        dermultipeak.d2baseline = corrected_baseline
        if display > 0:
            print("""applying corrections, please check if that's what you want""")

    if display > 0 or saveresults:
        if display > 0:
            color_list = _get_colors(1+number_of_peaks)
            cycol = cycle(color_list)
            plt.plot(derspec.x, derspec.y, 'o', mfc='none', ms = 4, mec=(0.1, 0.1, 0.1, 0.5),
                     label='raw data')
            plt.plot(dermultipeak.wn, dermultipeak.d2baseline + dermultipeak.curve + dermultipeak.linear_baseline,
                     color=next(cycol), label='fit envelope', linewidth=2);
            for current_peak in range(number_of_peaks):
                plt.plot(dermultipeak.wn, dermultipeak.d2baseline + dermultipeak.linear_baseline + dermultipeak.multicurve[:,current_peak],
                         ':', color=next(cycol),
                         label='peak at {:.1f}'.format(dermultipeak.specs_array[current_peak,0]),
                         linewidth=1)
            if apply_corrections:
                plt.plot(dermultipeak.wn, baseline_before_corrections,
                         ':', color='k',
                         label='BL before corrections',
                         linewidth=1)
            # plt.tick_params(labelleft=False)
            # plt.grid(False)
            plt.title('multipeak fit,  ' + r'$\lambda$' + ' = {:.2e}'.format(the_lambda), fontsize=10)
            plt.xlabel(x_axis_title)
            plt.ylabel(y_axis_title)
            if labels_on_plot:
                plt.legend()
                
            if saveresults:
                now = datetime.now()
                filename = 'fit_results/pic/' + filenameprefix + '_multipeak_conv_' + now.strftime("%H%M%S") + '.png'
                plt.savefig(filename, bbox_inches='tight', transparent=True)
            plt.show()

        # # plot resuduals        
        # plt.plot(derspec.x, derspec.y-current_multipeak.baseline-dermultipeak.curve,
        #          'o', mfc='none', ms = 4, mec=(0.1, 0.1, 0.1, 0.5),
        #          label='fit residuals')
        # plt.legend()
        # plt.show()

        # print results
        results_header = ' '
        col1 = 'position'
        col2 = 'fwhm'
        col3 = 'asym'
        col4 = 'gauss'
        col5 = 'area'
        results_header += f'{col1:>12}' + f'{col2:>12}' + f'{col3:>12}' + f'{col4:>12}' + f'{col5:>12}'
        print('\n      FIT RESULTS:\n')
        print(results_header)
        with np.printoptions(formatter={'float': '{: 11.3f}'.format}):
            print(dermultipeak.specs_array)
        if apply_corrections:
            print('before corrections:')
            print(results_header)
            with np.printoptions(formatter={'float': '{: 11.3f}'.format}):
                print(uncorrected_specs_array)
        if saveresults:
            # save peak parameters to file
            dermultipeak.save_specs_array_to_txt(filename='fit_results/txt/'+filenameprefix + '_peakparams_' + now.strftime("%H%M%S") + '.txt')

            # save decomposition to txt
            dermultipeak.write_decomposition_to_txt(exp_y=derspec.y,
                                                    filename='fit_results/txt/' + filenameprefix + '_decomp_' + now.strftime("%H%M%S") + '.txt')

    # restoring the working range of the ExpSpec class:
    derspec.working_range = original_working_range
    return dermultipeak



if __name__ == '__main__':

    # s = read_startingpoint_from_txt('test_spectrum_startingpoint.txt')
    # current_spectrum = np.genfromtxt('test_data_experimental_spectrum.txt') # read file to numpy format
    # testspec = ExpSpec(current_spectrum[:,0], current_spectrum[:,1]) # convert the spectrum to an *object* of a specific format.        
    
    # dat_result = multipeak_fit_with_BL(testspec,
    #                           fitrange=(500, 3700),
    #                           startingpoint='test_spectrum_startingpoint.txt',
    #                           the_lambda = 1e8, # 6e7
    #                           saveresults=True,
    #                           display=2,
    #                           apply_corrections=False,)






    # 1) generate test spectrum:
    print('''let's generate a test spectrum with two Lor functions, sine-like baseline and random noise''')
    number_of_points = 1025
    wavenumber = np.linspace(0, 1024, num=number_of_points)
    Lorentz_positions = (384, 720)
    Lorentz_FWHMs = (32, 64)
    amplitudes0 = (128*2, 128*8)
    synthetic_bl = 2*np.sin(np.pi * wavenumber/256)
    random_noise = 2*np.random.uniform(-1, 1, len(wavenumber))
    lor_func_0 = amplitudes0[0] * voigt_asym(wavenumber-Lorentz_positions[0], Lorentz_FWHMs[0], 0, 0)
    lor_func_1 = amplitudes0[1] * voigt_asym(wavenumber-Lorentz_positions[1], Lorentz_FWHMs[1], 0, 0)
    lor_funcs = lor_func_0 + lor_func_1
    offset = np.zeros_like(wavenumber)
    
    # optional: add linear offset:
    y1 = 8; y2 = -16
    offset = np.linspace(-1, 1, num=number_of_points) * (y2-y1)/2 + (y2+y1)/2

    # join baseline, noise, Lor functions and linear offset into the synthetic spectrum:
    full_f = synthetic_bl+random_noise+lor_funcs + offset
    # format it to an "ExpSpec" class:
    testspec = ExpSpec(wavenumber, full_f)

    # 2) fit it
    print('''let's fit it''')
    # l = multipeak_fit_with_BL(testspec, saveresults=True, remove_offset=False) #, the_lambda=2e6)
    # l = multipeak_fit_with_BL(testspec, saveresults=True, remove_offset=True) #, the_lambda=2e6)

    # full_f = synthetic_bl+random_noise+lor_funcs + 10
    # testspec = ExpSpec(wavenumber, full_f)

    # l = multipeak_fit_with_BL(testspec, saveresults=True, remove_offset=False) #, the_lambda=2e6)
    
    l = multipeak_fit_with_BL(testspec, saveresults=False, apply_corrections=True, the_lambda=2e6)
    