#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created

научный рабочий
"""

import numpy as np
from copy import deepcopy
from scipy.optimize import least_squares, minimize, Bounds, fmin_l_bfgs_b
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy import signal
from datetime import datetime
import colorsys
from itertools import cycle
from pathlib import Path
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
from spectroutines import *
from spectralfeature import *



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

    # step 0: check if we need to set fitting range
    if fitrange != None: 
        derspec.working_range = fitrange
        #@Test&Debug # 
        if display > 1:
            print('fitting range from input: ', derspec.working_range)

    # step 1: load startingpoint and check it
    if startingpoint=='auto_generate':
        # generate from find_peaks !!
        peak_positions = find_da_peaks(derspec, display=display)
        current_multipeak = MultiPeak(derspec.x, number_of_peaks=len(peak_positions))
        for i in range(len(peak_positions)):
            current_multipeak.peaks[i].params.position = peak_positions[i]
            current_multipeak.peaks[i].params.intensity = 1
    elif type(startingpoint)==str:
        current_multipeak = MultiPeak(derspec.x)
        current_multipeak.read_startingpoint_from_txt(startingpoint)
        # if display > 0:
        #     read_startingpoint_from_txt
    #         print('startingpoint file = {}, reading from file'.format(startingpoint))
    #     startingpoint = read_startingpoint_from_txt(startingpoint)
    #     print('(nan okay here)')
    # else:
    #     if display > 0:
    #         print('startingpoint from input array')

    # @Test&Debug: 
        # optional: write current starting point to file
    # write_startingpoint_to_file(startingpoint, filename='current_startingpoint.txt')

    # step 2: compute lambda
    if not the_lambda:
        the_lambda = auto_lambda(derspec.y, fwhm_to_pixelwidth_ratio='auto', pixelwidth=current_multipeak.interpoint_distance)

    if display > 0:
        print('in this run set the_lambda to {:.3e}'.format(the_lambda) )

    # step 3: initialize the multipeak (number_of_peaks)
    dermultipeak = deepcopy(current_multipeak) # MultiPeak(derspec.x, number_of_peaks)
    # dermultipeak.specs_array[:,0:4] = startingpoint
    # dermultipeak.specs_array[:,4] = 1
    dermultipeak.lam = the_lambda

    # current_multipeak would be updated at each iteration:
    # current_multipeak = MultiPeak(derspec.x, number_of_peaks)
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
    D = sparse.hstack([D, np.zeros((L, current_multipeak.number_of_peaks+2))])
    D = sparse.vstack([D, np.zeros((current_multipeak.number_of_peaks+2, L+current_multipeak.number_of_peaks+2))])
    D = sparse.csr_matrix(D)
    b_vector = np.zeros(L+current_multipeak.number_of_peaks+2)
    b_vector[0:L] = derspec.y
    
    def find_baseline_and_amplitudes(peakparams, display=2):
        """ This function calculates the norm of the deviation
            And updates the baseline and amplitudes to match the given peak parameters.
            Achtung! Peak parameters here are given as a vector, not array!
                (This vector is 1D, because the minimization takes a 1D vector)
            """        
        # calculate peak shapes:
        current_multipeak.optimization_params = peakparams
        for p in current_multipeak.peaks: # for all peaks, including bricks and tricks !
            p.params.intensity = 1
        # current_multipeak.specs_array[:,0:4] = np.reshape(peakparams, (current_multipeak.number_of_peaks, 4))
        # current_multipeak.specs_array[:,4] = 1 # set amplitudes to 1 to find them later !
        peak_shapes = current_multipeak.multicurve[:,:-1] # -1 because the last column is baseline
        
        # # add linear offset here:
        # current_multipeak.linear_baseline_scalarpart = current_multipeak.peaks[-2].curve # bricks
        # current_multipeak.linear_baseline_slopepart = current_multipeak.peaks[-1].curve # tricks
        peak_shapes = np.column_stack((peak_shapes , current_multipeak.peaks[-2].curve)) # bricks
        peak_shapes = np.column_stack((peak_shapes , current_multipeak.peaks[-1].curve)) # tricks
        
        # now set the values for the last rows and columns
        for current_peak in np.arange(current_multipeak.number_of_peaks+2):
            # construct it for off-diagonal also:
            # off-diagonal terms should be like
                # L1*L2 (scalar multiplication)
            for current_peak_2 in np.arange(current_multipeak.number_of_peaks+2):
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
        # the_amplitudes = full_solution[L:-2] #
        the_amplitudes = full_solution[L:] #
        for b in range(len(current_multipeak.peaks)): # for all, including bricks and tricks !
            current_multipeak.peaks[b].params.intensity = the_amplitudes[b]
        # current_multipeak.peaks[-2].params.intensity *= full_solution[-2] # bricks
        # current_multipeak.peaks[-1].params.intensity *= full_solution[-1] # tricks

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
        
        # current_multipeak.specs_array[:,4] = the_amplitudes
        # for a in range(current_multipeak.number_of_peaks):
        #     current_multipeak.peaks[a].params.intensity = the_amplitudes[a]

        norm_of_deviation = np.sum( (np.matmul(D2.toarray(), current_multipeak.d2baseline))**2 * the_lambda +
                                    (derspec.y - 
                                    # current_multipeak.d2baseline -
                                    # current_multipeak.linear_baseline -
                                    current_multipeak.curve)**2 )

        if supress_negative_peaks:
            min_intensity = np.min(current_multipeak.params.iloc[-1][0:-2])
            # (min(current_multipeak.peaks[a].params.intensity) for a in current_multipeak.peaks[0:-2])
            if min_intensity < 0:
                #@Test&Debug:
                # print('suppressing negative peaks: min intensity =', min_intensity)
                # EndofTest&Debug
                norm_of_deviation += 1
                norm_of_deviation *= 1e4

        return norm_of_deviation


    while True :
        # convert startingpoint and bounds to vectors:
        startingpoint1D = current_multipeak.optimization_params # np.reshape(startingpoint, 4*current_multipeak.number_of_peaks)
        bounds_low1D = current_multipeak.bounds[0] # np.reshape(bounds_low, 4*current_multipeak.number_of_peaks)
        bounds_high1D = current_multipeak.bounds[1] # np.reshape(bounds_high, 4*current_multipeak.number_of_peaks)
        
        try:
            solution = minimize(find_baseline_and_amplitudes,
                        startingpoint1D,
                        method = 'L-BFGS-B', #  'Nelder-Mead', #
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
            
            # converged_linear_offset = current_multipeak.linear_baseline
            converged_parameters = current_multipeak.optimization_params
            dermultipeak = deepcopy(current_multipeak)
            #     #@Test&Debug #
            if display > 1:
                print('least_squares converged')
            break
        except RuntimeError: # (RuntimeError, TypeError, NameError):
            #     #@Test&Debug #
            if display > 0:
                print('least_squares optimization error, expanding the fitting range')
            converged_parameters = current_multipeak.optimization_params
            dermultipeak.optimization_params = converged_parameters
            continue
        
    if saveresults:
        Path("fit_results/txt").mkdir(parents=True, exist_ok=True)
        Path("fit_results/pic").mkdir(parents=True, exist_ok=True)
        
    if apply_corrections:
        dermultipeak_before_corrections = deepcopy(dermultipeak)
        _ = dermultipeak.construct_corrections()
        # uncorrected_specs_array = deepcopy(dermultipeak.specs_array)
        # dermultipeak.specs_array += params_corrections_array
        # baseline_before_corrections = deepcopy(dermultipeak.baseline)
        # D2 = sparse.diags([1,-2,1],[-1,0,1], shape=(L,L))
        # D4 = D2.dot(D2.transpose())
        # D_short = deepcopy(D4) * the_lambda
        # w = np.ones(L)
        # W = sparse.spdiags(w, 0, L, L)
        # D_short += W
        # D_short = sparse.csr_matrix(D_short)
        # b_vector_short = derspec.y - dermultipeak.curve - dermultipeak.linear_baseline

        # corrected_baseline = spsolve(D_short, b_vector_short)
        # dermultipeak.d2baseline = corrected_baseline
        if display > 0:
            print("""applying corrections, please check if that's what you want""")

    if display > 0 or saveresults:
        now = datetime.now()
        if display > 0:
            color_list = get_colors(1+dermultipeak.number_of_peaks)
            cycol = cycle(color_list)
            plt.plot(derspec.x, derspec.y, 'o', mfc='none', ms = 4, mec=(0.1, 0.1, 0.1, 0.5),
                     label='raw data')
            plt.plot(dermultipeak.wn, dermultipeak.curve, # dermultipeak.d2baseline + + dermultipeak.linear_baseline,
                     color=next(cycol), label='fit envelope', linewidth=2);
            for current_peak in range(dermultipeak.number_of_peaks):
                plt.plot(dermultipeak.wn, dermultipeak.d2baseline + dermultipeak.linear_baseline + dermultipeak.multicurve[:,current_peak],
                         ':', color=next(cycol),
                         label='peak at {:.1f}'.format(dermultipeak.peaks[current_peak].params.position),
                         linewidth=1)
            if apply_corrections:
                plt.plot(dermultipeak.wn, dermultipeak_before_corrections.baseline,
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
                filename = 'fit_results/pic/' + filenameprefix + '_multipeak_conv_' + now.strftime("%H%M%S") + '.png'
                plt.savefig(filename, bbox_inches='tight', transparent=True)
            plt.show()

    # restoring the working range of the ExpSpec class:
    derspec.working_range = original_working_range
    if apply_corrections:
        return dermultipeak, dermultipeak_before_corrections
    else:
        return dermultipeak
    



if __name__ == '__main__':

    current_spectrum = np.genfromtxt('test_data_experimental_spectrum.txt') # read file to numpy format
    testspec = ExpSpec(current_spectrum[:,0], current_spectrum[:,1]) # convert the spectrum to an *object* of a specific format.        

    # optional: reduce number of points
    wn_interpolated = np.linspace(current_spectrum[0,0], current_spectrum[-1,0], 2048)
    intensity_interpolated = np.interp(wn_interpolated, current_spectrum[:,0], current_spectrum[:,1])
    testspec = ExpSpec(wn_interpolated, intensity_interpolated)

    current_startingpoint = 'test_spectrum_startingpoint.txt'

    dat_result = multipeak_fit_with_BL(testspec_interplated, saveresults=True,
                              the_lambda=1e8,
                              fitrange=(500, 3700),
                              startingpoint=current_startingpoint,
                              supress_negative_peaks=True,
                              )
    



    # # 1) generate test spectrum:
    # print('''let's generate a test spectrum with two Lor functions, sine-like baseline and random noise''')
    # number_of_points = 1025
    # wavenumber = np.linspace(0, 1024, num=number_of_points)
    # Lorentz_positions = (384, 720)
    # Lorentz_FWHMs = (32, 64)
    # amplitudes0 = (128*2, 128*8)
    # synthetic_bl = 2*np.sin(np.pi * wavenumber/256)
    # random_noise = 2*np.random.uniform(-1, 1, len(wavenumber))
    # dmp = MultiPeak(wavenumber, number_of_peaks=2)
    # dmp.peaks[0].function='Lorentz'
    # dmp.peaks[1].function='Gauss'
    # dmp.peaks[0].params.position = 384
    # dmp.peaks[0].params.fwhm = 32
    # dmp.peaks[0].params.intensity = 1e3
    # dmp.peaks[1].params.position = 720
    # dmp.peaks[1].params.fwhm = 64
    # dmp.peaks[1].params.intensity = 1e3
    # dmp.d2baseline += synthetic_bl + random_noise
    # plt.plot(dmp.wn, dmp.curve); plt.show()
    # # plt.plot(dmp.wn, dmp.baseline)
    # # plt.plot(dmp.wn, dmp.linear_baseline)

    # # 2) fit it
    # print('''let's fit it''')
    # l = multipeak_fit_with_BL(testspec, saveresults=True, the_lambda=1e7))
    


    