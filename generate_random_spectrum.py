# -*- coding: utf-8 -*-
"""
# How to use:
wn is x-axis (by default generated from 0 to 1024)
typically, you just call
    a, noise, baseline = generate_the_spectrum(number_of_peaks=2)
then, if you need to access some peak parameter, you type the parameter name like this:
    a.peaks[peak_number].params.position
    a.peaks[peak_number].params.fwhm
    a.peaks[peak_number].params.intensity # it's integral intensity
    a.peaks[peak_number].params.gaussshare
    a.peaks[peak_number].params.asym
    a.peaks[peak_number].peak_height # it's not a conventional parameter, the height
    
To get the shape of a particular peak separately, you type:
    a.peaks[peak_number].curve

To print all peak parameters:
    a.params
Achtung: disregard last two peaks with weird parameters
    (they are for the linear baseline, technically also described as peaks of types "bricks" and "tricks")
    
To save peak parameters to a file:
    a.save_params_to_txt(filename)

"""

import numpy as np
from spectralfeature import MultiPeak
from copy import deepcopy
# import pandas as pd
# from spectroutines import is_number
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [6.0, 3.2]
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# import time
from scipy.integrate import cumulative_trapezoid

def generate_the_spectrum(wn=np.linspace(0, 1024, 1025),
                          number_of_peaks=1,
                          baseline_multiplier=1e-4,
                          return_order=0,
                          function='asym_pV',
                          # peak amplitude
                          display=1):
    mp = MultiPeak(wn, number_of_peaks=number_of_peaks)
    
    the_noise = np.random.normal(0, 1, len(wn))
        # mp.d2baseline += the_noise
    
    # generate peaks with S/N ratio between the following numbers:
    SNratio_range = 1, 256
    peak_positions = np.random.uniform(np.min(wn), np.max(wn), number_of_peaks)
    peak_positions.sort()
    for pp in range(number_of_peaks):
        mp.peaks[pp].function=function
        mp.peaks[pp].params.position = peak_positions[pp]
        mp.peaks[pp].params.intensity = 1
        mp.peaks[pp].params.fwhm = np.random.uniform(mp.interpoint_distance, abs(np.min(wn)-np.max(wn))/16)
        if mp.peaks[pp].function == 'asym_pV':
            mp.peaks[pp].params.gaussshare = np.random.uniform(0, 1)
            mp.peaks[pp].params.asym = np.random.normal(0, 0.36/3) # /3 is because of 3sigma rule
            
        mp.peaks[pp].peak_height = SNratio_range[0] + abs(np.random.normal(0, SNratio_range[1]/3)) # /3 is because of 3sigma rule
        
        if mp.peaks[pp].function == 'asym_pV':
            print('peak at {:.2f}, fwhm={:.2f}, height={:.2f}, gaussshare={:.2f}, asym={:.2f}'.format(mp.peaks[pp].params.position, mp.peaks[pp].params.fwhm, mp.peaks[pp].peak_height, mp.peaks[pp].params.gaussshare, mp.peaks[pp].params.asym))
        else:
            print('peak at {:.2f}, fwhm={:.2f}, height={:.2f}'.format(mp.peaks[pp].params.position, mp.peaks[pp].params.fwhm, mp.peaks[pp].peak_height))

    def second_derivative_pseudoFourier_line(number_of_FourierCoeffs=255):
        Fourier_line_2ndDer = np.zeros_like(wn)
        
        # make x_local from -pi to pi, without requirement for even spacing (otherwise we'd use np.linspace)
        x_local = deepcopy(wn) 
        x_local -= 0.5 * (np.max(wn) + np.min(wn))
        x_local *= 2*np.pi / abs(np.max(wn) - np.min(wn))
    
        for i in range(number_of_FourierCoeffs):
            current_frequency = np.random.normal(0, 5)
            current_amplitude = np.random.normal(0, 1)
            current_prefactor = np.exp(-current_frequency**2 / 27) * current_amplitude
            current_phase = np.random.uniform(0, 2*np.pi)
            Fourier_line_2ndDer +=  current_prefactor * np.cos(x_local * current_frequency + current_phase)
        x0 = np.argmin(abs(x_local))
        Fourier_line_2ndDer *= baseline_multiplier
        Fourier_line_1stDer = cumulative_trapezoid(Fourier_line_2ndDer, x=wn, initial=0)
        Fourier_line_1stDer -= Fourier_line_1stDer[x0]
        Fourier_line_Integr = cumulative_trapezoid(Fourier_line_1stDer, x=wn, initial=0)
        Fourier_line_Integr -= Fourier_line_Integr[x0]
        return Fourier_line_Integr, Fourier_line_1stDer, Fourier_line_2ndDer

    baseline, bl_1stDer, bl_2ndDer = second_derivative_pseudoFourier_line()
    mp.d2baseline = baseline
    
    if display > 0:
        if return_order == 0:
            plt.plot(mp.wn, mp.curve+the_noise, 'r')
            plt.plot(mp.wn, mp.d2baseline, ':k')
            plt.show()
        if return_order == 1:
            plt.plot(mp.wn, np.gradient(mp.curve+the_noise), 'r')
            plt.plot(mp.wn, bl_1stDer, ':k')
            plt.show()
        if return_order == 2:
            plt.plot(mp.wn, np.gradient(np.gradient(mp.curve+the_noise)), 'r')
            plt.plot(mp.wn, bl_2ndDer, ':k')
            plt.show()
        print ('mp.params: \n', mp.params)
            
    if return_order == 0:
        return mp, the_noise, baseline
    if return_order == 1:
        return mp, np.gradient(the_noise), bl_1stDer
    if return_order == 2:
        return mp, np.gradient(np.gradient(the_noise)), bl_2ndDer



if __name__ == '__main__':
    
    current_multipeak, current_noise, current_baseline = generate_the_spectrum(number_of_peaks=1,
                                                                               baseline_multiplier=1e-4,
                                                                               return_order=0)

# # you should also try 
    # _ = generate_the_spectrum(number_of_peaks=2, return_order=1)
#     # and
    # _ = generate_the_spectrum(number_of_peaks=2, return_order=2)

    

    
