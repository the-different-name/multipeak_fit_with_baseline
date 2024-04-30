# -*- coding: utf-8 -*-
"""
# How to use:
wn is x-axis (by default generated from 0 to 1024)
typically, you just call
    a = generate_the_spectrum(number_of_peaks=2)
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
# from copy import deepcopy
# import pandas as pd
# from spectroutines import is_number
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [6.0, 3.2]
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def generate_the_spectrum(wn=np.linspace(0, 1024, 1025),
                          number_of_peaks=1,
                          generate_baseline=False,
                          generate_noise=True,
                          function='asym_pV',
                          display=1):
    mp = MultiPeak(wn, number_of_peaks=number_of_peaks)
    
    if generate_noise:
        # generate Gaussian noise with mean=0 and std=1:
        the_noise = np.random.normal(0, 1, len(wn))
        mp.d2baseline += the_noise
    if generate_baseline:
        print(' generate_baseline not yet implemented')
    
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
        np.random.uniform(SNratio_range[0], SNratio_range[1])
        
        if mp.peaks[pp].function == 'asym_pV':
            print('peak at {:.2f}, fwhm={:.2f}, height={:.2f}, gaussshare={:.2f}, asym={:.2f}'.format(mp.peaks[pp].params.position, mp.peaks[pp].params.fwhm, mp.peaks[pp].peak_height, mp.peaks[pp].params.gaussshare, mp.peaks[pp].params.asym))
        else:
            print('peak at {:.2f}, fwhm={:.2f}, height={:.2f}'.format(mp.peaks[pp].params.position, mp.peaks[pp].params.fwhm, mp.peaks[pp].peak_height))
    
    if display > 0:
        plt.plot(mp.wn, mp.curve)
        print ('mp.params: \n', mp.params)
    return mp


if __name__ == '__main__':

    a = generate_the_spectrum(number_of_peaks=2, display=1) # , function='Lorentz')

