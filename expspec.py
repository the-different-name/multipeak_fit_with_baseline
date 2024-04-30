#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:06:53 2020

@author: korepashki

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import detrend
from copy import deepcopy
from pathlib import Path
import matplotlib as mpl
from itertools import cycle
mpl.rcParams['figure.dpi'] = 300 # default resolution of the plot
from spectroutines import *


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
        


class MultiSpec ():
    def __init__(self, full_x, full_y) :
        self.full_x = full_x
        if len(np.shape(full_y)) == 1:
            self.full_y = np.reshape(full_y, (len(full_y), 1))
        else:
            self.full_y = full_y
        self.number_of_spectra = np.shape(self.full_y)[1]
        self.xrange = (np.min(full_x), np.max(full_x))
        self.x = full_x
        self.y = full_y
        self.lam = (len(full_x)/16)**4
        self.p = 1.5e-3
        
    @property
    def working_range(self):
        return self.xrange
    @working_range.setter
    def working_range(self, xrange):
        self.xrange = (np.maximum(np.min(xrange), np.min(self.full_x)), np.minimum(np.max(xrange), np.max(self.full_x)))
        self.x = self.full_x[np.where(np.logical_and(self.full_x>=np.amin(xrange), self.full_x<=np.amax(xrange)))]
        self.y = self.full_y[np.where(np.logical_and(self.full_x>=np.amin(xrange), self.full_x<=np.amax(xrange)))[0], :]
        # note the "[0]" above. Somehow it's important.
    
    
    def SVD_it(self, number_of_components_2plot=4, denoise=False, plot_it=True, **kwargs):
        options = {
            'filenameprefix' : 'current_',
            'number_of_components_to_keep' : 'auto', 
            'save_pictures' : True} # auto here is ~1/4 of number_of_spectra
        options.update(kwargs)
        # print(options)

        u, s, vh = np.linalg.svd(self.y, full_matrices=False)
        sx = np.arange(0, self.number_of_spectra, 1)
        if plot_it:
            plt.plot(sx, np.log10(s), 'o', mfc='none', ms = 6, mew=2, mec='r');
            # plt.xlim([-1, 0.2*sx[-1]]);
            plt.ylabel('Singular Value (log)')
            plt.xlabel("spectrum number")
            plot_annotation = 'SVD values'
            plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            if options['save_pictures']:
                Path('SVD').mkdir(parents=True, exist_ok=True)
                plt.savefig('SVD/' + options['filenameprefix']+'SVD_values' + '.png', format='png',  bbox_inches='tight')
            plt.show()

            plt.figure()
            color_list = get_colors(number_of_components_2plot)
            cycol = cycle(color_list)        
            for ip in range(number_of_components_2plot):
                plt.plot(self.x, u[:,ip], color=next(cycol))
            plt.ylabel('Intensity (U from SVD)')
            plt.xlabel("wavenumber / cm$^{-1}$")
            plot_annotation = 'SVD components'
            plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            if options['save_pictures']:
                plt.savefig('SVD/' + options['filenameprefix']+'SVD_components' + '.png', format='png',  bbox_inches='tight')
            plt.show()
            
            # plot SVD weights typically not needed
            # plt.figure()
            # color_list = get_colors(number_of_components_2plot)
            # cycol = cycle(color_list) 
            # for ip in range(number_of_components_2plot):
            #     plt.plot(sx, s[2][ip,:], color=next(cycol))
            # plt.ylabel('Impact (V from SVD)')
            # plt.xlabel("spectrum number")
            # plot_annotation = 'SVD weights'
            # plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            # if options['save_pictures']:
                # plt.savefig(filenameprefix+'SVD_weights' + '.png', format='png',  bbox_inches='tight')
            # # plt.savefig('SVD/' + options['filenameprefix']+'SVD_weights' + '.eps', format='eps',  bbox_inches='tight')
            # plt.show()
        
        if denoise:
            if options['number_of_components_to_keep']=='auto':
                number_of_components_to_keep = int(np.ceil(self.number_of_spectra / 4))
            else:
                number_of_components_to_keep = options['number_of_components_to_keep']
            print('leaving {} components'.format(number_of_components_to_keep))
            s[number_of_components_to_keep:] = 0
            self.y = np.dot(u * s, vh)
            
        return u, s, vh


    def NMF_it(self, number_of_components=3, plot_it=True, **kwargs):
        options = {
            'filenameprefix' : 'current_',
            'number_of_components_to_keep' : 'auto', 
            'save_pictures' : True,
            'normalize': False} # auto here is ~1/4 of number_of_spectra
        options.update(kwargs)
        print(options)
        from sklearn.decomposition import NMF
        y = deepcopy(self.y- np.min(self.y))
        if options['normalize'] == True:
            y2 = np.sum(y**2, axis=0)
            y2mean = np.mean(y2)
            for i in range(self.number_of_spectra):
                y[:,i] *= (y2mean/np.sum(y[:,i]**2))**0.5
            print('normalize it')

        model = NMF(n_components=number_of_components,
                    init='nndsvd',
                    # solver='mu',
                    # beta_loss=1,
                    max_iter=16000,
                    alpha_W=0.0, # 0.8 ?
                    l1_ratio=0.5, # 0.1 ?
                    verbose=0,
                    tol=1e-9)
    
        W = model.fit_transform(y)
        
        # plot components
        if plot_it:
            Path('NMF').mkdir(parents=True, exist_ok=True)
            color_list = get_colors(number_of_components)
            plt.figure()
            cycol = cycle(color_list)
            for iq in range(np.shape(W)[1]):
                ktnmpw = 1 # np.max(W[:,iq]) # = 1 meand that no normalization is imposed
                plt.plot(self.x, W[:,iq]/ktnmpw, color=next(cycol))
            plt.ylabel('Intensity') # ' (div. by max.)')
            plt.xlabel("wavenumber / cm$^{-1}$")
            if options['save_pictures'] == True:
                Path('NMF').mkdir(parents=True, exist_ok=True)
                plt.savefig('NMF/' + options['filenameprefix']+'auto_component_shapes' + '.png', format='png',  bbox_inches='tight');
            plt.show()
            
            # plot weights normalized by the sum (!)
            H = model.components_
            Hx = np.arange(0, np.shape(H)[1], 1)
            plt.figure()
            cycol = cycle(color_list)
            for iq in range(np.shape(H)[0]):
                plt.plot(Hx, H[iq, :] / np.sum(y, axis=0), 'o', mfc='none', ms = 6, mew=2, color=next(cycol))
            plt.ylabel('weight')
            plt.xlabel("spectrum number")
            if options['save_pictures'] == True:
                plt.savefig('NMF/' + options['filenameprefix']+'auto_component_weights' + '.png', format='png',  bbox_inches='tight')
            plt.show()
            
            plt.plot(Hx, np.sum(y, axis=0), 'o', mfc='none', ms = 6, mew=2, mec='k')
            plt.ylabel('total intensity')
            plt.xlabel("spectrum number")
            if options['save_pictures'] == True:
                plt.savefig('NMF/auto_total_intensity' + '.png', format='png',  bbox_inches='tight')
            plt.show()
            
            if options['save_pictures'] == True:
                np.savetxt('NMF/' + options['filenameprefix']+'auto_NMF_components.txt', W)
                np.savetxt('NMF/' + options['filenameprefix']+'auto_NMF_weights.txt', H)
                
        return H, W
    
    def test_baseline(self, spectrum_number=0, find_parameters=True):
        current_spectrum = ExpSpec(self.x, self.y[:, spectrum_number])
        if find_parameters:
            lam, p = find_both_optimal_parameters(current_spectrum)
            return lam, p
        else:
            _ = das_baseline(current_spectrum, display=2, als_lambda = self.lam, als_p_weight = self.p)
            return
    
    def de_baseline_it(self, display=0):
        # override the original spectrum range:
        self.full_x = deepcopy(self.x)
        self.full_y = deepcopy(self.y)
        for i in range(self.number_of_spectra):
            current_spectrum = ExpSpec(self.x, self.y[:, i])
            current_baseline = das_baseline(current_spectrum, display=display, als_lambda = self.lam, als_p_weight = self.p)
            self.y[:, i] -= current_baseline
            self.full_y[:, i] -= current_baseline


if __name__ == '__main__':
    
    # # let's import some map:
    testspec = np.genfromtxt('../../../../spectra/Raman/Panin_graphene/2023_10_04_graphene_Panin/raw/s3_map1_Senterra8553.dpt', delimiter=',')
    wn = testspec[:,0]
    mspec1 = MultiSpec(wn, testspec[:,1:])
    # k = mspec1.SVD_it(plot_it=False)
    # plt.plot(mspec1.x, mspec1.y); plt.show()

    # k = mspec1.SVD_it(denoise=True, plot_it=False, number_of_components_to_keep = 11)
    # plt.plot(mspec1.x, mspec1.y); plt.show()

    # k = mspec1.SVD_it(denoise=True, plot_it=True)
    # plt.plot(mspec1.x, mspec1.y); plt.show()    

    # l = mspec1.NMF_it()

    # mspec1.working_range = (500, 3000)
    # plt.plot(mspec1.x, mspec1.y)
    # mspec1.working_range = (-500, 6000)
    # plt.plot(mspec1.x, mspec1.y); plt.show()
    

    # mspec0 = MultiSpec(wn, testspec[:,1])
    # mspec0.working_range = (500, 3000)
    # plt.plot(mspec0.x, mspec0.y); plt.show()
    # plt.plot(mspec0.x, mspec0.y)
    # mspec0.working_range = (-500, 6000)
    # plt.plot(mspec0.x, mspec0.y); plt.show()

    mspec1.lam = 1e8
    mspec1.p = 6e-03

    # l, p = mspec1.test_baseline(-1)
    _ = mspec1.test_baseline(-5, find_parameters=False)

    