# -*- coding: utf-8 -*-
"""
# How to use:
#    MultiPeak is the Class.
#    Initialize it, for example, like
 number_of_points = 1025
 wavenumber = np.linspace(0, 1024, num=number_of_points)
 number_of_peaks = 1
 current_lambda = 1e6 # this is a regularization parameter
 current_multipeak = MultiPeak(wavenumber, number_of_peaks=number_of_peaks, lam=current_lambda)
# Then you can read parameters:
 current_multipeak.params[1]
 current_multipeak.params[1].position
# But if you need to set parameters, it should be done for a specific peak:
 current_multipeak.peaks[1].params.position = 400
# not like
 current_multipeak.params[1].position = 400 # this would not change anything
"""

import numpy as np
from copy import deepcopy
from scipy import sparse
from scipy.sparse.linalg import spsolve
import enum
import pandas as pd
from spectroutines import is_number


class PossibleFunctions(enum.Enum):
    bricks = {
        'function': 'bricks',
        'synonyms': ['b', 'bricks', 'briks', 'br'],
        'number_of_parameters': 1,
        'optimization_params_order': [],}
    tricks = {
        'function': 'tricks',
        'synonyms': ['t', 'tricks', 'triks', 'tr'],
        'number_of_parameters': 1,
        'optimization_params_order': [],}
    asym_pV = {
        'function': 'asym_pV',
        'optimization_params_order': ['position', 'fwhm', 'asym', 'gaussshare'],
        'synonyms': ['apv', 'asympv', 'pv-asym', 'asym-pv', 'asym_pv'],
        'number_of_parameters': 5}
    Gauss = {
        'function': 'Gauss',
        'optimization_params_order': ['position', 'fwhm'],
        'synonyms': ['gauss', 'gaussian', 'g'],
        'number_of_parameters': 3}
    Lorentz = {
            'function': 'Lorentz',
            'number_of_parameters': 3,
            'optimization_params_order': ['position', 'fwhm'],
            'synonyms': ['lor', 'l', 'lorentz', 'lorenz', 'lorens', 'lorents'],}
    Lorentz_asym_X = {
            'function': 'Lorentz_asym_X',
            'number_of_parameters': 3,
            'optimization_params_order': ['position', 'fwhm'],
            'synonyms': ['lor_asym_1', 'l_asym_1', 'lorentz_asym_x', 'lorentz_asym_1', 'lorentz_asym1'],}
    pV = {
        'function': 'pV',
        'optimization_params_order': ['position', 'fwhm', 'gaussshare'],
        'synonyms': ['pv', 'sympv', 'pv-sym', 'sym-pv'],
        'number_of_parameters': 4}
    def __init__(self, somefunc):
        self.synonyms = somefunc['synonyms']
        self.function = somefunc['function']
        self.number_of_parameters = somefunc['number_of_parameters']
        self.optimization_params_order = somefunc['optimization_params_order']
    @classmethod
    def set_function(self, name):
        for i in list(self):
            if name.lower() in i.synonyms: # or i.value['synonyms']:
                # print('found it!! i = ', i)
                return i

class SinglePeak () :
    """ 
    """

    def __init__(self, wn=np.linspace(0, 1, 129),
                 position='None',
                 fwhm='None',
                 asym=0,
                 gaussshare=0,
                 intensity=0,
                 linear_baseline_offset=0,
                 linear_baseline_slope=0,
                 function='asym_pV') :
        self.wn = wn
        # set function:
        f = PossibleFunctions.set_function(function)
        self.function = f.function
        # self.optimization_params_order = [] #  list(['position', 'fwhm', 'asym', 'gaussshare', 'wawa'])
        self.optimization_params_order = f.optimization_params_order
        self.number_of_parameters = f.number_of_parameters
        self.optimization_params_order = f.optimization_params_order
        if fwhm=='None':
            fwhm = abs((self.wn[-1]-self.wn[0]) / (len(self.wn)-1)) # interpoint distance
        if position=='None':
            # default position is at the center
            # self.params.position = (wn[-1]-wn[0])/2
            position = (wn[-1]-wn[0])/2
        self.params = pd.Series([position], index=['position'], dtype=np.float64)
        
        if 'fwhm' in f.optimization_params_order:
            self.params['fwhm'] = fwhm
        if 'asym' in f.optimization_params_order:
            self.params['asym'] = asym
        if 'gaussshare' in f.optimization_params_order:
            self.params['gaussshare'] = gaussshare            

# LB and UB are bounds for the optimization
        self.params['intensity'] = intensity # append intensity at the end !
        self.LB = deepcopy(self.params) # low boundary for optimization
        self.UB = deepcopy(self.params) # top boundary for optimization
        self.LB[:] = -np.inf
        self.UB[:] = np.inf
        if 'fwhm' in f.optimization_params_order:
            self.LB.fwhm = abs((self.wn[-1]-self.wn[0]) / (len(self.wn)-1)) # interpoint distance
            self.UB.fwhm = (self.wn[-1]-self.wn[0]) / 3 # 1/3 of the spectral range
        if 'asym' in f.optimization_params_order:
            self.LB.asym = -0.36; self.UB.asym = 0.36
        if 'gaussshare' in f.optimization_params_order:
            self.LB.gaussshare = 0; self.UB.gaussshare = 1
        if 'position' in f.optimization_params_order:
            self.UB.position = self.params.position + abs(self.wn[-1]-self.wn[0]) / 3 # if bounds are absent, set to position +- 1/3 of spectral range
            self.LB.position = self.params.position - abs(self.wn[-1]-self.wn[0]) / 3 # if bounds are absent, set to position +- 1/3 of spectral range


    @property
    def peak_area (self) :
        if self.function=='asym_pV':
            peak_area = (1 - self.params.gaussshare) * self.params.intensity * (1 + 0.69*self.params.asym**2 + 1.35 * self.params.asym**4) + self.params.gaussshare * self.params.intensity * (1 + 0.67*self.params.asym**2 + 3.43*self.params.asym**4)
        else:
            peak_area = self.params.intensity
        return peak_area

    @property
    def peak_height (self) :
        peak_height = self.params.intensity*getattr(self, self.function)(0)
        return peak_height
    @peak_height.setter
    def peak_height(self, newheight):
        self.params.intensity = newheight / getattr(self, self.function)(0)

    @property
    def fwhm_asym (self) :
        """ real fwhm of an asymmetric peak"""
        if self.function=='asym_pV':
            fwhm_asym = self.params.fwhm * (1 + 0.4*self.params.asym**2 + 1.35*self.params.asym**4)
        else: 
            fwhm_asym = self.params.fwhm
        return fwhm_asym
    
    @property
    def curve (self) :
        """ curve of the peak for a given set of parameters
        with real intensity
        """
        curve = self.params.intensity * getattr(self, self.function)(self.wn-self.params.position)
        return curve

    @property
    def shape (self) :
        """ shape of the peak for a given set of parameters
        with intensity = 1
        """
        shape = getattr(self, self.function)(self.wn-self.params.position)
        return shape

    def asym_pV(self, wn): # =self.wn #.wn, self.params.fwhm, self.params.asym, self.params.gaussshare):
        """ returns asymmetric pseudo-voigt profile composed of Gaussian and Lorentzian,
              which would be normalized by unit area if symmetric
              The funtion is defined in Analyst: 10.1039/C8AN00710A"""
        x_distorted = wn*(1 - np.exp(-wn**2/(2*(2*self.params.fwhm)**2))*self.params.asym*wn/self.params.fwhm)
        Lor_asym = self.params.fwhm / (x_distorted**2+self.params.fwhm**2/4) / (2*np.pi)
        Gauss_asym = (4*np.log(2)/np.pi)**0.5/self.params.fwhm * np.exp(-(x_distorted**2*4*np.log(2))/self.params.fwhm**2)
        voigt_asym = (1-self.params.gaussshare)*Lor_asym + self.params.gaussshare*Gauss_asym
        return voigt_asym
    
    def Gauss(self, wn): #.wn, self.params.fwhm, self.params.asym, self.params.gaussshare):
        """ returns Gaussian normalized by unit area"""
        Gauss = (4*np.log(2)/np.pi)**0.5/self.params.fwhm * np.exp(-(wn**2*4*np.log(2))/self.params.fwhm**2)
        return Gauss

    def Lorentz(self, wn): #.wn, self.params.fwhm, self.params.asym, self.params.gaussshare):
        """ returns Lorentzian normalized by unit area"""
        Lorentz = self.params.fwhm / (wn**2+self.params.fwhm**2/4) / (2*np.pi)
        return Lorentz
    
    def Lorentz_asym_X(self, wn): #.wn, self.params.fwhm, self.params.asym, self.params.gaussshare):
        """ returns Lorentzian normalized by unit area"""
        Lorentz_asym_X = wn * self.params.fwhm / (wn**2+self.params.fwhm**2/4) / (2*np.pi)
        return Lorentz_asym_X

    def pV(self, wn): # =self.wn #.wn, self.params.fwhm, self.params.asym, self.params.gaussshare):
        """ returns symmetric pseudo-voigt profile composed of Gaussian and Lorentzian"""
        Lor = self.params.fwhm / (wn**2+self.params.fwhm**2/4) / (2*np.pi)
        Gauss = (4*np.log(2)/np.pi)**0.5/self.params.fwhm * np.exp(-(wn**2*4*np.log(2))/self.params.fwhm**2)
        voigt_sym = (1-self.params.gaussshare)*Lor + self.params.gaussshare*Gauss
        return voigt_sym

    def bricks(self, wn):
        """ offset (scalar part) of the linear baseline"""
        return np.ones_like(wn)
    def tricks(self, wn):
        """ slope part of the linear baseline"""
        return np.linspace(-1, 1, len(wn))

    @property
    def optimization_parameters (self) :
        return self.params[0:-1] 
    @optimization_parameters.setter
    def optimization_parameters (self, parameters) :
        self.params[0:-1] = parameters


class MultiPeak ():
    """ 
    last two 'peaks' are 'bricks' and 'tricks'.
    >>>> number_of_peaks does not include 'bricks' and 'tricks'.

    """
    def __init__(self, wn=np.linspace(0, 1, 129), number_of_peaks=1, lam=1e5) :
        self.wn = wn
        self.lam = lam
        self.number_of_peaks = number_of_peaks
        self.d2baseline = np.zeros_like(wn) # "flexible" part of the baseline, without the linear part
        self.peaks = []
        for i in range(number_of_peaks):
            self.peaks.append(SinglePeak(wn))
        self.peaks.append(SinglePeak(wn, function='bricks'))
        self.peaks.append(SinglePeak(wn, function='tricks'))
        self.interpoint_distance = abs((self.wn[-1]-self.wn[0]) / (len(self.wn)-1))

# make a vector of optimization parameters
    @property
    def optimization_params (self):
        optimization_params = []
        for i in range(self.number_of_peaks):
            optimization_params = np.append(optimization_params, self.peaks[i].params[0:-1])
        return optimization_params
    @optimization_params.setter # works only with array, not with a single element!
    def optimization_params (self, the_params):
        # split params into separate peak params:
        the_params = np.split(the_params, np.cumsum([len(p.params)-1 for p in self.peaks[0:-2]])) ##check if it's -2 here!
        for i in range(self.number_of_peaks):
            self.peaks[i].params[0:-1] = the_params[i]
# @Test&Debug:
            # print (the_params[i])
# EndOf @Test&Debug:

# make a vector parameters
    @property
    def params (self):
        params = pd.DataFrame()
        for current_peak in self.peaks:
            params = pd.concat([params, current_peak.params], axis=1, ignore_index=True) 
        return params
# setter doesn't work
    # @params.setter # works only with array, not with a single element!
    # def params(self, the_params):
    #     print('called setter')
    #     self.params = the_params
    #     # tmpparam = deepcopy(self.params)
    #     # tmpparam = the_param
    #     # print('the_param = \n', the_param)
    #     # print('tmpparam = \n', tmpparam)
    #     # for i in range(self.number_of_peaks+2):
    #     #     self.peaks[i].params[:] = tmpparam[i][:]


# make a vector of optimization boundaries
    @property
    def bounds (self):
        LB = []
        UB = []
        for i in range(self.number_of_peaks):
            # for j in range(len(self.peaks[i].optimization_params_order)):
            LB = np.append(LB, self.peaks[i].LB[0:-1])
            UB = np.append(UB, self.peaks[i].UB[0:-1])
        return LB, UB
    @bounds.setter # works only with array, not with a single element!
    def bounds (self, bounds):
        LB = bounds[0]
        UB = bounds[1]
        a = 0
        for i in range(self.number_of_peaks):
            for j in range(len(self.peaks[i].params)-1):
                self.peaks[i].LB[j] = LB[a]
                self.peaks[i].UB[j] = UB[a]
                a += 1

# @Test&Debug:
            # print (the_params[i])
# EndOf @Test&Debug:


    @property
    def linear_baseline_offset(self):
        return self.peaks[-2].params.intensity
    @linear_baseline_offset.setter 
    def linear_baseline_offset(self, offset):
        self.peaks[-2].params.intensity = offset
    @property
    def linear_baseline_slope(self):
        return self.peaks[-1].params.intensity
    @linear_baseline_slope.setter 
    def linear_baseline_slope(self, slope):
        self.peaks[-1].params.intensity = slope

    
    @property
    def linear_baseline(self):
        """ Returns linear baseline.
        Coefficients are defined in terms of centered x-axis with unit span.
        """
        return self.peaks[-1].curve + self.peaks[-2].curve
    
    @property
    def baseline(self):
        return self.linear_baseline + self.d2baseline
    # setter shouldn't be called this way

    @property
    def curve(self):
        """ curve with baseline"""
        curve = sum([p.curve for p in self.peaks[0:-2]]) + self.baseline
        return curve

    @property
    def multicurve (self):
        """ array of curves
        """
        multicurve = np.zeros((len(self.wn), self.number_of_peaks+1))
        for i in range(self.number_of_peaks):
            multicurve[:,i] = self.peaks[i].curve
        multicurve[:,-1] = self.baseline
        return multicurve

    def construct_corrections(self, intensity=True, fwhm=True, position=False, gaussshare=False, asym=False, brickstricks=True):
        # construct extended x-axis:
        el = int(np.floor(len(self.wn)/2))
        wn_extended_boolean = np.concatenate((np.zeros(el), np.ones(len(self.wn)), np.zeros(el)))
        additional_interval = np.linspace(0, self.interpoint_distance*(el-1), el)
        wn_extended = np.concatenate((additional_interval+(self.wn[0]-additional_interval[-1]-self.interpoint_distance),
                                          self.wn,
                                          additional_interval+self.wn[-1]+self.interpoint_distance))
        bricks_extended = np.concatenate((np.zeros(el),
                                          self.peaks[-2].curve,
                                          np.zeros(el)))
        tricks_extended = np.concatenate((np.zeros(el),
                                          self.peaks[-1].curve,
                                          np.zeros(el)))
        self.wn = wn_extended
        for p in self.peaks:
            p.wn  = wn_extended
        
        # self.derivatives_array = self.construct_derivatives_over_params()
        # self.specs_array = deepcopy(multipeak.specs_array)

        # S, w, W and LSS1 are needed for theLinearOperator, but it's useful to pre-compute'em
        S = sparse.diags([1,-2,1],[-1,0,1], shape=(len(wn_extended),len(wn_extended)))
        w = np.ones(len(wn_extended))
        W = sparse.spdiags(w, 0, len(wn_extended), len(wn_extended)) # diagonal 1-matrix
        LSS1 = S.dot(S.transpose()) * self.lam + W

        parameters_order = [] # this vector is needed to construct derivatives, alpha and beta
        for p in self.peaks[0:-2]:
            parameters_order += p.params.index.tolist() # old: + ['intensity']
        parameters_order += ('intensity', 'intensity') # for bricks and tricks
        parameters_order_filter = np.zeros(len(parameters_order)) # here we make a filter for corrections. We would not need all of them!
        correctionslist = []
        if intensity:
            correctionslist.append('intensity')
        if fwhm:
            correctionslist.append('fwhm')
        if position:
            correctionslist.append('position')
        if asym:
            correctionslist.append('asym')
        if gaussshare:
            correctionslist.append('gaussshare')
        for ipp in range(len(parameters_order)):
            if parameters_order[ipp] in correctionslist:
                parameters_order_filter[ipp] = 1
        if brickstricks:
            parameters_order_filter[-2:] = 1

        def theLinearOperator(self, somevector):
            i0 = spsolve(LSS1, somevector)
            lambdified_vector = i0 - somevector
            return lambdified_vector

        def construct_derivatives_over_params(self, shift_over_param=128):
            """ Shift over params by default = 1/128.
                To make it smaller, set, say, shift_over_param=512
                """
            derivatives_array = np.zeros(( len(self.wn), len(parameters_order)))
            
            current_peak_number = 0
            for k in range(len(parameters_order)):
                if parameters_order[k] == 'position':
                    position_shift = (1/shift_over_param) * abs((self.wn[-1]-self.wn[0]) / (len(self.wn)-1)) # 1/shift_over_param of interpoint distance
                    shifted_peak_1 = deepcopy(self.peaks[current_peak_number])
                    shifted_peak_1.params.position += position_shift
                    shifted_peak_2 = deepcopy(self.peaks[current_peak_number])
                    shifted_peak_2.params.position -= position_shift
                    derivatives_array[:, k] = (shifted_peak_1.curve - shifted_peak_2.curve) / (2*position_shift)
                elif parameters_order[k] == 'fwhm':
                    fwhm_shift = (1/shift_over_param) * abs(self.peaks[current_peak_number].params.fwhm) # 1/shift_over_param of current fwhm
                    shifted_peak_1 = deepcopy(self.peaks[current_peak_number])
                    shifted_peak_1.params.fwhm += fwhm_shift
                    shifted_peak_2 = deepcopy(self.peaks[current_peak_number])
                    shifted_peak_2.params.fwhm -= fwhm_shift                    
                    derivatives_array[:, k] = (shifted_peak_1.curve - shifted_peak_2.curve) / (2*fwhm_shift)
                elif parameters_order[k] == 'gaussshare':
                    gs_shift = (1/shift_over_param) * np.sign (0.5-self.peaks[current_peak_number].params.gaussshare) 
                    if gs_shift == 0: # in case if Gaussian_share was 0.5 eggzakktly
                        gs_shift = 1/shift_over_param
                    shifted_peak = deepcopy(self.peaks[current_peak_number])
                    shifted_peak.params.gaussshare += gs_shift
                    derivatives_array[:, k] = (shifted_peak.curve - self.peaks[current_peak_number].curve) / gs_shift
                elif parameters_order[k] == 'asym':
                    asym_shift = (1/shift_over_param)
                    shifted_peak_1 = deepcopy(self.peaks[current_peak_number])
                    shifted_peak_1.params.asym += asym_shift
                    shifted_peak_2 = deepcopy(self.peaks[current_peak_number])
                    shifted_peak_2.params.asym -= asym_shift                    
                    derivatives_array[:, k] = (shifted_peak_1.curve - shifted_peak_2.curve) / (2*asym_shift)                    
                elif parameters_order[k] == 'intensity':
                    if self.peaks[current_peak_number].function == 'bricks':
# @Test&Debug:
                        # print('function == bricks')
# end of @Test&Debug
                        derivatives_array[:, k] = self.peaks[current_peak_number].shape * wn_extended_boolean
                    elif self.peaks[current_peak_number].function == 'tricks':
                        derivatives_array[:, k] = self.peaks[current_peak_number].shape * wn_extended_boolean
                        derivatives_array[:, k] /= np.max(derivatives_array[:, k])
# @Test&Debug:
                        # plt.plot(self.wn, derivatives_array[:, k], 'k'); plt.show()
                        # print('function == tricks, max and min = {}, {}'.format(np.max(derivatives_array[:, k]), np.min(derivatives_array[:, k])))
# end of @Test&Debug
                    else:
                        derivatives_array[:, k] = self.peaks[current_peak_number].shape
                    current_peak_number += 1
                else:
                    print('peak number {}, derivative over {} not implemented'.format(current_peak_number, parameters_order[k]))
            return derivatives_array


# @Test&Debug:        
        # a = theLinearOperator(self, bricks_extended)
        # plt.plot(wn_extended, bricks_extended); plt.show()
        # plt.plot(wn_extended, a); plt.show()
        # b = theLinearOperator(self, tricks_extended)
        # plt.plot(wn_extended, tricks_extended); plt.show()
        # plt.plot(wn_extended, b); plt.show()
        
        # getattr(self, self.function)(0)
        # p1c = getattr(self.peaks[0], self.peaks[0].function)(self.wn - self.peaks[0].params.position)  # / self.peaks[0].params.intensity #  * getattr(self.peaks[1], self.peaks[1].function)(wn_extended-self.peaks[1].params.position)
        # c = theLinearOperator(self, p1c)
        # plt.plot(wn_extended, p1c, 'k'); plt.show()
        # plt.plot(wn_extended, c, 'b'); plt.show()
        
        # plt.plot(self.wn, self.peaks[1].curve, 'b');
        # self.peaks[1].params.position += 400
        # self.peaks[1].params.fwhm *= 4
        # self.peaks[1].params.intensity *= 16
        # plt.plot(self.wn, self.peaks[1].curve, 'r');
        # plt.show()
#end-of-@Test&Debug

        the_derivatives = construct_derivatives_over_params(self)

# @Test&Debug:
        # print('parameters_order:', parameters_order)
        # plt.plot(wn_extended, the_derivatives[:,0], 'k'); plt.show()
        # plt.plot(wn_extended, the_derivatives[:,1], ':k'); plt.show()
        # plt.plot(wn_extended, theLinearOperator(self, the_derivatives[:,1]), 'r'); plt.show() # over fwhm
        # plt.plot(wn_extended, the_derivatives[:,2], 'g'); plt.show()
        # plt.plot(wn_extended, the_derivatives[:,3], 'b'); plt.show()
        # plt.plot(wn_extended, the_derivatives[:,4], 'y'); plt.show()
# end of @Test&Debug

        def construct_alpha_matrix(self):
            alpha_matrix = np.zeros((len(parameters_order), len(parameters_order)))
            
            for i in range(len(parameters_order)):
                for j in range(len(parameters_order)):
                    # intensity=True, fwhm=True, position=False, gaussshare=False, asymmetry=False, brickstricks=True
                    alpha_matrix[i, j] = np.sum(theLinearOperator(self, the_derivatives[:,i]) *
                                                theLinearOperator(self, the_derivatives[:,j]))
                    alpha_matrix[i, j] *= parameters_order_filter[i] * parameters_order_filter[j] # filter out the corrections that we don't need
            return alpha_matrix

        def construct_beta(self):
            beta_vector = np.zeros((len(parameters_order)))
            # the_derivatives
            for i in range(len(parameters_order)):
                # d2baseline without bricks and tricks
                baseline_at_beta = ( np.concatenate((np.zeros(el), self.d2baseline, np.zeros(el))))
                beta_vector[i] = np.sum(baseline_at_beta * theLinearOperator(self, the_derivatives[:, i]))
            return beta_vector

        alpha_matrix = construct_alpha_matrix(self)
        beta_vector = construct_beta(self)
        # params_corrections_vector = np.linalg.solve(alpha_matrix, beta_vector)
        params_corrections_vector,_,_,_ = np.linalg.lstsq(alpha_matrix, beta_vector)
        
#@Test&Debug:            
        # np.savetxt('fit_results/tmp/current_alpha_matrix.txt', alpha_matrix)
#end-of-@Test&Debug

# write corrections to peak parameters:
        current_peak_number = 0
        for i in range(len(parameters_order)):
            self.peaks[current_peak_number].params[parameters_order[i]] += params_corrections_vector[i]
            if parameters_order[i] == 'intensity':
                current_peak_number += 1
# convert back to original wavenumber axis:
        for p in self.peaks:
            p.wn  = wn_extended[wn_extended_boolean.astype(bool)]
        self.wn = wn_extended[wn_extended_boolean.astype(bool)]


        # baseline_before_corrections = deepcopy(dermultipeak.baseline)
        # D2 = sparse.diags([1,-2,1],[-1,0,1], shape=(L,L))
         # S = sparse.diags([1,-2,1],[-1,0,1], shape=(len(wn_extended),len(wn_extended)))
        # D4 = D2.dot(D2.transpose())
        # D_short = deepcopy(D4) * the_lambda
        # w = np.ones(L)
        # W = sparse.spdiags(w, 0, L, L)
        # D_short += W
        # D_short = sparse.csr_matrix(D_short)
        # b_vector_short = derspec.y - dermultipeak.curve - dermultipeak.linear_baseline

        # corrected_baseline = spsolve(D_short, b_vector_short)
        # dermultipeak.d2baseline = corrected_baseline
        # if display > 0:
        #     print("""applying corrections, please check if that's what you want""")


#@Test&Debug:            
        # plt.plot(self.wn, self.peaks[1].curve, ':k'); plt.show()
#end-of-@Test&Debug


    def save_params_to_txt(self, filename='current_peakparams.txt'):
        # print(self.params.to_markdown(numalign="right", floatfmt=".4e"))
        with open(filename, 'w') as f:
            f.write(self.params.to_markdown(numalign="right", floatfmt=".4e"))
    # def read_params_from_txt(self, filename):
    #     current_params = pd.read_table(filename, sep="|", skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]

    # def write_decomposition_to_txt(self, exp_y='zero', filename='current_decomp.txt'):
    #     # save decomposition to txt
    #     the_header = '' 
    #     col1 = ' x axis'
    #     col2 = ' raw data'
    #     col3 = ' baseline'
    #     the_header += f'{col1:<25}' + f'{col2:<25}' + f'{col3:<25}'
    #     for current_peak in range(self.number_of_peaks):
    #         col_n = ' peak at {:.1f}'.format(self.specs_array[current_peak,0])
    #         the_header += f'{col_n:<25}'
        
    #     if exp_y=='zero':
    #         exp_y = np.zeros_like(self.wn)
    #     current_decomposition = np.column_stack((self.wn, exp_y, self.baseline))
    #     current_decomposition = np.column_stack((current_decomposition, self.multicurve))
    #     np.savetxt(filename,
    #                current_decomposition,
    #                fmt='% .17e',
    #                header = the_header)



    def read_startingpoint_from_txt(self, filename):
        """ ???
        """
        with open(filename) as input_data:
            # locate the beginning of the startingpoint:
            while True:
                line = input_data.readline()
                if line.lower().strip().startswith('#') == True:
                    continue
                else:
                    break
            self.number_of_peaks = 0
            self.peaks = []
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
                    string_number = 0
                    parameter_index = 0                
                    tmpv = currentline[string_number]
                    if not is_number(tmpv): # then tmpv is a peak type
                        self.peaks.append(SinglePeak(self.wn, function=tmpv, position = float(currentline[string_number+1])))
                        # self.number_of_peaks+= 1
                        string_number += 1
                    else:
                        self.peaks.append(SinglePeak(self.wn, function='asym_pV', position = float(currentline[string_number])))
                        # self.number_of_peaks+= 1
                    current_peak = self.peaks[self.number_of_peaks-1]
                    while True:
                        try:
                            tmpv = currentline[string_number]
                            tmpv = float(tmpv) # np.asarray(tmpv, dtype=float)
                            if parameter_index >= len(current_peak.optimization_params_order):
                                # print('  ACHTUNG !!\nstartingpoint has a wrong number of parameters for peak {} at {}'.format(self.number_of_peaks, current_peak.params.position))
                                string_number += 1 # ???
                                break
                            else:
                                # assert parameter_index < len(current_peak.optimization_params_order), "startingpoint has a wrong number of parameters !"
                                current_peak.params[current_peak.optimization_params_order[parameter_index]] = deepcopy(tmpv)
                                string_number += 1

                                if currentline[string_number].startswith('('):
                                    constraint_L = currentline[string_number]
                                    constraint_L = constraint_L.replace('(', '')
                                    if is_number(constraint_L):
                                        constraint_L = float(constraint_L)
                                        if not np.isnan(constraint_L):
                                            current_peak.LB[current_peak.optimization_params_order[parameter_index]] = deepcopy(constraint_L)
                                    string_number += 1
                                    constraint_U = currentline[string_number]
                                    constraint_U = constraint_U.replace(')', '')
                                    if is_number(constraint_U):
                                        constraint_U = float(constraint_U)
                                        if not np.isnan(constraint_U):
                                            current_peak.UB[current_peak.optimization_params_order[parameter_index]] = deepcopy(constraint_U)
                                    string_number += 1
                            parameter_index +=1
                        except IndexError:
                            break
                line = input_data.readline()
                if not line:
                    # print('reading finished')
                    break
        assert self.check_startingpoint() == 1, "startingpoint read from file; check startingpoint ERROR !"
        self.number_of_peaks = len(self.peaks)
        self.peaks.append(SinglePeak(self.wn, function='bricks'))
        self.peaks.append(SinglePeak(self.wn, function='tricks'))



    def check_startingpoint(self):
        # check if LB <= UB
        for i in range(self.number_of_peaks):
            current_peak = self.peaks[i]
            for parameter_number in range(len(current_peak.optimization_parameters)):
                # if LB > UB:
                if current_peak.LB[current_peak.optimization_params_order[parameter_number]] > current_peak.UB[current_peak.optimization_params_order[parameter_number]]:
                    print('Error! \n LB > UB')
                    print (' Peak number ', i+1, ', \N{greek small letter omega} = ', current_peak.params.position, ' cm\N{superscript minus}\N{superscript one}', sep = '')
                    raise ValueError
                    return 0
        # # check if Starting value of a parameter is within the [LB, UB] interval
        # for i in range(self.number_of_peaks):
        #     for parameter_number in range(np.shape(self.startingpoint)[1]):
                if current_peak.params[current_peak.optimization_params_order[parameter_number]] < current_peak.LB[current_peak.optimization_params_order[parameter_number]] or current_peak.UB[current_peak.optimization_params_order[parameter_number]] < current_peak.params[current_peak.optimization_params_order[parameter_number]]:
                    print('Error! \n Starting value of a parameter is not within the [LB, UB] interval')
                    print (' Peak number ', i+1, ', \N{greek small letter omega} = ', current_peak.params.position, ' cm\N{superscript minus}\N{superscript one}', sep = '')
                    raise ValueError
                    return 0
        return 1

    def write_startingpoint_to_file(self, filename='auto_generated_startingpoint.txt'):
        """ startingpoint is a list (or a tuple):
            (peak_types, startingpoint array, LB, UB)
            """
        with open(filename, 'w', encoding='utf-8') as the_file:
            startingpoint_header = ' # auto-generated startingpoint \n'
            startingpoint_header += ' # the column labels below are for the asym_pV peak type \n'
            startingpoint_header +=  '# nan (not-a-number) are okay here \n'
            startingpoint_header += '# peak_types' + 14*' '
            startingpoint_header += 'positions' + 22*' '
            startingpoint_header += ' fwhms' + 26*' '
            startingpoint_header += ' [asyms]' + 25*' '
            startingpoint_header += '[Gauss_shares]'  + 18*' '
            startingpoint_header += '[amplitudes]' + 18*' '
            startingpoint_header += '\n'
            the_file.write(startingpoint_header)
            for i in range(self.number_of_peaks):
                current_peak = self.peaks[i]
                current_line  = '  {:18s}  '.format(current_peak.function)
                # current_peak.params[current_peak.optimization_params_order[parameter_number]] < current_peak.LB[current_peak.optimization_params_order[parameter_number]] or current_peak.UB[current_peak.optimization_params_order[parameter_number]] < current_peak.params[current_peak.optimization_params_order[parameter_number]]:
                # for j in range(np.shape(self.startingpoint)[1]):
                for parameter_number in range(len(current_peak.optimization_parameters)):
                    current_line +=  '{:8.2f}  '.format(current_peak.params[current_peak.optimization_params_order[parameter_number]])
                    current_line += '({:8.2f}, '.format(current_peak.LB[current_peak.optimization_params_order[parameter_number]])
                    current_line += '{:8.2f})  '.format(current_peak.UB[current_peak.optimization_params_order[parameter_number]])
                current_line += '{:8.2f}  '.format(current_peak.params.intensity)
                the_file.write(current_line + '\n')



if __name__ == '__main__':
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['figure.figsize'] = [6.0, 3.2]
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    # sp = SinglePeak(np.linspace(0, 1024, 1025))
    # (self, wn=np.linspace(0, 1, 129), position=0, fwhm=1, asym=0, gaussshare=0, intensity=0, function='asym_pV')
    mp = MultiPeak(np.linspace(0, 1024, 1025), number_of_peaks=3)
    print('mp.peaks[0].position', mp.peaks[0].params.position)
    
    
    
    current_startingpoint = 'test_spectrum_startingpoint.txt'
    mp.read_startingpoint_from_txt(current_startingpoint)
    
    # # mp.optimization_params
    # # mp.optimization_params = [1, 2, 0.0, 0.0,   101, 80, 0.0, 0.0,   601, 16, 0.0, 0.0]
    # mp.peaks[0].params.position = 200
    # mp.peaks[0].params.intensity = 2e3
    # mp.peaks[1].params.position = 400
    # mp.peaks[1].params.intensity = 160
    # mp.peaks[2].params.intensity = 31
    # mp.linear_baseline_slope=1
    # mp.linear_baseline_offset=1
    # # plt.plot(mp.wn, mp.peaks[-1].curve)
    # plt.plot(mp.wn, mp.curve)
    # # plt.plot(mp.wn, np.sum(p.curve for p in mp.peaks[0:-2]))
    # mp2 = deepcopy(mp)
    
    # mp2.construct_corrections()



    # # number_of_points = 1025
    # # wavenumber = np.linspace(-400, 624, num=number_of_points)

    # number_of_points = 1025
    # wavenumber = np.linspace(-400, 99624, num=number_of_points)
    # new_mp = MultiPeak(wavenumber)

    # new_mp.read_startingpoint_from_txt('test_read_startingpoint.txt')
    # # new_mp.write_startingpoint_to_file('test_write_sp_integraB.txt')


    
# # now test alpha matrix by Larkin's:    

#     wavenumber = np.linspace(0, 1024, num=number_of_points)
#     Lorentz_positions = (301)
#     Lorentz_FWHMs = (160)
#     amplitudes0 = 160 # (np.pi * 32) # (128*2, 128*8)


#     new_mp.peaks[0].params.intensity = amplitudes0
#     # new_mp.peaks[0].params.intensity = amplitudes0
#     new_mp.optimization_params = [Lorentz_positions, Lorentz_FWHMs, 0.0, 0]
#     new_mp.lam = 2**17
#     synthetic_bl = -((624 - wavenumber)*wavenumber*(400 + wavenumber))/4.0e7 + np.sin((np.pi*(400 + wavenumber))/256)
#     # synthetic_bl = np.zeros_like(wavenumber) # -((624 - wavenumber)*wavenumber*(400 + wavenumber))/4.0e7 + np.sin((np.pi*(400 + wavenumber))/256)
#     new_mp.d2baseline = synthetic_bl
#     new_mp.construct_corrections()