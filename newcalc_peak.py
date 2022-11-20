import numpy as np

from spectralfeature import SpectralFeature, voigt_asym
from enum import Enum

class Peak:
    def __init__(self, spec_array):
        if spec_array:
            self.spec_array = spec_array


class CalcCustomPeak:
    def __init__(self, featurePeak: Peak):
        self.featurePeak = featurePeak


class CalcSpecPeak:
    def __init__(self, featurePeak: SpectralFeature, wn=np.linspace(0, 1, 129)):
        self.featurePeak = featurePeak
        self.wn = wn

    @property
    def peak_area(self):
        peak_area = (1 - self.featurePeak.specs_array[3]) * self.featurePeak.specs_array[4] * (
                    1 + 0.69 * self.featurePeak.specs_array[2] ** 2 + 1.35 * self.featurePeak.specs_array[2] ** 4) + self.featurePeak.specs_array[3] * \
                    self.featurePeak.specs_array[4] * (1 + 0.67 * self.featurePeak.specs_array[2] ** 2 + 3.43 * self.featurePeak.specs_array[2] ** 4)
        return peak_area

    @property
    def peak_height(self):
        amplitudes_L = self.featurePeak.specs_array[4] * 2 / (np.pi * self.featurePeak.specs_array[1])
        amplitudes_G = self.featurePeak.specs_array[4] * (4 * np.log(2) / np.pi) ** 0.5 / self.featurePeak.specs_array[1]
        peak_height = self.featurePeak.specs_array[3] * amplitudes_G + (1 - self.featurePeak.specs_array[3]) * amplitudes_L
        return peak_height

    @peak_height.setter
    def peak_height(self, newheight):
        self.featurePeak.specs_array[4] = newheight / (
                self.featurePeak.specs_array[3] * (4 * np.log(2) / np.pi) ** 0.5 / self.featurePeak.specs_array[1] + (
                    1 - self.featurePeak.specs_array[3]) * 2 / (np.pi * self.featurePeak.specs_array[1])
        )

    @property
    def fwhm_asym(self):
        """ real fwhm of an asymmetric peak"""
        fwhm_asym = self.featurePeak.fwhm * (1 + 0.4 * self.featurePeak.asymmetry ** 2 + 1.35 * self.featurePeak.asymmetry ** 4)
        return fwhm_asym

    @property
    def curve(self):
        """ Asymmetric pseudo-Voigt funtion as defined in Analyst: 10.1039/C8AN00710A
        """
        curve = self.featurePeak.specs_array[4] * voigt_asym(self.wn - self.featurePeak.specs_array[0], self.featurePeak.specs_array[1],
                                                 self.featurePeak.specs_array[2], self.featurePeak.specs_array[3])
        return curve

    @property
    def curve_with_BL(self):
        curve_with_BL = self.curve + self.featurePeak.specs_array[5] * (self.wn - self.featurePeak.specs_array[0]) + self.featurePeak.specs_array[6]
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
                   header=the_header)


class GaussianPeak(SpectralFeature):
    def __init__(self, specs_array):
        super().__init__(specs_array)

    @property
    def Gaussian_share(self):
        return 1.0

    @Gaussian_share.setter
    def Gaussian_share (self, Gaussian_share) :
        return


class LorentzianPeak(SpectralFeature):
    def __init__(self, specs_array):
        super().__init__(specs_array)

    @property
    def Gaussian_share(self):
        return 0.0

    @Gaussian_share.setter
    def Gaussian_share (self, Gaussian_share) :
        return

class VoigtPeak(SpectralFeature):
    def __init__(self, specs_array):
        super().__init__(specs_array)

class AsymVoigtPeak(SpectralFeature):
    def __init__(self, specs_array):
        super().__init__(specs_array)


class PeakType(Enum):
    GAUSSIAN = "Gaussian"
    LORENTZIAN = "Lorentzian"
    PSEUDO_VOIGT = "Pseudo_Voigt"
    ASYM_PSEUDO_VOIGT = "Asym_Pseudo_Voigt"
    UNDEFINED = "Undefined"

    @staticmethod
    def get_peak_type(value):
        A = {
            "Gaussian": PeakType.GAUSSIAN,
            "Lorentzian" : PeakType.LORENTZIAN,
            "Pseudo_Voigt" : PeakType.PSEUDO_VOIGT,
            "Asym_Pseudo_Voigt": PeakType.ASYM_PSEUDO_VOIGT,
            "Undefined" : PeakType.UNDEFINED
        }
        return A[value]



class PeakFactory:
    @classmethod
    def get_peak(self, starting_point) -> Peak:
        if starting_point[0] == PeakType.GAUSSIAN.value:
            return GaussianPeak(starting_point)
        elif starting_point[1] == PeakType.LORENTZIAN.value:
            return LorentzianPeak(starting_point)
        elif starting_point[1] == PeakType.PSEUDO_VOIGT.value:
            return VoigtPeak(starting_point)
        elif starting_point[1] == PeakType.ASYM_PSEUDO_VOIGT.value:
            return AsymVoigtPeak(starting_point)
        else:
            return Peak(starting_point)




