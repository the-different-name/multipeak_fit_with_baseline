
## Multipeak fit with simultaneous baseline estimation

Primary intended use is Raman spectroscopy. Baseline and peaks are found by Tikhonov regularization.
The regularization parameter can be automatically estimated from the 4th derivative of the longpass-filtered spectrum.

The algorithm is described in the following paper:
**Separation of Spectral Lines from a Broadband Background and Noise Filtering by Modified Tikhonov Regularization**
IA Larkin, AV Vagov, VI Korepanov - Optoelectronics, Instrumentation and Data Processing, 2023
[doi:10.3103/S8756699023060080](https://dx.doi.org/10.3103/S8756699023060080)



## Current paradigm:
We have a spectrum, which has three contributions:
* signal (relatively sharp peaks);
* noise (white noise);
* baseline (relatively broad features.
We intend to find the baseline and the peak parameters. We don't know the shape of the peaks, therefore they are allowed to be flexible (asymmetric pseudo-Voigt as in https://doi.org/10.1039/C8AN00710A ).



## Structure:
```fit_multipeak.py``` -- this is the main file. Just run it for the illustration.

```expspec.py``` -- spectral class format, some supplementary function - probably you don't want to look at it.

```spectralfeature.py``` -- format of the 'multi-peak'. Possible peak shapes are defined here, see "PossibleFunctions".

```spectroutines.py``` -- supplementary procedures.

```generate_random_spectrum.py``` -- generate multipeak with random parameters. With noise.


Typically, if you have your test spectrum as *numpy-readable* *x-y* file (a text file with two columns separated with tabs or spaces), you could use the following example:
```python
from fit_multipeak import * # load the script
current_spectrum = np.genfromtxt('my_spectrum.txt') # read file to numpy format
testspec = ExpSpec(current_spectrum[:,0], current_spectrum[:,1]) # convert the spectrum to an *object* of a specific format.
dat_result = multipeak_fit_with_BL(testspec, saveresults = True) # fit it! The starting point will be generated automatically from *find_da_peaks* function.
```

The example above is a basic usage. You can set the starting point for the fit, specify the fitting range, control the display options etc.
There is a test spectrum and an example file for the starting point. You can try the following:
```python
from fit_multipeak import * # load the script
s = read_startingpoint_from_txt('test_spectrum_startingpoint.txt')
current_spectrum = np.genfromtxt('test_data_experimental_spectrum.txt') # read file to numpy format
testspec = ExpSpec(current_spectrum[:,0], current_spectrum[:,1]) # convert the spectrum to an *object* of a specific format.
dat_result = multipeak_fit_with_BL(testspec,
                         fitrange=(500, 3700),
                         startingpoint='test_spectrum_startingpoint.txt',
                         als_lambda = 6e7,
                         saveresults=True, display=2)
```

If you need to display the peak parameters, just type:
```python
dat_result.params
```

If you need to display the peak parameters, just type:
```python
dat_result.params
```

To save peak parameters to some file:
```python
dat_result.save_params_to_txt(filename)
```