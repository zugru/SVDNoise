# coding: utf-8

import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import skewnorm, beta, gamma

#---------------------------------------------------------------------
# Parameterize noise distributions
#
# We parameterize noise probabilities in the (signal, t0) space
# by threshold, waveform noise, and waveform width.
# We calculate exact probabiities of an APV 6-waveform exceeding
# a ZS cut on a (signal, t0) grid for all combinations of threshold,
# waveform noise, and waveform width. Signals are S/N amplitudes.
#
# Parameterization is based on skew gaussians, which are supported both by Python (scipy) and C++ (Boost::math), so we can sample from the distributions easily.
#---------------------------------------------------------------------

#--------------------------------------------------------------------------
# Functions related to APV waveform signals
#--------------------------------------------------------------------------

apv_dt = 31.44 # ns

bp_default_tau = 250  # ns

def betaprime_wave(t, tau=bp_default_tau):
    ''' Beta-prime waveform
    betaprime_wave(t, tau) = 149.012 * (t/tau)**2 * (1 + t/tau)**10 if t > 0 else 0
    t - numpy vector of times
    tau - (scalar) width of the distribution
    return - numpy vector of betaprime_wave values of betaprime_wave at times t
    '''
    z = np.clip(t / tau, 0, 1000)
    return 149.012 * z**2 / (1 + z)**10

def generate_sample(amplitude, t0, width, n_samples = 6):
    """Returns a size-6 np array with waveform samples.
    No randomness involved!"""
    if n_samples == 6:
        times = np.linspace(-apv_dt-t0, 4*apv_dt-t0, 6, endpoint = True)
    elif n_samples == 3:
        times = np.linspace(-t0, 2*apv_dt-t0, 3, endpoint = True)
    elif n_samples == 1:
        times = np.asarray([-t0])
    else:
        print("Unsupported number of APV samples in generate_sample!")
        times = np.zeros(1)
    return amplitude * betaprime_wave(times, width)


#--------------------------------------------------------------------------
# Functions to parameterize marginal distributions.
#--------------------------------------------------------------------------

def single_peak(x, mean, sigma, alpha):
    """This is a single skew-gaussian. Note the order of parameters in skewnorm.pdf!
    """
    return skewnorm.pdf(x, alpha, mean, sigma)

def single_peak_cdf(x, mean, sigma, alpha):
    """CDF for the skew-gaussian.
    Additional safety to filter out large arguments."""
    z = np.clip((x-mean)/sigma, -10, 10)
    return skewnorm.cdf(z, alpha)

def peak_comb(x, mean, sigma, alpha, dt, n_samples = 6):
    """Combination of 6 or 3 skew Gaussians dt apart, with
       common width and shape parameters.
    """
    res = np.zeros_like(x)
    for i in range(n_samples):
        res += skewnorm.pdf(x, alpha, mean + i*dt, sigma)
    return 1.0/n_samples * res

def peak_comb_cdf(x, mean, sigma, alpha, dt, n_samples = 6):
    """Cdf for the combination of skew Gaussians dt apart, with
       common width and shape parameters. Additional safety for large arguments added.
    """
    res = np.zeros_like(x)
    for i in range(n_samples):
        z = np.clip((x - mean - i*dt)/sigma, -10, 10)
        res += skewnorm.cdf(z, alpha)
    return 1.0/n_samples * res

#--------------------------------------------------------------------------
# Functions for fitting the partial copula
#--------------------------------------------------------------------------
def fu_fit(fu, v):
    """Function to fit v-distributions in partial copula by beta(a,1) cdfs."""
    popt, pcov = curve_fit(
        lambda v, a : beta.cdf(v, a, 1),
        v, fu,
        p0 = (1,)
    )
    chi2 = np.sum((beta.cdf(v, *popt, 1) - fu)**2 / (len(v)-2))
    res = np.zeros(len(popt)+1)
    res[0:1] = popt
    res[1] = math.sqrt(chi2)
    return res

def left_edge(x, scale):
    """Edge function to account for left edge deviation."""
    return gamma.pdf(np.clip(x,0,1), 1, loc = 0, scale = min(0.99, max(0.01, scale)))

def right_edge(x, scale):
    """Edge function to account for right edge deviation."""
    return gamma.pdf(np.clip(1-x,0,1), 1, loc = 0, scale = min(0.99, max(0.01, scale)))

def beta_periodic(u, shift, n, peaks):
    """This is the periodic component for u in <0,1>, with peaks derived from margin fit, shift and first shape parameter for the beta kernel. """
    res = np.zeros_like(u)
    safe_n = min(50, max(1.1,n))
    safe_shift = min(0.05, max(-0.05, shift))
    size = np.mean(np.diff(peaks))
    for m in peaks:
        z = (u - m + safe_shift)/size
        indices = np.abs(z) < 0.5
        res[indices] += 0.5 * (beta.pdf(0.5 + z[indices], safe_n, safe_n) + beta.pdf(0.5 + z[indices], 5, 5))
    return res

def beta_full(u, coefs, peaks):
    return coefs[0] + coefs[1] * left_edge(u, coefs[4]) + coefs[2] * right_edge(u, coefs[5]) + coefs[3] * beta_periodic(u, coefs[6], coefs[7], peaks)

#--------------------------------------------------------------------------
# Discrepancy measures
#--------------------------------------------------------------------------
def kolmogorov_distance(pdfx, pdfy):
    """kolmogorov distance is the maximum distance between two
       probability distributions. This calculates the KD of two
       pdfs in the form of numpy arrays."""
    cdfx = np.cumsum(pdfx)/np.sum(pdfx)
    cdfy = np.cumsum(pdfy)/np.sum(pdfy)
    return np.max(np.abs(cdfx - cdfy))

def q_distance(pdfx, pdfy, index):
    """This is like Kolmogorov distance, just measured
       horizontally. It gives RMS deviation of random samples
       from the two distributions."""
    cdfx = np.cumsum(pdfx)/np.sum(pdfx)
    cdfy = np.cumsum(pdfy)/np.sum(pdfy)
    d = np.linspace(0.0, 1.0, 20, endpoint = False) + 1/40
    xd = np.interp(d, cdfx, index)
    yd = np.interp(d, cdfy, index)
    return np.sqrt(simps((xd - yd)**2, d))

#--------------------------------------------------------------------------
# ### Study range
#--------------------------------------------------------------------------

threshold_range = (3.0, 7.0)
sigma_range = (0.1, 1.0)
width_range = (200, 360)

base_thresholds = np.asarray([3.0, 4.0, 5.0, 6.0, 7.0])
base_sigmas = np.asarray([0.1, 0.25, 0.5, 1.0])
base_widths = np.asarray([200.0, 250.0, 300.0, 350.0])

def generate_regular_floorplan(thresholds = base_thresholds, sigmas = base_sigmas, widths = base_widths):
    '''Generate a floorplan in terms of threshold, sigma, and waveform width.
       Input are lists of values. Returns a dataframe with all combinations
       in 3 columns.'''
    n_table = len(thresholds) * len(sigmas) * len(widths)
    floorplan = pd.DataFrame(
        index = pd.MultiIndex.from_product([thresholds, sigmas, widths],
                                       names = ['threshold', 'sigma', 'width'])
    )
    return floorplan.reset_index()

def generate_random_floorplan(n = 200, thresholds = threshold_range, sigmas = sigma_range, widths = width_range):
    '''Generate a floorplan in terms of threshold, sigma, and waveform width.
       Input are ranges of values. Returns a dataframe with random combinations
       in 3 columns.'''
    floorplan = pd.DataFrame({
        'threshold' : np.random.uniform(*thresholds, n),
        'sigma' : np.random.uniform(*sigmas, n),
        'width' : np.random.uniform(*widths, n)
    })
    return floorplan

def generate_mixed_floorplan(n = 200, thresholds = base_thresholds, sigmas = sigma_range, widths = width_range):
    '''Generate a floorplan with fixed thresholds and random sigmas and waveform width.
       Inputs are array of thresholds and ranges of sigmas and widths. Returns a dataframe 
       with thresholds sampled from the fixed values and random combinations of sigmas and
       widths.'''
    floorplan = pd.DataFrame({
        'threshold' : np.random.choice(np.array(thresholds), size = n, replace = True),
        'sigma' : np.random.uniform(*sigmas, n),
        'width' : np.random.uniform(*widths, n)
    })
    return floorplan
