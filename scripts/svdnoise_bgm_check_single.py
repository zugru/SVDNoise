# coding: utf-8

import itertools
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import ticker as ticker
import seaborn as sns
from scipy import linalg
from scipy.stats import multivariate_normal as n2d

from sklearn.mixture import BayesianGaussianMixture
from sklearn.externals import joblib
from svdnoise_noisegenerator import NoiseGenerator

#---------------------------------------------------------------------
# This script trains and validates a neural network random generator.
#---------------------------------------------------------------------
color_iter = itertools.cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])

def _fmt(x, pos):
    """Helper function to format colorbar ticks"""
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

# %% Create a noise generator object
generator = NoiseGenerator()
plotdir = '../pictures/mixture_fits'

def plot_weights(bgm, threshold, sigma, width):
    fig = plt.figure(figsize=(10,10))
    plt.scatter(range(len(bgm.weights_)), np.flip(np.sort(bgm.weights_), axis = 0))
    plt.savefig('{0}/bgm_weights_thr{1}_sig{2}_w{3}.png'.format(plotdir, threshold, sigma, width))

def plot_results(bgm, threshold, sigma, width, n_samples = 10000):
    recX, labels = bgm.sample(n_samples)
    means = bgm.means_
    covariances = bgm.covariances_
    fig = plt.figure(figsize = (10,10))
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # Plot data points (do we want them?)
        if not np.any(labels == i):
            continue
        plt.scatter(recX[labels == i, 0], recX[labels == i, 1], .2, color=color)

        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle,  facecolor = color, alpha = 0.1, edgecolor=color)
        ell.set_clip_box(splot.bbox)
        #ell.set_alpha(0.1)
        splot.add_artist(ell)
    title = 'Gaussian mixture compoents, thr {0}, sigma {1}, w {2}'.format(threshold, sigma, width)
    plt.title(title)
    plt.savefig('{0}/bgm_components_thr{1}_sig{2}_w{3}.png'.format(plotdir, threshold, sigma, width))

def plot_mixture_density(bgm, threshold, sigma, width):
    weights = bgm.weights_
    means = bgm.means_
    covariances = bgm.covariances_
    generator = NoiseGenerator()
    mpdf = np.zeros_like(generator.t0s)
    X = np.zeros((generator.t0s.shape[0], generator.t0s.shape[1], 2))
    X[:,:,0] = generator.t0s
    X[:,:,1] = generator.amplitudes
    for i, (weight, mean, covar) in enumerate(zip(weights, means, covariances)):
        mpdf += weight * n2d.pdf(X, mean, covar)

    fig = plt.figure(figsize = (10,10))
    plt.contour(generator.t0s, generator.amplitudes, mpdf, 15, cmap = 'Blues')
    title = 'Gaussian mixture density, thr {0}, sigma {1}, w {2}'.format(threshold, sigma, width)
    plt.title(title)
    plt.savefig('{0}/bgm_density_thr{1}_sig{2}_w{3}.png'.format(plotdir, threshold, sigma, width))


def plot_density(sampler, threshold, sigma, width, n_random_samples = 10000):
    """ Plot P_1over pdf and some random samples over it."""
    recX, labels = sampler.sample(n_random_samples)
    rec_t0 = recX[:,0]
    rec_amplitude = recX[:,1]
    generator.generate_pdf(threshold, sigma, width)
    fig = plt.figure(figsize = (12, 12))
    # pdf and random samples go to bottom right, margins on appropriate sides
    ax1 = plt.subplot2grid((12,12),(4,0), colspan = 9, rowspan = 8)
    pdf_map = ax1.contourf(generator.t0s, generator.amplitudes, generator.pdf, 10, cmap = 'Blues')
    ax1.scatter(rec_t0, rec_amplitude, s = 0.03, c = 'y')
    ax1.set_title('Probability density and random samples'.format(n_random_samples))
    ax1.set_xlabel('t0 [ns]')
    ax1.set_ylabel('amplitude [S/N]')
    ax1c = plt.subplot2grid((12,12), (1,9), rowspan = 3, colspan = 2)
    plt.colorbar(pdf_map, cax = ax1c, format = ticker.FuncFormatter(_fmt))
    ax2 = plt.subplot2grid((12,12),(1,0), colspan = 9, rowspan = 3, sharex = ax1)
    ax2.plot(generator.t0s[:,-1], generator.pdfu)
    ax2.hist(rec_t0, bins = generator.t0s[:,0], normed = True, alpha = 0.5)
    ax2.set_title('t0 margin distribution')
    ax2.set_ylabel('P(1 over)')
    plt.setp(ax2.get_xticklabels(), visible = False)
    ax3 = plt.subplot2grid((12,12),(4,9), rowspan = 8, colspan = 3, sharey = ax1)
    ax3.plot(generator.pdfv, generator.amplitudes[-1,:])
    ax3.hist(rec_amplitude, bins = generator.amplitudes[0,:], normed = True, orientation = 'horizontal', alpha = 0.5)
    ax3.set_title('Amplitude margin distribution')
    ax3.set_xlabel('P(1 over)')
    plt.setp(ax3.get_yticklabels(), visible = False)
    ax4 = plt.subplot2grid((12,12),(0,0), colspan = 9)
    ax4.text(0.5, 1.0, 'Exact P(one over) distribution and {0} random samples \nthreshold : {1}, sigma : {2}, width : {3}'.format(n_random_samples, threshold, sigma, width), horizontalalignment = 'center', verticalalignment = 'top', fontsize = 18)
    ax4.set_axis_off()
    plt.tight_layout()
    plt.savefig('{0}/rng_test_thr{1}_sig{2}_w{3}.png'.format(plotdir, threshold, sigma, width))

if __name__ == '__main__':
    # %% Load the model
    bgm_fit = joblib.load('../data/bgm_fit.sav')
    # %% Make the plot
    threshold = 4.0
    sigma = 0.25
    width = 200
    print('Plotting maps...')
    plot_weights(bgm_fit, threshold, sigma, width)
    plot_results(bgm_fit, threshold, sigma, width)
    plot_mixture_density(bgm_fit, threshold, sigma, width)
    plot_density(bgm_fit, threshold, sigma, width, 10000)

