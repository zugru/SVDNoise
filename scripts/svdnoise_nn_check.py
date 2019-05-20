# coding: utf-8

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import ticker as ticker
import seaborn as sns

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from svdnoise_noisegenerator import NoiseGenerator
# We have to import the transformation functions from the training script.
# It would be better to have them in a separate module.
from svdnoise_train_nn import q_to_z, z_to_q

#---------------------------------------------------------------------
# This script trains and validates a neural network random generator.
#---------------------------------------------------------------------

def _fmt(x, pos):
    """Helper function to format colorbar ticks"""
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

# %% Create a noise generator object
generator = NoiseGenerator()
plotdir = '../pictures/nn_fits'

def plot_diff_map(threshold, sigma, width):
    u, v = np.mgrid[0.01:0.99:0.02, 0.01:0.99:0.02]
    len_u = len(u.flatten())
    recX = np.empty((len_u, 5))
    recX[:,0] = threshold
    recX[:,1] = sigma
    recX[:,2] = width
    recX[:,3] = u.flatten()
    recX[:,4] = v.flatten()
    rec_t0 = model_t0.predict(recX[:,:-1]).reshape(u.shape)
    rec_amplitude = model_amplitude.predict(recX).reshape(v.shape)
    generator.generate_pdf(threshold, sigma, width)
    original_samples = generator.random_transform(u.flatten(), v.flatten())
    orig_t0 = original_samples['t0'].reshape(u.shape)
    orig_amplitude = original_samples['amplitude'].reshape(v.shape)
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax1.scatter(orig_t0.flatten(), rec_t0.flatten())
    ax2 = plt.subplot2grid((2,1),(1,0))
    ax2.scatter(orig_amplitude.flatten(), rec_amplitude.flatten())
    fig = plt.figure(figsize = (12,12))
    lw = 0.5
    for i in range(orig_t0.shape[0]):
        plt.plot(orig_t0[:,i], orig_amplitude[:,i], c = 'b', linewidth = lw)
        plt.plot(rec_t0[:,i], rec_amplitude[:,i], c = 'r', linewidth = lw)
    for j in range(orig_t0.shape[1]):
        plt.plot(orig_t0[j,:], orig_amplitude[j,:], c = 'b', linewidth = lw)
        plt.plot(rec_t0[j,:], rec_amplitude[j,:], c = 'r', linewidth = lw)
    plt.savefig('{0}/uv_map_thr{1}_sig{2}_w{3}.png'.format(plotdir, threshold, sigma, width))

def plot_density(threshold, sigma, width, n_random_samples = 10000):
    """ Plot P_1over pdf and some random samples over it."""
    u = np.random.uniform(0,1,n_random_samples)
    v = np.random.uniform(0,1,n_random_samples)
    recX = np.empty((n_random_samples, 5))
    recX[:,0] = threshold
    recX[:,1] = sigma
    recX[:,2] = width
    recX[:,3] = u
    recX[:,4] = v
    rec_t0 = model_t0.predict(recX[:,:-1])
    rec_amplitude = model_amplitude.predict(recX)
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
    ax2.hist(rec_t0, bins = generator.t0s[:,0], density = True, alpha = 0.5)
    ax2.set_title('t0 margin distribution')
    ax2.set_ylabel('P(1 over)')
    plt.setp(ax2.get_xticklabels(), visible = False)
    ax3 = plt.subplot2grid((12,12),(4,9), rowspan = 8, colspan = 3, sharey = ax1)
    ax3.plot(generator.pdfv, generator.amplitudes[-1,:])
    ax3.hist(rec_amplitude, bins = generator.amplitudes[0,:], density = True, orientation = 'horizontal', alpha = 0.5)
    ax3.set_title('Amplitude margin distribution')
    ax3.set_xlabel('P(1 over)')
    plt.setp(ax3.get_yticklabels(), visible = False)
    ax4 = plt.subplot2grid((12,12),(0,0), colspan = 9)
    ax4.text(0.5, 1.0, 'Exact P(one over) distribution and {0} random samples \nthreshold : {1}, sigma : {2}, width : {3}'.format(n_random_samples, threshold, sigma, width), horizontalalignment = 'center', verticalalignment = 'top', fontsize = 18)
    ax4.set_axis_off()
    plt.tight_layout()
    plt.savefig('{0}/rng_test_thr{1}_sig{2}_w{3}.png'.format(plotdir, threshold, sigma, width))

if __name__=='__main__':
    # %% Load the model
    model_t0 = joblib.load('../data/nn_model_t0.sav')
    model_amplitude = joblib.load('../data/nn_model_amplitude.sav')
    # %% Make the plot
    threshold = 7.0
    sigma = 0.25
    width = 300
    print('Plotting maps...')
    plot_diff_map(threshold, sigma, width)
    plot_density(threshold, sigma, width,100000)
