# coding: utf-8
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

plotdir = '../pictures/linearfits'

def add_predictions(floorplan, fits):
    """Plots model predictions and data on the grid of a regular floorplan.
    fits is a dict of linear models for individual thresholds.
    Plots data for all thresholds on a single canvas."""
    # Find grid dimensions
    thresholds = floorplan['threshold'].unique()
    n_thresholds = thresholds.shape[0]
    sigmas = floorplan['sigma'].unique()
    n_sigmas = sigmas.shape[0]
    widths = floorplan['width'].unique()
    n_widths = widths.shape[0]
    # Calculate predicitons and residuals
    for threshold, group in floorplan.groupby('threshold'):
        floorplan.loc[group.index, 'prediction'] = fits[threshold].predict(group[['sigma','width']])
    floorplan['residual'] = floorplan['prediction'] - floorplan['lognorm']
    floorplan['threshold_squared'] = floorplan['threshold']**2
    return floorplan

def plot_predictions(floorplan):
    fig = plt.Figure()
    grid = sns.FacetGrid(floorplan, col = 'sigma', col_wrap = 3, hue = 'width', sharex = True, sharey = True)
    grid.map(plt.scatter, 'threshold_squared', 'lognorm')
    grid.map(plt.plot, 'threshold_squared', 'prediction')
    grid.add_legend()
    grid.savefig('{0}/normfit_predictions.png'.format(plotdir))

def plot_residuals(floorplan):
    fig = plt.Figure()
    grid = sns.FacetGrid(floorplan, col = 'sigma', col_wrap = 3, hue = 'width', sharex = True, sharey = True)
    grid.map(plt.scatter, 'threshold_squared', 'residual')
    grid.add_legend()
    grid.savefig('{0}/ormfit_residuals.png'.format(plotdir))

if __name__ == '__main__':
    # Load floorplan
    # %matplotlib inline
    print('Loading floorplan...')
    floorplan = pd.read_json('../data/norms_regular_floorplan.json')
    thresholds = floorplan['threshold'].unique()
    # Load fits
    fits = {}
    print('Loading fits...')
    for threshold in thresholds:
        lin_fit = joblib.load('../data/normfit_thr{0}.sav'.format(int(threshold)))
        fits[threshold] = lin_fit
    # Add predicitons to floorplan
    print('Adding predictions and residuals...')
    floorplan = add_predictions(floorplan, fits)
    print('Plotting predictions...')
    plot_predictions(floorplan)
    print('Plotting residuals...')
    plot_residuals(floorplan)
# %%
