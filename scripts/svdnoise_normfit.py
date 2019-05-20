# coding: utf-8

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

#---------------------------------------------------------------------
# This file contains fucntions related to linear model fitting of
# noise pdf margin parameters.
# Even if we train the network, we will still have to make a linear model for norm. This will be relatively easy and the linear model works nicely for lognorm.
#---------------------------------------------------------------------

# Names of data frame columns.
floorplan_cols = ['threshold', 'sigma', 'width']
norm_cols = ['norm']

def fit_norms(input_frame, degree = 2):
    """This fits a linear model to pdf norm.
       For norms, the dominant part of the model is log(norm) ~ threshold**2, different than for other parameters.
       Returns the fitted model object. Input frame is unchanged."""
    X  = np.asarray(input_frame[floorplan_cols])
    # We regress against threshold squared - the dependence should be approximately linear
    X[:,0] = X[:,0] ** 2
    y = np.asarray(input_frame[norm_cols])
    # And we regress log(norm)
    y = np.log10(y)
    model = make_pipeline(
        PolynomialFeatures(degree),
        #StandardScaler(),
        LinearRegression(normalize = True)
    )
    model.fit(X, y)
    return model

def norm_fit_frame(input_frame, model):
    '''This puts predictions and results of a linear model fit to margin
       norms into a data frame. The input_frame is expected to have only
       input columns: threshold, sigma, width.
       This allows to predict data for a different floorplan.'''
    # Add the column, in case the input doesn't have it.
    if not 'thresold2' in input_frame.columns :
        input_frame['threshold2'] = input_frame['threshold']**2
    X  = np.asarray(input_frame[['threshold2', 'sigma', 'width']])
    y_fit = model.predict(X)
    for i, varname in enumerate(['lognorm']):
        input_frame[varname + '_fit'] = y_fit[:,i]
        if varname in input_frame.columns: # Add residuals, if original values present
            input_frame[varname + '_res'] = input_frame[varname] - input_frame[varname + '_fit']
    return input_frame

def plot_norm_fit(plot_frame):
    '''This plots results of a linear model fit to margin norms.
       Only works for regular floorplans!
       Only things actually present in plot_frame are plotted.'''
    sns.set_context('notebook', font_scale = 1.2)
    sns.set(style = 'whitegrid', palette = 'muted')
    for i, varname in enumerate(['lognorm']):
        have_val = varname in plot_frame.columns
        if not have_val: # check if there is _amp column and if yes, create _logamp
            predname = varname.replace('log','')
            if predname in plot_frame.columns:
                plot_frame[varname] = np.log10(plot_frame[predname])
                have_val = True
        fitname = varname + '_fit'
        have_fit = fitname in plot_frame.columns
        resname = varname + '_res'
        have_res = resname in plot_frame.columns
        if not have_res : # try to calculate
            if have_val and have_fit:
                plot_frame[resname] = plot_frame[varname] - plot_frame[fitname]
                have_res = True
        if have_val or have_fit:
            fig = plt.Figure()
            g = sns.FacetGrid(data = plot_frame, col = 'sigma', hue = 'width')
            if have_val:
                g.map(plt.scatter, 'threshold2', varname)
            if have_fit:
                g.map(plt.plot, 'threshold2', fitname)
            g.add_legend()
            plt.savefig('../pictures/linearfits/{0}_fit_tsw.png'.format(varname))
            fig = plt.Figure()
            g = sns.FacetGrid(data = plot_frame, col = 'width', hue = 'sigma')
            if have_val:
                g.map(plt.scatter, 'threshold2', varname)
            if have_fit:
                g.map(plt.plot, 'threshold2', fitname)
            g.add_legend()
            plt.savefig('../pictures/linearfits/{0}_fit_tws.png'.format(varname))
        if have_res:
            fig = plt.Figure()
            g = sns.FacetGrid(data = plot_frame, col = 'sigma', hue = 'width')
            g.map(plt.scatter, 'threshold2', resname)
            plt.savefig('../pictures/linearfits/{0}_res.png'.format(varname))
            fig = plt.figure()
            plt.scatter(plot_frame[varname], plot_frame[resname])
            plt.ylabel(resname)
            plt.xlabel(varname)
            plt.savefig('../pictures/linearfits/{0}_resplot.png'.format(varname))
            plt.close('all')
    return
