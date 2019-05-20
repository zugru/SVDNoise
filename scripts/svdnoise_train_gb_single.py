# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

#---------------------------------------------------------------------
# This script trains and validates a regression tree estimator for the
# strip noise generator.
#---------------------------------------------------------------------

# Functions to transform between quantiles and z:
def q_to_z(X, ncols = 2):
    """Trnasforms quantile (0 to 1) data to corresponding standard normal deviates. Only transforms the last ncols of X."""
    Xtrans = X.copy()
    Xtrans[:,-ncols:] = norm.ppf(X[:,-ncols:])
    return Xtrans

def z_to_q(X, ncols = 2):
    """Transforms normal deviates z back to quantiles of standard normal distribution (0 to 1). Inverse to u_to_z. Only transforms the last ncols of X."""
    Xtrans = X.copy()
    Xtrans[:,-ncols:] = norm.cdf(X[:,-ncols:])
    return Xtrans

# Names of data frame columns.
floorplan_cols = ['threshold', 'sigma', 'width']
input_cols = ['u', 'v']
output_cols = ['t0', 'amplitude']

def fit_tree(input_frame, test_frac = 0.1):
    """This fits a tree regressor estimator to data. Returns trained model."""
    nrows = len(input_frame)
    n_test = int(test_frac * nrows)
    X  = np.asarray(input_frame[input_cols])
    X_train = X[:-n_test,:]
    X_test = X[-n_test:,:]
    Y  = np.asarray(input_frame[output_cols])
    Y_train = Y[:-n_test,:]
    Y_test = Y[-n_test:,:]
    params = {
            'max_leaf_nodes' : 1000,
            'loss' : 'ls'
    }
    time_fit = make_pipeline(
        FunctionTransformer(q_to_z, z_to_q, kw_args = {'ncols' : 1}, inv_kw_args = {'ncols' : 1}),
        StandardScaler(),
        GradientBoostingRegressor(**params)
    )
    params2 = {
            'max_leaf_nodes' : 1000,
            'loss' : 'ls'
    }
    amplitude_fit = make_pipeline(
        FunctionTransformer(q_to_z, z_to_q, kw_args = {'ncols' : 2}, inv_kw_args = {'ncols' : 2}),
        StandardScaler(),
        GradientBoostingRegressor(**params2)
    )
    time_fit.fit(X_train, Y_train[:,0])
    joblib.dump(time_fit, '../data/gb_single_time_fit.sav')
    amplitude_fit.fit(X_train, Y_train[:,1])
    joblib.dump(amplitude_fit, '../data/gb_single_amplitude_fit.sav')
    print('Gradient boost fit scores: time {0}, amplitude {1}'.format(time_fit.score(X_test, Y_test[:,0]), amplitude_fit.score(X_test, Y_test[:,1])))
    return (time_fit, amplitude_fit)

if __name__ == '__main__':
    # %% Load the source file
    tree_data = pd.read_json('../data/single_training_data_100000.json')
    # Get the model
    (time_fit, amplitude_fit) = fit_tree(tree_data)

