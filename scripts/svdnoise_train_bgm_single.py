# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.externals import joblib

#---------------------------------------------------------------------
# This script trains and validates a gaussian mixture estimator for the
# strip noise generator.
#---------------------------------------------------------------------

# Names of data frame columns.
floorplan_cols = ['threshold', 'sigma', 'width']
input_cols = ['u', 'v']
output_cols = ['t0', 'amplitude']

def fit_mixture(input_frame):
    """This fits a tree regressor estimator to data. Returns trained model."""
    nrows = len(input_frame)
    X  = np.asarray(input_frame[output_cols])
    bgm = BayesianGaussianMixture(n_components = 30, tol = 0.1, n_init = 1, covariance_type = 'full', max_iter = 1000, verbose = 2, verbose_interval = 50).fit(X)
    joblib.dump(bgm, '../data/bgm_fit.sav')
    return (bgm)

if __name__ == '__main__':
    # %% Load the source file
    tree_data = pd.read_json('../data/single_training_data_100000.json')
    # Get the model
    bgm = fit_mixture(tree_data)
    print(np.sort(bgm.weights_))
