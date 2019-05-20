# coding: utf-8

import numpy as np
import pandas as pd
from timeit import default_timer as timer

#from matplotlib import pyplot as plt
#import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

#---------------------------------------------------------------------
# This script trains and validates a neural network random generator.
#---------------------------------------------------------------------

# %% Functions to transform between quantiles and z:
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


def fit_nn(input_frame, test_frac = 0.01):
    """This fits a neural network to the data. Returns trained model."""
    nrows = len(input_frame)
    n_test = int(test_frac * nrows)
    X  = np.asarray(input_frame[floorplan_cols + input_cols])
    X_train = X[:-n_test,:]
    X_test = X[-n_test:,:]
    y = np.asarray(input_frame[output_cols])
    y_train = y[:-n_test,:]
    y_test = y[-n_test:,:]
    params1 = {
        'hidden_layer_sizes' : (100,100,100,100),
        'activation' : 'relu',
        'batch_size' : 1000,
        'alpha' : 1.0e-2,
        'tol' : 1.0e-8,
        'verbose' : True
        }
    print(params1)
    params2 = {
        'hidden_layer_sizes' : (100,100,100,100),
        'activation' : 'relu',
        'batch_size' : 1000,
        'alpha' : 1.0e-2,
        'tol' : 1.0e-8,
        'verbose' : True
        }
    print(params2)
    model_t0 = make_pipeline(
        FunctionTransformer(q_to_z, z_to_q, kw_args = {'ncols' : 1}, inv_kw_args = {'ncols' : 1}),
        StandardScaler(),
        MLPRegressor(**params1)
    )
    model_amplitude = make_pipeline(
        FunctionTransformer(q_to_z, z_to_q, kw_args = {'ncols' : 2}, inv_kw_args = {'ncols' : 2}),
        StandardScaler(),
        MLPRegressor(**params2)
    )
    model_t0.fit(X_train[:,:-1], y_train[:,0])
    model_amplitude.fit(X_train, y_train[:,1])
    joblib.dump(model_t0, '../data/nn_model_t0.sav')
    joblib.dump(model_amplitude, '../data/nn_model_amplitude.sav')
    print('t0 model score: {0}'.format(model_t0.score(X_test[:,:-1], y_test[:,0])))
    print('amplitude model score: {0}'.format(model_amplitude.score(X_test, y_test[:,1])))
    y_rep = np.empty_like(y_test)
    y_rep[:,0] = model_t0.predict(X_test[:,:-1])
    y_rep[:,1] = model_amplitude.predict(X_test)
    print("Chi2: {0}".format(np.sqrt(np.sum((y_rep - y_test)**2, axis = 0)/y_rep.shape[0])))
    return (model_t0, model_amplitude)

if __name__=='__main__':
    start = timer()
    # %% Load the source file
    nn_data = pd.read_json('../data/training_data_2000000.json')
    nn_data.head()
    model_t0, model_amplitude = fit_nn(nn_data)
    end = timer()
    print('Elapsed time: {0}'.format(end - start))
