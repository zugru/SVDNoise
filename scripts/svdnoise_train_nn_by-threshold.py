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

# We separate amplitude and time training, so they can be run and optimized independently.j

def fit_nn_t0(threshold, input_frame, nn_params):
    """This fits a neural network to t0 data. Returns training result."""
    start = timer()
    test_frac = 0.1
    # Names of data frame columns.
    nrows = len(input_frame)
    n_test = int(test_frac * nrows)
    X  = np.asarray(input_frame[['sigma', 'width', 'u']])
    X_train = X[:-n_test,:]
    X_test = X[-n_test:,:]
    y = np.asarray(input_frame[['t0']])
    y_train = y[:-n_test,:]
    y_test = y[-n_test:,:]
    model_t0 = make_pipeline(
        FunctionTransformer(q_to_z, z_to_q, kw_args = {'ncols' : 1}, inv_kw_args = {'ncols' : 1}),
        StandardScaler(),
        MLPRegressor(**nn_params)
    )
    model_t0.fit(X_train, y_train.ravel())
    score_t0 = model_t0.score(X_test, y_test)
    y_rep = model_t0.predict(X_test)
    chi2_t0 = np.sqrt(np.sum((y_rep - y_test.ravel())**2)/len(y_rep))
    end = timer()

    return {
            'threshold' : threshold,
            'value': 't0', 
            'model': model_t0, 
            'score': score_t0, 
            'chi2': chi2_t0, 
            'time' : end - start
            }

def fit_nn_amplitude(threshold, input_frame, nn_params):
    """This fits a neural network to amplitude data. Returns training result."""
    start = timer()
    test_frac = 0.1
    # Names of data frame columns.
    nrows = len(input_frame)
    n_test = int(test_frac * nrows)
    X  = np.asarray(input_frame[['sigma', 'width', 'u', 'v']])
    X_train = X[:-n_test,:]
    X_test = X[-n_test:,:]
    y = np.asarray(input_frame[['amplitude']])
    y_train = y[:-n_test,:]
    y_test = y[-n_test:,:]
    model_amplitude = make_pipeline(
        FunctionTransformer(q_to_z, z_to_q, kw_args = {'ncols' : 2}, inv_kw_args = {'ncols' : 2}),
        StandardScaler(),
        MLPRegressor(**nn_params)
    )
    model_amplitude.fit(X_train, y_train.ravel())
    score_amplitude = model_amplitude.score(X_test, y_test)
    y_rep = model_amplitude.predict(X_test)
    chi2_amplitude = np.sqrt(np.sum((y_rep - y_test.ravel())**2)/len(y_rep))
    end = timer()

    return {
            'threshold': threshold, 
            'value' : 'amplitude', 
            'model': model_amplitude, 
            'score': score_amplitude, 
            'chi2': chi2_amplitude, 
            'time' : end - start
            }


if __name__=='__main__':
    start = timer()
    # %% Load the source file
    nn_data = pd.read_json('../data/training-data_fixed-thr_n100000.json')
    nn_data.head()
    # Get thresholds
    thresholds = nn_data['threshold'].unique() 
    print('Threshold set> {0}'.format(thresholds))
    # Output dataframe
    results = np.ndarray((2*thresholds.shape[0], 5))
    t0_parameters = {
        'hidden_layer_sizes' : (100,100,100,100),
        'activation' : 'relu',
        'batch_size' : 1000,
        'alpha' : 1.0e-2,
        'tol' : 1.0e-8,
        'verbose' : False
        }
    print('t0 fit parameters:')
    print(t0_parameters)
    amplitude_parameters = {
        'hidden_layer_sizes' : (100,100,100,100),
        'activation' : 'relu',
        'batch_size' : 500,
        'alpha' : 1.0e-2,
        'tol' : 1.0e-8,
        'verbose' : False 
        }
    print('amplitude fit parameters:')
    print(amplitude_parameters)
    print('Fitting...')
    # Teach one by one
    last_row = 0
    for i, threshold in enumerate(thresholds):
        print("Fitting threshold = {0}".format(threshold))
        section = nn_data[nn_data.threshold == threshold]
        result_t0 = fit_nn_t0(threshold, section, t0_parameters)
        joblib.dump(result_t0['model'], '../data/nn_model_t0_thr{0}.sav'.format(threshold))
        print('t0 fit for threshold {0}: score {1}, chi2 {2}, time {3}'.format(
            threshold, result_t0['score'], result_t0['chi2'], result_t0['time']))
        results[last_row, 0] = threshold
        results[last_row, 1] = 1
        results[last_row, 2] = result_t0['score']
        results[last_row, 3] = result_t0['chi2']
        results[last_row, 4] = result_t0['time']
        last_row += 1
        result_amplitude = fit_nn_amplitude(threshold, section, amplitude_parameters)
        joblib.dump(result_amplitude['model'], '../data/nn_model_amplitude_thr{0}.sav'.format(threshold))
        print('amplitude fit for threshold {0}: score {1}, chi2 {2}, time {3}'.format(
            threshold, result_amplitude['score'], result_amplitude['chi2'], result_amplitude['time']))
        results[last_row, 0] = threshold
        results[last_row, 1] = 0
        results[last_row, 2] = result_t0['score']
        results[last_row, 3] = result_t0['chi2']
        results[last_row, 4] = result_t0['time']
        last_row += 1

    end = timer()
    print('Elapsed time: {0}'.format(end - start))
    result_frame = pd.DataFrame({
        'threshold' : results[:,0],
        'is_t0' : results[:,1],
        'score' : results[:,2],
        'chi2' : results[:,3],
        'time' : results[:,4]
        })
    print(result_frame)
    result_frame.to_csv('../data/nn_fit_results_layered.csv')
