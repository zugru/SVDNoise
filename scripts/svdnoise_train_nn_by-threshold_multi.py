# coding: utf-8
import multiprocessing as mp
import logging
import time
import sys

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
# This script trains a neural network random generator.
#
# It trains separate networks for each threshold
# It uses multithreading to run training in parallel
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

def fit_nn_t0(threshold, input_frame, nn_params, queue):
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
    result =  {
            'threshold' : threshold,
            'value': 't0', 
            'model': model_t0, 
            'score': score_t0, 
            'chi2': chi2_t0, 
            'n_iter' : model_t0.named_steps['mlpregressor'].n_iter_,
            'time' : end - start
            }
    queue.put(result)
    return 0

def fit_nn_amplitude(threshold, input_frame, nn_params, queue):
    """This fits a neural network to amplitude data. Returns training result."""
    start = timer()
    test_frac = 0.5
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
    result = {
            'threshold': threshold, 
            'value' : 'amplitude', 
            'model': model_amplitude, 
            'score': score_amplitude, 
            'chi2': chi2_amplitude, 
            'n_iter' : model_amplitude.named_steps['mlpregressor'].n_iter_,
            'time' : end - start
            }
    queue.put(result)
    return 0

if __name__=='__main__':
    # Multiprocessing queue will collect the results 
    queue = mp.Queue()
    # logging setup
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    processes = []
    start = timer()
    # %% Load the source file
    nn_data = pd.read_json('../data/training-data_fixed-thr_n2000000.json')
    nn_data.head()
    # Get thresholds
    thresholds = nn_data['threshold'].unique() 
    logger.info('Threshold set {0}'.format(thresholds))
    # Set and save parameters
    results_dir = '../data/results1'
    t0_parameters = {
        'hidden_layer_sizes' : (100,100),
        'activation' : 'tanh',
        'batch_size' : 500,
        'alpha' : 1.0e-7,
        'tol' : 1.0e-8,
        'random_state' : 37,
        'learning_rate_init' : 3.0e-4,
        'max_iter' : 500,
        'verbose' : False
        }
    logger.info('t0 fit parameters:')
    logger.info(t0_parameters)
    amplitude_parameters = {
        'hidden_layer_sizes' : (256,256),
        'activation' : 'relu',
        'batch_size' : 500,
        'alpha' : 1.0e-7,
        'tol' : 1.0e-8,
        'random_state' : 37,
        'learning_rate_init' : 1.0e-4,
        'max_iter' : 500,
        'verbose' : False 
        }
    logger.info('amplitude fit parameters:')
    logger.info(amplitude_parameters)
    logger.info('Fitting...')
    # In multiprocessing case, we first launch all processes, then wait for all to finish, 
    # then collect results.
    for threshold in thresholds:
        logger.info("Fitting threshold = {0}".format(threshold))
        section = nn_data[nn_data.threshold == threshold].copy()
        p = mp.Process(
                target = fit_nn_t0, 
                name = 'Time_fit_thr{0}'.format(threshold),
                args = (threshold, section, t0_parameters, queue)
                )
        p.start()
        processes.append(p)
        q = mp.Process(
                target = fit_nn_amplitude,
                name = 'Amplitude_fit_thr{0}'.format(threshold),
                args = (threshold, section, amplitude_parameters, queue)
                )
        q.start()
        processes.append(q)
        
    # Wait until queue fills
    print('Waiting for jobs to finish...')
    wstart = time.time()
    while queue.qsize() < 2 * thresholds.shape[0]:
        delta = int(time.time() - wstart + 0.5)
        s = '\rElapsed time {0}s, done {1} of {2} tasks   '.format(delta, queue.qsize(), 2*thresholds.shape[0])
        print(s, end = '') 
        time.sleep(10)

    # Collect results
    logger.info("Collecting results...")
    results = np.ndarray((2*thresholds.shape[0], 6))
    last_row = 0
    while not queue.empty():
        result = queue.get()
        joblib.dump(result['model'], '../data/nn_model_{0}_thr{1}.sav'.format(
            result['value'],
            result['threshold'] 
            ))
        results[last_row, 0] = result['threshold']
        results[last_row, 1] = 1 if result['value'] == 't0' else 0
        results[last_row, 2] = result['score']
        results[last_row, 3] = result['chi2']
        results[last_row, 4] = result['n_iter']
        results[last_row, 5] = result['time']
        last_row += 1
        
    result_frame = pd.DataFrame({
        'threshold' : results[:,0],
        'is_t0' : results[:,1],
        'score' : results[:,2],
        'chi2' : results[:,3],
        'n_iter' : results[:,4],
        'time' : results[:,5]
        })
    logger.info(result_frame)
    result_frame.to_csv('../data/nn_fit_results_layered.csv')
    end = timer()
    logger.info('Elapsed time: {0}'.format(end - start))
    
    # Wait until all processes have finished.
    for p in processes:
        logger.info('calling join on process {0}'.format(p.name))
        p.join()
