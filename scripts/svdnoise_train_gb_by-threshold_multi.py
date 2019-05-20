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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.ensemble import GradientBoostingRegressor
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

def fig_gb_t0(threshold, input_frame, bg_params, queue):
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
        #FunctionTransformer(q_to_z, z_to_q, kw_args = {'ncols' : 1}, inv_kw_args = {'ncols' : 1}),
        #StandardScaler(),
        MinMaxScaler(),
        GradientBoostingRegressor(**bg_params)
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
            'time' : end - start
            }
    queue.put(result)
    return 0

def fig_gb_amplitude(threshold, input_frame, bg_params, queue):
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
        #FunctionTransformer(q_to_z, z_to_q, kw_args = {'ncols' : 2}, inv_kw_args = {'ncols' : 2}),
        #StandardScaler(),
        MinMaxScaler(),
        GradientBoostingRegressor(**bg_params)
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
    bg_data = pd.read_json('../data/training-data_fixed-thr_n2000000.json')
    bg_data.head()
    # Get thresholds
    thresholds = bg_data['threshold'].unique() 
    logger.info('Threshold set {0}'.format(thresholds))
    t0_parameters = {
            'max_leaf_nodes' : 2000,
            'loss' : 'ls',
            'n_estimators' : 1000
    }
    logger.info('t0 fit parameters:')
    logger.info(t0_parameters)
    amplitude_parameters = {
            'max_leaf_nodes' : 2000,
            'loss' : 'ls',
            'n_estimators' : 1000
    }
    logger.info('amplitude fit parameters:')
    logger.info(amplitude_parameters)
    logger.info('Fitting...')
    # In multiprocessing case, we first launch all processes, then wait for all to finish, 
    # then collect results.
    for threshold in thresholds:
        logger.info("Fitting threshold = {0}".format(threshold))
        section = bg_data[bg_data.threshold == threshold].copy()
        p = mp.Process(
                target = fig_gb_t0, 
                name = 'Time_fit_thr{0}'.format(threshold),
                args = (threshold, section, t0_parameters, queue)
                )
        p.start()
        processes.append(p)
        q = mp.Process(
                target = fig_gb_amplitude,
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
    results = np.ndarray((2*thresholds.shape[0], 5))
    last_row = 0
    while not queue.empty():
        result = queue.get()
        joblib.dump(result['model'], '../data/gb_model_{0}_thr{1}.sav'.format(
            result['value'],
            result['threshold'] 
            ))
        results[last_row, 0] = result['threshold']
        results[last_row, 1] = 1 if result['value'] == 't0' else 0
        results[last_row, 2] = result['score']
        results[last_row, 3] = result['chi2']
        results[last_row, 4] = result['time']
        last_row += 1
        
    result_frame = pd.DataFrame({
        'threshold' : results[:,0],
        'is_t0' : results[:,1],
        'score' : results[:,2],
        'chi2' : results[:,3],
        'time' : results[:,4]
        })
    logger.info(result_frame)
    result_frame.to_csv('../data/bg_fit_results_layered.csv')
    end = timer()
    logger.info('Elapsed time: {0}'.format(end - start))
    
    # Wait until all processes have finished.
    for p in processes:
        logger.info('calling join on process {0}'.format(p.name))
        p.join()
