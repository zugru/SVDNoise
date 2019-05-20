# coding: utf-8
import multiprocessing as mp
import logging
import time
import sys

import numpy as np
import pandas as pd
from timeit import default_timer as timer

from scipy.stats import norm

#from matplotlib import pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.mixture import BayesianGaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

#---------------------------------------------------------------------
# This script trains a Bayesian Gaussian Mixture random generator.
#
# It trains separate mixtures for each threshold. Random samples are taken
# from conditional distributions. 
# It uses multithreading to run training in parallel
#---------------------------------------------------------------------

def margin_density_rms(bgm_fit):
    """Calculates RMS of margin densities for sigma and width from uniform."""
    bgm = bgm_fit.named_steps['bayesiangaussianmixture']
    weights = bgm.weights_ # n_components
    means = bgm.means_ # n_components x n_features
    covariances = bgm.covariances_ # n_components x n_features x n_features
    n_components = means.shape[0]
    n_features = 4
    n_points = 100
    u = np.linspace(0.0,1.0,n_points, endpoint = True)
    m_sigma = np.zeros_like(u)
    m_width = np.zeros_like(u)
    for comp in range(n_components):
        comp_weight = weights[comp]
        comp_mean_sigma = means[comp,0]
        comp_mean_width = means[comp,1]
        comp_std_sigma = np.sqrt(covariances[comp,0,0])
        comp_std_width = np.sqrt(covariances[comp,1,1])
        m_sigma += comp_weight * norm.pdf(u, comp_mean_sigma, comp_std_sigma)
        m_width += comp_weight * norm.pdf(u, comp_mean_width, comp_std_width)
    rms_sigma = np.sqrt(np.sum(m_sigma-1)**2 / n_points)
    rms_width = np.sqrt(np.sum(m_width-1)**2 / n_points)
    return (rms_sigma, rms_width)

def fit_bgm(threshold, input_frame, bgm_params, queue):
    """This fits a gaussian mixture to threshold data. Returns training result."""
    start = timer()
    test_frac = 0.5
    # Names of data frame columns.
    nrows = len(input_frame)
    n_test = int(test_frac * nrows)
    X  = np.asarray(input_frame[['sigma', 'width', 't0', 'amplitude']])
    X_train = X[:-n_test,:]
    X_test = X[-n_test:,:]
    model_bgm = make_pipeline(
        MinMaxScaler(), # scale all to (0,1)
        BayesianGaussianMixture(**bgm_params)
    )
    model_bgm.fit(X_train)
    score_bgm = model_bgm.score(X_test)
    rms_sigma, rms_width = margin_density_rms(model_bgm)
    end = timer()
    result =  {
            'threshold' : threshold,
            'value': 'bgm', 
            'model': model_bgm, 
            'score': score_bgm, 
            'rms_sigma' : rms_sigma,
            'rms_width' : rms_width,
            'n_iter' : model_bgm.named_steps['bayesiangaussianmixture'].n_iter_,
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
    nn_data = pd.read_json('../data/training-data_fixed-thr_n100000.json')
    nn_data.head()
    # Get thresholds
    thresholds = nn_data['threshold'].unique() 
    logger.info('Threshold set {0}'.format(thresholds))
    bgm_parameters = {
        'n_components' : 100, 
        'tol' : 0.1, 
        'n_init' : 1, 
        'covariance_type' : 'full', 
        'max_iter' : 1000, 
        'weight_concentration_prior' : 2.0/100,
        'random_state' : 37,
        'verbose' : 1, 
        'verbose_interval' : 100
        }
    logger.info('BGM fit parameters:')
    logger.info(bgm_parameters)
    logger.info('Fitting...')
    # In multiprocessing case, we first launch all processes, then wait for all to finish, 
    # then collect results.
    for threshold in thresholds:
        logger.info("Fitting threshold = {0}".format(threshold))
        section = nn_data[nn_data.threshold == threshold].copy()
        p = mp.Process(
                target = fit_bgm, 
                name = 'Time_fit_thr{0}'.format(threshold),
                args = (threshold, section, bgm_parameters, queue)
                )
        p.start()
        processes.append(p)
        
    # Wait until queue fills
    print('Waiting for jobs to finish...')
    wstart = time.time()
    while queue.qsize() < thresholds.shape[0]:
        delta = int(time.time() - wstart + 0.5)
        s = '\rElapsed time {0}s, done {1} of {2} tasks   '.format(delta, queue.qsize(), thresholds.shape[0])
        print(s, end = '') 
        time.sleep(10)

    # Collect results
    logger.info("Collecting results...")
    results = np.ndarray((thresholds.shape[0], 7))
    last_row = 0
    while not queue.empty():
        result = queue.get()
        joblib.dump(result['model'], '../data/bgm_model_thr{0}.sav'.format(
            result['threshold'] 
            ))
        results[last_row, 0] = result['threshold']
        results[last_row, 1] = 1 if result['value'] == 'bgm' else 0
        results[last_row, 2] = result['score']
        results[last_row, 3] = result['rms_sigma']
        results[last_row, 4] = result['rms_width']
        results[last_row, 5] = result['n_iter']
        results[last_row, 6] = result['time']
        last_row += 1
        
    result_frame = pd.DataFrame({
        'threshold' : results[:,0],
        'value' : 'bgm',
        'score' : results[:,2],
        'rms_sigma' : results[:,3],
        'rms_width' : results[:,4],
        'n_iter' : results[:,5],
        'time' : results[:,6]
        })
    logger.info(result_frame)
    result_frame.to_csv('../data/bgm_fit_results_layered.csv')
    end = timer()
    logger.info('Elapsed time: {0}'.format(end - start))
    
    # Wait until all processes have finished.
    for p in processes:
        logger.info('calling join on process {0}'.format(p.name))
        p.join()
