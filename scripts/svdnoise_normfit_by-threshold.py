# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

#---------------------------------------------------------------------
# This file contains fucntions related to linear model fitting of
# pdf norms. We make separate fits for each threshold value.
#---------------------------------------------------------------------

# Names of data frame columns.
floorplan_cols = ['sigma', 'width']
norm_cols = ['lognorm']

def fit_norm(threshold, fit_frame, reg_frame, degree = 3):
    """
    This fits a linear model to pdf norm. The dominant part of the model is
    log(norm) ~ threshold**2, which is however split away from our model, so
    only minor and potentially irregular dependencies on sigma and width
    remain.
    fit_frame - mixed floorplan data used in fit,
    reg_frame - regular floorplan data used for validation
    Returns the fitted model object. Input frame is unchanged.
    """
    X  = np.asarray(fit_frame[floorplan_cols])
    X_reg = np.asarray(reg_frame[floorplan_cols])
    y = np.asarray(fit_frame[norm_cols])
    y_reg = np.asarray(reg_frame[norm_cols])
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree),
        LinearRegression(normalize = True)
    )
    model.fit(X, y)
    fit_score = model.score(X_reg, y_reg)
    y_pred = model.predict(X_reg)
    fit_chi2 = np.sqrt(np.sum((y_reg - y_pred)**2) / y_reg.shape[0])
    return model, fit_score, fit_chi2

# %%
if __name__ == '__main__':
    norm_data = pd.read_json('../data/norms_mixed_floorplan_n5000.json')
    ref_data = pd.read_json('../data/norms_regular_floorplan.json')
    results = []
    for threshold in [3.0, 4.0, 5.0, 6.0, 7.0]:
        print('Fitting threshold {0}'.format(threshold))
        threshold_data = norm_data[norm_data.threshold == threshold].copy()
        threshold_ref = ref_data[ref_data.threshold == threshold].copy()
        model, score, chi2 = fit_norm(threshold, threshold_data, threshold_ref, degree = 5)
        results.append({
            'threshold' : threshold,
            'score' : score,
            'chi2' : chi2
        })
        joblib.dump(model, '../data/normfit_thr{0}.sav'.format(int(threshold)))

    res_frame = pd.DataFrame(results)
    print(res_frame)
    print("Done.")
