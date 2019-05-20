# coding: utf-8

# ---------------------------------------------------------------------
# Generate training data by threshold
# 
# This generates separate data files for different thresholds.
# ---------------------------------------------------------------------
import math
import numpy as np
import pandas as pd
import root_pandas

from timeit import default_timer as timer

from svdnoise_pdfmap import generate_mixed_floorplan
from svdnoise_noisegenerator import NoiseGenerator

# %% Function to process a single floorplan row
def process_row(row):
    """This processes a single floorplan row."""
    global pieces
    global generator
    global n_samples
    global n_combos
    global samples_per_combo
    n_done = len(pieces)
    if (n_done % 100) == 0:
        print('Done {0} of {1}'.format(n_done, n_combos))
    generator.generate_pdf(row.threshold, row.sigma, row.width, n_samples)
    u = np.random.uniform(0,1,samples_per_combo)
    v = np.random.uniform(0,1,samples_per_combo)
    t0amps = generator.random_transform(u, v)
    result = pd.DataFrame({
        'threshold' : np.repeat(row.threshold, samples_per_combo),
        'sigma' : np.repeat(row.sigma, samples_per_combo),
        'width' : np.repeat(row.width, samples_per_combo),
        'u' : u,
        'v' : v,
        'vprime' : t0amps['vprime'],
        't0' : t0amps['t0'],
        'amplitude' : t0amps['amplitude']
    })
    pieces += [result]
    return 0

if __name__ == '__main__':
    start = timer()
    # %% Create fitter instance
    generator = NoiseGenerator()
    n_samples = 6
    samples_per_combo = 1000 # (u,v) pairs for each (threshold, sigma, width) triplet
    n_combos = 2000 # number of (threshold, sigma, width) triplets
    n_rows = n_combos * samples_per_combo
    pieces = []
    floorplan = generate_mixed_floorplan(n_combos) # otherwise use defaults
    floorplan.apply(process_row, axis = 1)
    samples = pd.concat(pieces).sample(frac = 1).reset_index(drop = True)
    samples.head()
    samples.to_json('../data/training-data_fixed-thr_n{0}.json'.format(n_rows))
    samples.to_root('../data/training-data_fixed-thr_n{0}.root'.format(n_rows), key = 'tree')
    end = timer()
    print("Done in {0} seconds.".format(end - start))
