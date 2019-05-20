# coding: utf-8

import math
import numpy as np
import pandas as pd
import root_pandas

from svdnoise_pdfmap import generate_random_floorplan
from svdnoise_noisegenerator import NoiseGenerator

# %% Create fitter instance
generator = NoiseGenerator()
n_samples = 6
samples_per_combo = 1 # (u,v) pairs for each (threshold, sigma, width) triplet
n_combos = 1000 # number of (threshold, sigma, width) triplets
n_rows = n_combos * samples_per_combo
pieces = []
# %% Function to process a single floorplan row
def process_row(row):
    global pieces
    n_done = len(pieces)
    if (n_done % 100) == 0:
        print('Done {0} of {1}'.format(n_done, n_combos))
    """This processes a single floorplan row."""
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

# %% Generate random floorplan
# No plots for random floorplan
generator.set_make_plots(False)
random_floorplan = generate_random_floorplan(n_combos) # otherwise use defaults
random_floorplan.apply(process_row, axis = 1)
samples = pd.concat(pieces).sample(frac = 1).reset_index(drop = True)
samples.head()
samples.to_json('noise/data/training_data_{0}.json'.format(n_rows))
samples.to_root('noise/data/training_data_{0}.root'.format(n_rows), key = 'tree')
print("Done.")
samples.head()
