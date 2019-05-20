# coding: utf-8

import math
import numpy as np
import pandas as pd
from svdnoise_pdfmap import generate_random_floorplan
from svdnoise_noisegenerator import NoiseGenerator

# %% Create fitter instance
generator = NoiseGenerator()
n_samples = 6
samples_per_combo = 100000 # (u,v) pairs for each (threshold, sigma, width) triplet
n_combos = 1 # number of (threshold, sigma, width) triplets
n_rows = n_combos * samples_per_combo
pieces = []
# %% Function to process a single floorplan row
def process_row(row):
    global pieces
    if (len(pieces) % 1000) == 0:
        print(len(pieces))
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

# %% no floorplan, just a single combination of threshold/sigma/with :
generator.set_make_plots(False)
floorplan = pd.DataFrame({
    'threshold' : [4.0], 'sigma' : [0.25], 'width' : [200]
}) # otherwise use defaults
floorplan.apply(process_row, axis = 1)
samples = pd.concat(pieces).sample(frac = 1).reset_index(drop = True)
samples.head()
samples.to_json('../data/single_training_data_{0}.json'.format(n_rows))
# %%
print("Done.")
# %%
samples.head()
print(len(samples))
