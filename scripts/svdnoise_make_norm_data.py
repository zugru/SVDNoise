# coding: utf-8

import math
import numpy as np
import pandas as pd
#import root_pandas
%matplotlib inline
from matplotlib import pyplot as plt
from svdnoise_pdfmap import generate_mixed_floorplan, generate_regular_floorplan, base_thresholds
from svdnoise_noisegenerator import NoiseGenerator

# %% Create generator instance
generator = NoiseGenerator()
n_samples = 6
# %% Function to process a single floorplan row. We only take norm data.
def process_row(row):
    """This processes a single floorplan row."""
    #print('Computing norm: threshold = {0}, sigma = {1}, width = {2}...'.format(row.threshold, row.sigma, row.width))
    generator.generate_pdf(row.threshold, row.sigma, row.width, n_samples)
    return generator.get_norm()

# %% Generate regular floorplan
regular_floorplan = generate_regular_floorplan(
    thresholds = base_thresholds,
    sigmas = np.arange(0.05, 0.95, 0.1),
    widths = np.arange(200, 376, 25)
    ) # make finer floorplan
regular_floorplan['norm'] = regular_floorplan.apply(process_row, axis = 1)
regular_floorplan['lognorm'] = np.log10(regular_floorplan['norm'])
regular_floorplan.to_json('../data/norms_regular_floorplan.json')
#regular_floorplan.to_root('noise/data/norms_regular_floorplan.root', key = 'tree')
# %% Generate random floorplan
n_rows = 5000
mixed_floorplan = generate_mixed_floorplan(n_rows) # otherwise use defaults
mixed_floorplan['norm'] = mixed_floorplan.apply(process_row, axis = 1)
mixed_floorplan['lognorm'] = np.log10(mixed_floorplan['norm'])
mixed_floorplan.to_json('../data/norms_mixed_floorplan_n{0}.json'.format(n_rows))
#mixed_floorplan.to_root('noise/data/norms_mixed_floorplan_n{0}.root'.format(n_rows), key = 'tree')
print("Done.")
mixed_floorplan.head()
