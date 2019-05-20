# SVDNoise

Authors: Zuzana Gruberova, Peter Kvasnicka


This repository contains various pieces of code for the development of SVD noise generator.

The *scripts* folder contains scripts for

1.  generating probability distribution.  noise signals (svdnoise_pfmap.py)

2.  generatinng random samples from "exact" probability distributions (svdnoise_noisegenerator.py) 

3.  training neural network random samplers  (svdnoise_train_nn_*.py) and checking the results (svdnoise_nn_check_*.py) 

4.  training a gradient-boosted decistion tree estimator to work as noise sampler (svdnoise_train_gb_*.py) and checking the results (svdnoise_gb_check_*.py)

5.  training a Bayesian Gaussian Mixtrure noise sampler (svdnoise_train_bgm_*.py) and checking the results (svdnoise_bgm_check_*.py)

The *data* folder contains training data and trained estimators. Due to GitHub file size limits, only smaller training data samples are included (i.e., no n1000000 or n20000000 files).

The *pictures* folder contains a variety of diagnostic plots. No pictures included.
