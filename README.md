# GP4GW-likelihood
Gaussian Processes for Gravitational Waves Likelihood Reconstruction.

The likelihood surface is modelled with a Gaussian Process as a multi-dimensional function y = f(x), where y are the likelihood evaluations over parameter space (need to use at least all sampled dimensions) and x are MCMC posterior samples.

The GP model is built using `GPFlow` and `TensorFlow`.
We built a `gp4gw` package which allows to load and prepare GW data for the modelling step. The package also allows to sample the modelled likelihood multiplied by the analytical priors in order to obtain more posterior samples in any desired region of the parameter space.

## Set-up on Eyas cluster

## Running notebooks

## Saving models 
