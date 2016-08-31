#! /usr/bin/env python3

import os
import sys
import json
import pickle
import types

import h5py
import colorama

import numpy as np
from sklearn.mixture import GMM

sys.path.append(os.path.join(os.environ['HOME'], 'python'))

import manu.util

# --------------------- parameters are read

with open('parameters.json') as json_data:

	# the parameters file is read to memory
	parameters = json.load(json_data)

n_trials = parameters["number of trials"]

true_means = np.array(parameters["true means"])
variance = parameters["variance"] # variance of *both* Gaussians (known)
ro = parameters["mixture coefficient"]
N = parameters["number of observations"]

# Monte Carlo
Ms = parameters["Monte Carlo"]["number of particles"]
nu = parameters["Monte Carlo"]["prior hyperparameters"]["nu"]
lamb = parameters["Monte Carlo"]["prior hyperparameters"]["lambda"]
n_clipped_particles_from_overall = eval(parameters["Monte Carlo"]["number of clipped particles from overall number"])

# this should be a function
assert type(n_clipped_particles_from_overall) == types.FunctionType

# if a random seed is not provided, this is None
random_seed = parameters.get("random seed")

# ---------------------

# number of highest weights for the clipping procedure
M_Ts_list = [n_clipped_particles_from_overall(M) for M in Ms]

# [<trial>, <component within the state vector>, <number of particles>, <algorithm>]
estimates = np.empty((n_trials, 2, len(Ms), 2))

# [<trial>, <number of particles>, <algorithm>]
M_eff = np.empty((n_trials, len(Ms), 2))
max_weight = np.empty((n_trials, len(Ms), 2))

# the proposal is a Gaussian density with parameters...
proposal_mean, proposal_sd = nu, np.sqrt(variance/lamb)

# pseudo-random numbers generator
prng = np.random.RandomState(random_seed)

# a Gaussian Mixture Model is built...
gmm = GMM(n_components=2, random_state=prng, n_iter=1)

# ...and the required parameters set up
gmm.means_ = np.reshape(true_means, (-1, 1))
gmm.covars_ = np.full(shape=(2, 1), fill_value=variance, dtype=float)
gmm.weights_ = np.array([ro, 1-ro])

for i_trial in range(n_trials):

	print('trial ' + colorama.Fore.LIGHTWHITE_EX + '{}'.format(i_trial) + colorama.Style.RESET_ALL)

	observations = gmm.sample(n_samples=N, random_state=prng).flatten()

	# the maximum number of particles
	max_M = max(Ms)

	# samples are drawn for every particle *and* every component (for the maximum number of particles required)
	samples = prng.normal(loc=proposal_mean, scale=proposal_sd, size=(max_M, 2))

	# computation of the factor every *individual* observations contributes to the likelihood
	# (every row is associated with an observation, and every column with a particle)
	likelihood_factors = ro*np.exp(-(observations[:, np.newaxis]-samples[:, 0][np.newaxis, :])**2
	                               /(2*variance))/np.sqrt(2*np.pi*variance) + \
	                     (1.0-ro)*np.exp(-(observations[:, np.newaxis]-samples[:, 1][np.newaxis, :])**2
	                                     /(2*variance))/np.sqrt(2*np.pi*variance)

	# the likelihood is given by the product of the individual factors
	likelihood = likelihood_factors.prod(axis=0)

	for i_M, (M, M_T) in enumerate(zip(Ms, M_Ts_list)):

		# the first "M" likelihoods...
		M_likelihoods = likelihood[:M].copy()
		
		# ...and samples are selected
		M_samples = samples[:M, :]

		# weights are obtained by normalizing the likelihoods
		weights = M_likelihoods / M_likelihoods.sum()

		# regular IW-based estimate
		estimates[i_trial, :, i_M, 0] = weights @ M_samples

		# effective sample size...
		M_eff[i_trial, i_M, 0] = 1.0/np.sum(weights**2)

		# ...and maximum weight
		max_weight[i_trial, i_M, 0] = weights.max()

		# --------------------- transformed importance weights

		# NOTE: *M_likelihoods* is modified below

		# clipping
		i_clipped = np.argpartition(M_likelihoods, -M_T)[-M_T:]

		# minimum (unnormalized) weight among those to be clipped
		clipping_threshold = M_likelihoods[i_clipped[0]]

		# the largest (unnormalized) weights are "clipped"
		M_likelihoods[i_clipped] = clipping_threshold

		# normalized weights are obtained
		weights = M_likelihoods / M_likelihoods.sum()

		#  TIW-based estimate
		estimates[i_trial, :, i_M, 1] = weights @ M_samples

		M_eff[i_trial, i_M, 1] = 1.0 / np.sum(weights ** 2)
		max_weight[i_trial, i_M, 1] = weights.max()

# MMSE computation [<trial>, <number of particles>, <algorithm>]
mmse = np.sum((estimates - true_means[np.newaxis, :, np.newaxis, np.newaxis])**2, axis=1)

# [<number of particles>, <algorithm>]
average_mmse = mmse.mean(axis=0)

# variance [<component of the state vector>, <number of particles>, <algorithm>]
estimates_variance = np.var(estimates, axis=0)

print(average_mmse)

# --------------------- data saving

# output data file
output_file = manu.util.filename_from_host_and_date()

file = h5py.File('res_' + output_file + '.hdf5', 'w')

file.create_dataset('estimated means', shape=estimates.shape, data=estimates)
file.create_dataset('true means', shape=true_means.shape, data=true_means)
file.create_dataset('M_eff', shape=M_eff.shape, data=M_eff)
file.create_dataset('maximum weight', shape=max_weight.shape, data=max_weight)

file.attrs['number of particles'] = Ms

if random_seed:

	file.attrs['random seed'] = random_seed

file.close()

# in a separate file with the same name as the data file but different extension...
parameters_file = 'res_{}.parameters'.format(output_file)

with open(parameters_file, mode='wb') as f:

	#  ...parameters are pickled
	pickle.dump(parameters, f)

print('parameters saved in "{}"'.format(parameters_file))
