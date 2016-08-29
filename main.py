import json

# output file name
import os
import time
import socket

# data saving
import h5py

import numpy as np
from sklearn.mixture import GMM

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
M_T_over_M = parameters["Monte Carlo"]["ratio of particles to be clipped"]

# if a random seed is not provided, this is None
random_seed = parameters.get("random seed")

# ---------------------

# [<trial>, <component within the state vector>, <number of particles>, <algorithm>]
estimates = np.empty((n_trials, 2, len(Ms), 2))

# the proposal is a Gaussian density with parameters...
proposal_mean, proposal_sd = nu, np.sqrt(variance/lamb)

# pseudo-random numbers generator
prng = np.random.RandomState(random_seed)

# a Gaussian Mixture Model is built...
gmm = GMM(n_components=2, random_state=prng, n_iter=1)

# ...and the required parameters set up
gmm.means_ = np.reshape(true_means, (-1, 1))
gmm.covars_ = np.full(shape=(2,1),fill_value=variance, dtype=float)
gmm.weights_ = np.array([ro, 1-ro])

for i_trial in range(n_trials):

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

	# # in order to avoid divide-by-zero issues
	# likelihood += 1e-200

	for i_M, M in enumerate(sorted(Ms)):

		# the first "M" likelihoods...
		M_likelihoods = likelihood[:M].copy()
		
		# ...and samples are selected
		M_samples = samples[:M, :]

		# weights are obtained by normalizing the likelihoods
		weights = M_likelihoods / M_likelihoods.sum()

		# regular IW-based estimate
		estimates[i_trial, :, i_M, 0] = weights @ M_samples

		# --------------------- transformed importance weights

		# NOTE: *M_likelihoods* is modified below

		# number of highest weights for the clipping procedure
		M_T = int(M * M_T_over_M)

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

# MMSE computation [<trial>, <number of particles>, <algorithm>]
mmse = np.sum((estimates - true_means[np.newaxis, :, np.newaxis, np.newaxis])**2, axis=1)

# [<number of particles>, <algorithm>]
average_mmse = mmse.mean(axis=0)

# variance [<component of the state vector>, <number of particles>, <algorithm>]
variance = np.var(estimates, axis=0)

# --------------------- the name of the output file is constructed

# the name of the machine running the program (supposedly, using the socket module gives rise to portable code)
hostname = socket.gethostname()

# date and time
date = time.strftime("%a_%Y-%m-%d_%H:%M:%S")

# output data file
output_file = hostname + '_' + date + '_' + str(os.getpid())

# --------------------- data saving

file = h5py.File('res_' + output_file + '.hdf5', 'w')

file.create_dataset('estimated means', shape=estimates.shape, data=estimates)
file.create_dataset('true means', shape=true_means.shape, data=true_means)

file.attrs['number of particles'] = Ms
file.attrs['random seed'] = random_seed

file.close()

# --------------------- plotting

# import plot
# plot.mmse(Ms, average_mmse)
# plot.variance(Ms, variance)
