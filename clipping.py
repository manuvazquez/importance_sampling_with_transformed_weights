#! /usr/bin/env python3

import os
import sys
import json
import pickle

import h5py
import colorama

import numpy as np
from sklearn.mixture import GMM

sys.path.append(os.path.join(os.environ['HOME'], 'python'))

import manu.util

# --------------------- parameters are read

with open('clipping.json') as json_data:

	# the parameters file is read to memory
	parameters = json.load(json_data)

n_trials = parameters["number of trials"]

true_means = np.array(parameters["true means"])
variance = parameters["variance"]  # variance of *both* Gaussians (known)
ro = parameters["mixture coefficient"]
Ns = parameters["number of observations"]

# Monte Carlo
M = parameters["Monte Carlo"]["number of particles"]
nu = parameters["Monte Carlo"]["prior hyperparameters"]["nu"]
lamb = parameters["Monte Carlo"]["prior hyperparameters"]["lambda"]

# if a random seed is not provided, this is None
random_seed = parameters.get("random seed")

# ---------------------

# number of particles *clipped* to be tested
M_Ts = np.arange(1, int(M/2), 5)

# [<trial>, <component within the state vector>, <number of clipped particles>, <number of observations>]
estimates = np.empty((n_trials, 2, len(M_Ts), len(Ns)))

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

	for i_N, N in enumerate(Ns):

		observations = gmm.sample(n_samples=N, random_state=prng).flatten()

		# samples are drawn for every particle *and* every component (for the maximum number of particles required)
		samples = prng.normal(loc=proposal_mean, scale=proposal_sd, size=(M, 2))

		# computation of the factor every *individual* observations contributes to the likelihood
		# (every row is associated with an observation, and every column with a particle)
		likelihood_factors = ro*np.exp(-(observations[:, np.newaxis]-samples[:, 0][np.newaxis, :])**2
		                               /(2*variance))/np.sqrt(2*np.pi*variance) + \
		                     (1.0-ro)*np.exp(-(observations[:, np.newaxis]-samples[:, 1][np.newaxis, :])**2
		                                     /(2*variance))/np.sqrt(2*np.pi*variance)

		# in order to avoid underflows/overflows, we work with the logarithm of the likelihoods
		log_likelihood_factors = np.log(likelihood_factors)

		# the (log) likelihood is given by the (sum) product of the individual factors
		log_likelihood = log_likelihood_factors.sum(axis=0)

		for i_M_T, M_T in enumerate(M_Ts):

			copy_log_likelihood = log_likelihood.copy()

			# clipping
			i_clipped = np.argpartition(copy_log_likelihood, -M_T)[-M_T:]

			# minimum (unnormalized) weight among those to be clipped
			clipping_threshold = copy_log_likelihood[i_clipped[0]]

			# the largest (unnormalized) weights are "clipped"
			copy_log_likelihood[i_clipped] = clipping_threshold

			# log is removed
			likelihood = np.exp(copy_log_likelihood - copy_log_likelihood.max())

			# weights are obtained by normalizing the likelihoods
			weights = likelihood / likelihood.sum()

			#  TIW-based estimate
			estimates[i_trial, :, i_M_T, i_N] = weights @ samples

# MSE computation [<trial>, <number of clipped particles>, <number of observations>]
mse = np.sum((estimates - true_means[np.newaxis, :, np.newaxis, np.newaxis]) ** 2, axis=1)

# [<number of clipped particles>, <number of observations>]
average_mse = mse.mean(axis=0)
variance_mse = np.var(mse, axis=0)

# [<component within the state vector>, <number of clipped particles>, <number of observations>]
estimates_variance = np.var(estimates, axis=0)

print(average_mse)
print(variance_mse)

# --------------------- data saving

# output data file
output_file = manu.util.filename_from_host_and_date()

file = h5py.File('res_' + output_file + '.hdf5', 'w')

file.create_dataset('estimated means', shape=estimates.shape, data=estimates)
file.create_dataset('true means', shape=true_means.shape, data=true_means)

file.attrs['number of clipped particles'] = M_Ts
file.attrs['number of observations'] = Ns

if random_seed:

	file.attrs['random seed'] = random_seed

file.close()

# in a separate file with the same name as the data file but different extension...
parameters_file = 'res_{}.parameters'.format(output_file)

with open(parameters_file, mode='wb') as f:

	#  ...parameters are pickled
	pickle.dump(parameters, f)

print('parameters saved in "{}"'.format(parameters_file))

# import plot
# plot.single_curve(
# 	M_Ts, average_mse, 'mse',
# 	axes_properties={'xlabel': '$M_T$', 'ylabel': 'MSE', 'yscale': 'log'},
# 	output_file='average_mse_N={}_trials={}.pdf'.format(N, n_trials))
# plot.single_curve(
# 	M_Ts, variance_mse, 'variance',
# 	axes_properties={'xlabel': '$M_T$', 'ylabel': 'variance MSE', 'yscale': 'log'},
# 	output_file='variance_mse_N={}_trials={}.pdf'.format(N, n_trials))

# import code
# code.interact(local=dict(globals(), **locals()))