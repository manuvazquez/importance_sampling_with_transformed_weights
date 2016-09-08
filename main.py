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
variance = parameters["variance"]  # variance of *both* Gaussians (known)
mixture_coefficients = np.array(parameters["mixture coefficients"])
N = parameters["number of observations"]

# Monte Carlo
Ms = parameters["Monte Carlo"]["number of particles"]
n_monte_carlo_trials = parameters["Monte Carlo"]["number of trials"]
nu = parameters["Monte Carlo"]["prior hyperparameters"]["nu"]
lamb = parameters["Monte Carlo"]["prior hyperparameters"]["lambda"]
n_clipped_particles_from_overall = eval(parameters["Monte Carlo"]["number of clipped particles from overall number"])

# if a random seed is not provided, this is None
random_seed = parameters.get("random seed")

# ---------------------

# the number of components
n_mixture_components = len(mixture_coefficients)

# the maximum number of particles
max_M = max(Ms)

# the parameter below should be a function
assert isinstance(n_clipped_particles_from_overall, types.FunctionType)

# there should be one mixture coefficient per mean
assert len(true_means) == n_mixture_components

# bad things can happen when computing "likelihood_factors" if these don't hold due to numpy's "broadcasting rules"
assert n_mixture_components != N
assert n_mixture_components != max_M

# number of highest weights for the clipping procedure
M_Ts_list = [n_clipped_particles_from_overall(M) for M in Ms]

# [<trial>, <component within the state vector>, <number of particles>, <algorithm>]
estimates = np.empty((n_trials, n_monte_carlo_trials, n_mixture_components, len(Ms), 2))

# [<trial>, <number of particles>, <algorithm>]
M_eff = np.empty((n_trials, n_monte_carlo_trials, len(Ms), 2))
max_weight = np.empty((n_trials, n_monte_carlo_trials, len(Ms), 2))

# the proposal is a Gaussian density with parameters...
proposal_mean, proposal_sd = nu, np.sqrt(variance/lamb)

# pseudo-random numbers generator
prng = np.random.RandomState(random_seed)

# a Gaussian Mixture Model is built...
gmm = GMM(n_components=n_mixture_components, random_state=prng, n_iter=1)

# ...and the required parameters set up
gmm.means_ = np.reshape(true_means, (-1, 1))
gmm.covars_ = np.full(shape=(n_mixture_components, 1), fill_value=variance, dtype=float)
gmm.weights_ = mixture_coefficients

for i_trial in range(n_trials):

	observations = gmm.sample(n_samples=N, random_state=prng).flatten()

	for i_monte_carlo_trial in range(n_monte_carlo_trials):

		print('trial ' + colorama.Fore.LIGHTWHITE_EX + '{}'.format(i_trial) + colorama.Style.RESET_ALL + ' | ' +
			'MC trial ' + colorama.Fore.LIGHTGREEN_EX + '{}'.format(i_monte_carlo_trial) + colorama.Style.RESET_ALL)

		# samples are drawn for every particle *and* every component (for the maximum number of particles required)
		samples = prng.normal(loc=proposal_mean, scale=proposal_sd, size=(max_M, n_mixture_components))

		# intermediate result for computating the factor every *individual* observations contributes to the likelihood
		# [<observation>, <particle>, <component>]
		aux_likelihood_factors = np.exp(
			-(observations[:, np.newaxis, np.newaxis] - samples[np.newaxis, :, :])**2 / (2 * variance)
		) /np.sqrt(2*np.pi*variance)

		likelihood_factors = np.sum(mixture_coefficients * aux_likelihood_factors, axis=-1)

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# in order to avoid underflows/overflows, we work with the logarithm of the likelihoods
		log_likelihood_factors = np.log(likelihood_factors)

		# the (log) likelihood is given by the (sum) product of the individual factors
		log_likelihood = log_likelihood_factors.sum(axis=0)

		for i_M, (M, M_T) in enumerate(zip(Ms, M_Ts_list)):

			# the first "M" log-likelihoods...
			M_log_likelihoods = log_likelihood[:M].copy()

			# ...and samples are selected
			M_samples = samples[:M, :]

			# scaling followed by exponentiation to drop the logarithm
			M_likelihoods = np.exp(M_log_likelihoods - M_log_likelihoods.max())

			# weights are obtained by normalizing the likelihoods
			weights = M_likelihoods / M_likelihoods.sum()

			# regular IW-based estimate
			estimates[i_trial, i_monte_carlo_trial, :, i_M, 0] = weights @ M_samples

			# effective sample size...
			M_eff[i_trial, i_monte_carlo_trial, i_M, 0] = 1.0/np.sum(weights**2)

			# ...and maximum weight
			max_weight[i_trial, i_monte_carlo_trial, i_M, 0] = weights.max()

			# --------------------- transformed importance weights

			# NOTE: *M_log_likelihoods* is modified below

			# clipping
			i_clipped = np.argpartition(M_log_likelihoods, -M_T)[-M_T:]

			# minimum (unnormalized) weight among those to be clipped
			clipping_threshold = M_log_likelihoods[i_clipped[0]]

			# the largest (unnormalized) weights are "clipped"
			M_log_likelihoods[i_clipped] = clipping_threshold

			# log is removed
			M_likelihoods = np.exp(M_log_likelihoods - M_log_likelihoods.max())

			# weights are obtained by normalizing the likelihoods
			weights = M_likelihoods / M_likelihoods.sum()

			#  TIW-based estimate
			estimates[i_trial, i_monte_carlo_trial, :, i_M, 1] = weights @ M_samples

			# effective sample size and maximum weight
			M_eff[i_trial, i_monte_carlo_trial, i_M, 1] = 1.0 / np.sum(weights ** 2)
			max_weight[i_trial, i_monte_carlo_trial, i_M, 1] = weights.max()


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
