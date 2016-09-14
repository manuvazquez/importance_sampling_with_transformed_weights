#! /usr/bin/env python3

import json

import colorama

import numpy as np
from sklearn.mixture import GMM

import util

# --------------------- parameters are read

with open('clipping.json') as json_data:

	# the parameters file is read to memory
	parameters = json.load(json_data)

n_trials = parameters["number of trials"]

true_means = np.array(parameters["true means"])
variance = parameters["variance"]  # variance of *both* Gaussians (known)
mixture_coefficients = np.array(parameters["mixture coefficients"])
Ns = parameters["number of observations"]

# Monte Carlo
M = parameters["Monte Carlo"]["number of particles"]
n_monte_carlo_trials = parameters["Monte Carlo"]["number of trials"]
nu = parameters["Monte Carlo"]["prior hyperparameters"]["nu"]
lamb = parameters["Monte Carlo"]["prior hyperparameters"]["lambda"]
M_Ts_list = parameters["Monte Carlo"]["number of particles clipped"]

# if a random seed is not provided, this is None
random_seed = parameters.get("random seed")

# ---------------------

max_N = max(Ns)

# the number of components
n_mixture_components = len(mixture_coefficients)

# there should be one mixture coefficient per mean
assert len(true_means) == n_mixture_components

# bad things can happen when computing "likelihood_factors" if these don't hold due to numpy's "broadcasting rules"
assert n_mixture_components != max_N
assert n_mixture_components != M

# [<trial>, <component within the state vector>, <number of particles>, <algorithm>]
estimates = np.empty((n_trials, n_monte_carlo_trials, n_mixture_components, len(M_Ts_list), len(Ns)))

# [<trial>, <number of particles>, <algorithm>]
M_eff = np.empty((n_trials, n_monte_carlo_trials, len(M_Ts_list), len(Ns)))
max_weight = np.empty((n_trials, n_monte_carlo_trials, len(M_Ts_list), len(Ns)))

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

	observations = gmm.sample(n_samples=max_N, random_state=prng).flatten()

	for i_N, N in enumerate(Ns):

		for i_monte_carlo_trial in range(n_monte_carlo_trials):

			print('trial ' + colorama.Fore.LIGHTWHITE_EX + '{}'.format(i_trial) + colorama.Style.RESET_ALL + ' | ' +
			      'N ' + colorama.Fore.LIGHTBLUE_EX + '{}'.format(N) + colorama.Style.RESET_ALL + ' | ' +
				'MC trial ' + colorama.Fore.LIGHTGREEN_EX + '{}'.format(i_monte_carlo_trial) + colorama.Style.RESET_ALL)

			# samples are drawn for every particle *and* every component (for the maximum number of particles required)
			samples = prng.normal(loc=proposal_mean, scale=proposal_sd, size=(M, n_mixture_components))

			# the log likelihood of every sample is computed
			log_likelihood = util.compute_loglikelihoods(samples, observations[:N], mixture_coefficients, variance)

			for i_M_T, M_T in enumerate(M_Ts_list):

				# the first "M" log-likelihoods...
				copy_log_likelihoods = log_likelihood.copy()

				# clipping
				i_clipped = np.argpartition(copy_log_likelihoods, -M_T)[-M_T:]

				# minimum (unnormalized) weight among those to be clipped
				clipping_threshold = copy_log_likelihoods[i_clipped[0]]

				# the largest (unnormalized) weights are "clipped"
				copy_log_likelihoods[i_clipped] = clipping_threshold

				# log is removed
				likelihoods = np.exp(copy_log_likelihoods - copy_log_likelihoods.max())

				# weights are obtained by normalizing the likelihoods
				weights = likelihoods / likelihoods.sum()

				# TIW-based estimate
				estimates[i_trial, i_monte_carlo_trial, :, i_M_T, i_N] = weights @ samples

				# effective sample size...
				M_eff[i_trial, i_monte_carlo_trial, i_M_T, i_N] = 1.0/np.sum(weights**2)

				# ...and maximum weight
				max_weight[i_trial, i_monte_carlo_trial, i_M_T, i_N] = weights.max()


# data saving
util.save_data(parameters, estimates, true_means, M_eff, max_weight, random_seed, {
	'number of particles': M, 'M_Ts': M_Ts_list})
