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

import util
import manu.util
import manu.resampling

# --------------------- parameters are read

with open('pmc.json') as json_data:

	# the parameters file is read to memory
	parameters = json.load(json_data)

n_trials = parameters["number of trials"]

true_means = np.array(parameters["true means"])
variance = parameters["variance"]  # variance of *both* Gaussians (known)
mixture_coefficients = np.array(parameters["mixture coefficients"])
N = parameters["number of observations"]

# Monte Carlo
M = parameters["Monte Carlo"]["number of particles"]
n_monte_carlo_trials = parameters["Monte Carlo"]["number of trials"]
n_pmc_iterations = parameters["Monte Carlo"]["number of PMC iterations"]
nu = parameters["Monte Carlo"]["prior hyperparameters"]["nu"]
lamb = parameters["Monte Carlo"]["prior hyperparameters"]["lambda"]
n_clipped_particles_from_overall = eval(parameters["Monte Carlo"]["number of clipped particles from overall number"])

# if a random seed is not provided, this is None
random_seed = parameters.get("random seed")

# ---------------------

# the number of components
n_mixture_components = len(mixture_coefficients)

# the parameter below should be a function
assert isinstance(n_clipped_particles_from_overall, types.FunctionType)

# there should be one mixture coefficient per mean
assert len(true_means) == n_mixture_components

# number of highest weights for the clipping procedure
M_T = n_clipped_particles_from_overall(M)

# [<trial>, <component within the state vector>, <number of PMC iteration>, <algorithm>]
estimates = np.empty((n_trials, n_monte_carlo_trials, n_mixture_components, n_pmc_iterations, 2))

# [<trial>, <number of PMC iteration>, <algorithm>]
M_eff = np.empty((n_trials, n_monte_carlo_trials, n_pmc_iterations, 2))
max_weight = np.empty((n_trials, n_monte_carlo_trials, n_pmc_iterations, 2))

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

# ----------


class NonlinearPopulationMonteCarlo:

	def __init__(self, prior_mean, prior_covariance, M, M_T, mixture_coefficients, variance, prng):

		# NOTE: this is fine because "self.mean" is never modified inplace, but rather the *reference is replaced
		self.mean = self.prior_mean = prior_mean
		self.covariance = self.prior_covariance = prior_covariance

		self.M = M
		self.M_T = M_T

		self.mixture_coefficients = mixture_coefficients
		self.variance = variance

		self.prng = prng

		self.i_clipped, self.log_likelihoods = None, None
		self.samples, self.weights = None, None

		self.resampling_alg = manu.resampling.MultinomialResamplingAlgorithm(prng)

	def reset(self):

		self.mean = self.prior_mean
		self.covariance = self.prior_covariance

	def resample_and_update_proposal(self):

		# minimum (unnormalized) weight among those to be clipped
		clipping_threshold = self.log_likelihoods[self.i_clipped[0]]

		# the likelihoods with the largest (unnormalized) weights are "clipped"
		self.log_likelihoods[self.i_clipped] = clipping_threshold

		# scaling followed by exponentiation to drop the logarithm
		likelihoods = np.exp(self.log_likelihoods - self.log_likelihoods.max())

		self.weights = likelihoods / likelihoods.sum()

		i_resampled = self.resampling_alg.get_indexes(self.weights)
		self.samples = self.samples[i_resampled, :]

		self.mean = self.samples.mean(axis=0)
		self.covariance = np.cov(self.samples.T, ddof=0)

	def iterate(self, obs):

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# samples are drawn for every particle *and* every component (for the maximum number of particles required)
		self.samples = self.prng.multivariate_normal(self.mean, self.covariance, size=M)

		# the log likelihood of every sample is computed
		self.log_likelihoods = util.compute_loglikelihoods(self.samples, obs, self.mixture_coefficients, self.variance)

		# clipping
		self.i_clipped = np.argpartition(self.log_likelihoods, -self.M_T)[-self.M_T:]

		self.resample_and_update_proposal()

		# return self.weights @ self.samples


class MyNonlinearPopulationMonteCarlo(NonlinearPopulationMonteCarlo):

	def resample_and_update_proposal(self):

		# scaling followed by exponentiation to drop the logarithm and get the *unclipped* (log-less) likelihoods
		unclipped_likelihoods = np.exp(self.log_likelihoods - self.log_likelihoods.max())

		# minimum (unnormalized) weight among those to be clipped
		clipping_threshold = self.log_likelihoods[self.i_clipped[0]]

		# the likelihoods with the largest (unnormalized) weights are "clipped"
		self.log_likelihoods[self.i_clipped] = clipping_threshold

		# *clipped* likelihoods
		likelihoods = np.exp(self.log_likelihoods - self.log_likelihoods.max())

		# *clipped* weights
		weights = likelihoods / likelihoods.sum()

		i_resampled = self.resampling_alg.get_indexes(weights)

		# covariance is computed using clipping
		self.covariance = np.cov(self.samples[i_resampled, :].T, ddof=0)

		# the *unclipped* weights...
		self.weights = unclipped_likelihoods / unclipped_likelihoods.sum()

		# ...are used for the *actual* resampling
		i_resampled = self.resampling_alg.get_indexes(self.weights)
		self.samples = self.samples[i_resampled, :]

		# mean is computed without resorting to clipping
		self.mean = self.samples.mean(axis=0)


# ----------

mean = np.full(n_mixture_components, proposal_mean, dtype=float)
covariance = np.diag(np.full(n_mixture_components, proposal_sd))

npmc = NonlinearPopulationMonteCarlo(mean, covariance, M, M_T, mixture_coefficients, variance, prng)
my_npmc = MyNonlinearPopulationMonteCarlo(mean, covariance, M, M_T, mixture_coefficients, variance, prng)

n_trials = 1
n_monte_carlo_trials = 1

# algorithms to be run
algorithms = [npmc, my_npmc]

for i_trial in range(n_trials):

	observations = gmm.sample(n_samples=N, random_state=prng).flatten()

	for i_monte_carlo_trial in range(n_monte_carlo_trials):

		for i_pmc_iteration in range(n_pmc_iterations):

			print(
				'trial ' + colorama.Fore.LIGHTWHITE_EX + '{}'.format(i_trial) + colorama.Style.RESET_ALL + ' | ' +
				'MC trial ' + colorama.Fore.LIGHTGREEN_EX + '{}'.format(i_monte_carlo_trial) + colorama.Style.RESET_ALL + ' | ' +
				'population trial ' + colorama.Fore.LIGHTBLUE_EX + '{}'.format(i_pmc_iteration) + colorama.Style.RESET_ALL
			)

			for i_alg, alg in enumerate(algorithms):

				alg.iterate(observations)

				estimates[i_trial, i_monte_carlo_trial, :, i_pmc_iteration, i_alg] = alg.weights @ alg.samples
				M_eff[i_trial, i_monte_carlo_trial, i_pmc_iteration, i_alg] = 1.0 / np.sum(alg.weights ** 2)
				max_weight[i_trial, i_monte_carlo_trial, i_pmc_iteration, i_alg] = alg.weights.max()

				print(estimates[i_trial, i_monte_carlo_trial, :, i_pmc_iteration, i_alg])

			print('========')

		# algorithms are reinitialized
		for alg in algorithms:

			alg.reset()

# --------------------- data saving

# output data file
output_file = manu.util.filename_from_host_and_date()

file = h5py.File('res_' + output_file + '.hdf5', 'w')

file.create_dataset('estimated means', shape=estimates.shape, data=estimates)
file.create_dataset('true means', shape=true_means.shape, data=true_means)
file.create_dataset('M_eff', shape=M_eff.shape, data=M_eff)
file.create_dataset('maximum weight', shape=max_weight.shape, data=max_weight)

file.attrs['M_T'] = M_T

if random_seed:

	file.attrs['random seed'] = random_seed

file.close()

# in a separate file with the same name as the data file but different extension...
parameters_file = 'res_{}.parameters'.format(output_file)

with open(parameters_file, mode='wb') as f:

	#  ...parameters are pickled
	pickle.dump(parameters, f)

print('parameters saved in "{}"'.format(parameters_file))