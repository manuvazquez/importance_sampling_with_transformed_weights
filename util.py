import sys
import os
import pickle

import numpy as np
import h5py
import numba

sys.path.append(os.path.join(os.environ['HOME'], 'python'))
import manu.util


@numba.jit(numba.float64[:](numba.float64[:, :], numba.float64[:], numba.float64[:], numba.float64), nopython=True)
def compute_loglikelihoods(samples, obs, mixture_coefficients, variance):

	M = len(samples)
	N = len(obs)
	n_mixture_components = len(mixture_coefficients)

	res = np.empty(M)

	for i_sample in range(M):

		loglikelihood = 0.0

		for i_obs in range(N):

			addend = 0.0

			for i_component in range(n_mixture_components):

				addend += np.exp(
					-(obs[i_obs] - samples[i_sample, i_component]) ** 2 / (2 * variance)
				)/np.sqrt(2*np.pi*variance) * mixture_coefficients[i_component]

			loglikelihood += np.log(addend)

		res[i_sample] = loglikelihood

	return res


def save_data(parameters, estimates, true_means, M_eff, max_weight, random_seed, attributes):

	# output data file
	output_file = manu.util.filename_from_host_and_date()

	file = h5py.File('res_' + output_file + '.hdf5', 'w')

	file.create_dataset('estimated means', shape=estimates.shape, data=estimates)
	file.create_dataset('true means', shape=true_means.shape, data=true_means)
	file.create_dataset('M_eff', shape=M_eff.shape, data=M_eff)
	file.create_dataset('maximum weight', shape=max_weight.shape, data=max_weight)

	for k,v in attributes.items():

		file.attrs[k] = v

	if random_seed:

		file.attrs['random seed'] = random_seed

	file.close()

	# in a separate file with the same name as the data file but different extension...
	parameters_file = 'res_{}.parameters'.format(output_file)

	with open(parameters_file, mode='wb') as f:

		#  ...parameters are pickled
		pickle.dump(parameters, f)

	print('parameters saved in "{}"'.format(parameters_file))
