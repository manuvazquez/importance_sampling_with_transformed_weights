import numpy as np
import numba


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