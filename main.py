import numpy as np
from sklearn.mixture import GMM

# --------------------- general parameters

# number of trials
n_trials = 100

# mixture coefficient
ro = 0.2

# variance of both Gaussians (known)
variance = 1

# a vector with the *true* true_means of the Gaussians
true_means = np.array([0, 2])

# number of observations
N = 100

# --------------------- PMC

# number of particles
M = 10000

# hyperparameters for the prior pdf
nu, lamb = 1, 0.1

# ratio between the number of particles and that of non-negligible weights
M_T_over_M = 0.1

# ---------------------

# number of highest weights for the clipping procedure
M_T = int(M*M_T_over_M)

estimates = np.empty((n_trials, 2, 2))

# the proposal is a Gaussian density with parameters...
proposal_mean, proposal_sd = nu, np.sqrt(variance/lamb)

# pseudo-random numbers generator
prng = np.random.RandomState(123456)

# a Gaussian Mixture Model is built...
gmm = GMM(n_components=2, random_state=prng, n_iter=1)

# ...and the required parameters set up
gmm.means_ = np.reshape(true_means, (-1, 1))
gmm.covars_ = np.full(shape=(2,1),fill_value=variance, dtype=float)
gmm.weights_ = np.array([ro, 1-ro])

for i_trial in range(n_trials):

	observations = gmm.sample(n_samples=N, random_state=prng).flatten()

	# samples are drawn for every particle *and* every component
	samples = prng.normal(loc=proposal_mean, scale=proposal_sd, size=(M, 2))

	# computation of the factor every *individual* observations contributes to the likelihood
	# (every row is associated with an observation, and every column with a particle)
	likelihood_factors = ro*np.exp(-(observations[:, np.newaxis]-samples[:, 0][np.newaxis, :])**2
	                               /(2*variance))/np.sqrt(2*np.pi*variance) + \
	                     (1.0-ro)*np.exp(-(observations[:, np.newaxis]-samples[:, 1][np.newaxis, :])**2
	                                     /(2*variance))/np.sqrt(2*np.pi*variance)

	# the likelihood is given by the product of the individual factors
	likelihood = likelihood_factors.prod(axis=0)

	# weights are obtained by normalizing the values above
	weights = likelihood / likelihood.sum()

	# regular IW-based estimate
	estimates[i_trial, :, 0] = weights @ samples

	# --------------------- transformed importance weights

	# NOTE: *likelihood* is modified

	# clipping
	i_clipped = np.argpartition(likelihood, -M_T)[-M_T:]

	# minimum weight among those to be clipped
	clipping_threshold = likelihood[i_clipped[0]]

	# the largest weights are "clipped"
	likelihood[i_clipped] = clipping_threshold

	# weights are obtained by normalizing the values above
	weights = likelihood / likelihood.sum()

	#  TIW-based estimate
	estimates[i_trial, :, 1] = weights @ samples

# MMSE computation (every row is a trial, every column a component of the state vector)
mmse = np.sum((estimates - true_means[np.newaxis, :, np.newaxis])**2, axis=1)

average_mmse = mmse.mean(axis=0)

# variance (every row is an *algorithm*, every column a component of the state vector)
variance = np.var(estimates, axis=0).T