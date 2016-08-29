import numpy as np
from sklearn.mixture import GMM

# --------------------- general parameters

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

# number of highest weights for the clipping procedure
M_T = M//10

# ---------------------

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

observations = gmm.sample(n_samples=N, random_state=prng).flatten()

# a samples is drawn for every particle *and* every component
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
regular_estimate = weights @ samples

# MMSE
regular_mmse = np.sum((regular_estimate - true_means)**2)

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
tiw_estimate = weights @ samples

# performance
tiw_mmse = np.sum((tiw_estimate - true_means)**2)

# n_largest_weights = 10
#
# # the "n_largest_weights" higher weights
# i_largest_weights = np.argpartition(weights, -n_largest_weights)[-n_largest_weights:]
# weights[i_largest_weights]
# print('the {} best particles are\n{}'.format(n_largest_weights, samples[i_largest_weights, :]))