import torch
from .gaussian import log_gaussian


def log_gauss_mix(x, mu, logvar, log_prior):
    """
    Returns the log density of `x` under the parameterized gaussian mixture.
    :param x: (batch, dimension)
    :param mu: the mean of each component (nb_components, dimension)
    :param logvar: the log-variance of each component (nb_components, dimension)
    :param log_prior: the mixing weight / log-prior probability of each component (nb_components)
        note: expected to be normalised. i.e. logsumexp(log_prior) = 0
    :return: (batch)
    """

    # add singleton dimensions for broadcasting
    x = x.unsqueeze(0)  # (1, batch, dimension)
    mu = mu.unsqueeze(1)  # (nb_components, 1, dimension)
    logvar = logvar.unsqueeze(1)  # (nb_components, 1, dimension)
    log_prior = log_prior.unsqueeze(1)  # (nb_components, 1)

    # get likelihoods
    p = log_gaussian(x, mu, logvar)

    # add prior
    p = p + log_prior

    # sum over component dimension (0)
    p = torch.logsumexp(p, dim=0, keepdim=False)

    return p
