import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from popgen.distributions import log_gaussian
from functools import partial

# default non-linearity for pseudo-inputs
# note: used in original implementation, but has no gradients defined outside the clamping region.
# could be interesting to test smooth non-linearity instead (e.g. sigmoid)
hardtanh = partial(F.hardtanh, min_val=0.0, max_val=1.0)


class VAMPPrior(nn.Module):
    def __init__(self, input_dim, nb_components=32, nonlinearity=hardtanh, **kwargs):
        """
        Variational Mixture of Posteriors Prior (Tomczak, 2017)

        References:
            - Original paper https://arxiv.org/abs/1705.07120
            - Author's implementation https://github.com/jmtomczak/vae_vampprior

        Example usage:
            TODO

        :param input_dim: dimensionality of data
        :param nb_components: number of pseudo-inputs
        :param nonlinearity: optional non-linearity to constrain domain of pseudo inputs.
        :param kwargs:
        """
        super().__init__()

        # create the pseudo-inputs with random-normal initialisation
        self.nb_inputs = nb_components
        U = torch.empty(nb_components, input_dim).normal_(0.0, 0.1)
        self.U = nn.Parameter(U, requires_grad=True)

        # use a non-linearity to constrain the scale of pseudo-inputs
        self.nonlinearity = nonlinearity

    def get_U(self):
        """
        Get the matrix of pseudo-inputs, passed through optional non-linearity.
        :return: torch.Parameter (nb_components, input_dim)
        """
        if self.nonlinearity is None:
            return self.U

        return self.nonlinearity(self.U)

    def forward(self, x, mus, logvars, sum=True):
        """
        Compute the prior probabilities under a uniform gaussian mixture.
        Mixture parameters are defined by passing the matrix of learnable pseudo-inputs
        `U` through the posterior_flow encoder.

        For example:
        ```
        prior = VAMPPrior(...)
        U = prior.get_U()
        mu_prior, logvar_prior = encoder(U)
        log_p = prior(x, mu_prior, logvar_prior)
        ```

        :param x: data points (batch, latent_dim)
        :param mus: prior means (nb_components, latent_dim)
        :param logvars: prior log variances (nb_components, latent_dim)
        :param sum: whether to sum over the components
        :return:
            if sum=True: log probability under the mixture (batch)
            else: log probability of each point under each mixture component (nb_components, batch)
        """

        # note: `nb_components` refers to the number of gaussians / modes
        nb_inputs, latent_dim = mus.size()

        # get probability of each latent under each gaussian
        # (1, batch, latent_dim), (nb_components, 1, latent_dim) -> (nb_components, batch, latent_dim)
        log_ps = log_gaussian(x.unsqueeze(0), mus.unsqueeze(1), logvars.unsqueeze(1))
        log_ps = log_ps.sum(-1)  # (nb_components, batch)

        # add prior (uniform)
        log_ps = log_ps - math.log(nb_inputs)

        if sum:
            log_ps = torch.logsumexp(log_ps, dim=0, keepdim=False)

        return log_ps
