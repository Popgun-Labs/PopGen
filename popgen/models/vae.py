import torch
import torch.nn as nn
import torch.nn.functional as F

from popgen.nn.mlp import MLP_Encoder, MLP_Decoder
from popgen.nn.vamp_prior import VAMPPrior
from popgen.nn.flows import HouseholderSylvesterFlow
from popgen.distributions.gaussian import reparameterize, log_gaussian


class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior=None, posterior_flow=None, **kwargs):
        """
        :param encoder: encoder settings
        :param decoder: decoder settings
        :param prior: `None` for standard normal prior, OR dict of settings for VAMPPrior
        :param posterior_flow: `None` for diagonal gaussian posterior
            OR dict of settings for HouseholderSylvesterFlows
        :param kwargs: captures all common model settings (passed to each sub-module)
        """
        super().__init__()

        # posterior q(z|x)
        self.encoder = MLP_Encoder(**encoder, **kwargs)

        # (optional) posterior normalizing flow
        self.posterior_flow = None
        if posterior_flow is not None:
            self.posterior_flow = HouseholderSylvesterFlow(**posterior_flow, **kwargs)

        # bernoulli likelihood model p(x|z)
        self.decoder = MLP_Decoder(**decoder, **kwargs)

        # prior model p(z)
        self.prior = None
        if prior is not None:
            self.prior = VAMPPrior(nonlinearity=torch.sigmoid, **prior, **kwargs)
            self.prior.U.data.uniform_()

        # store reference to parameters for gradient clipping.
        self.params = [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        """
        :param x: (batch, input_dim)
        :return:
            logits: log pixel probabilities (batch, input_dim)
            likelihood: log likelihood p(x|z)
            log_q_z: posterior probability q(z|x)
            ln_det: volume transformation (zero if no flows)
        """

        # infer latents
        z_K, z_0, mu, logvar, ln_det = self.infer(x)

        # get the pixel log probabilities
        logits = self.decoder(z_K)

        # likelihood
        likelihood = self.likelihood(logits, x)

        # KL terms
        ln_det = ln_det.mean()
        log_p_z = self.log_p_z(z_K)
        log_q_z = self.log_q_z0(z_0, mu, logvar) - ln_det

        return logits, likelihood, log_p_z, log_q_z, ln_det

    def log_p_z(self, z, cache_params=False, reduce=True):
        """
        Probability of `z` under the prior.
        :param z: (batch, latent)
        :param cache_params: cache parameters of the prior
            for faster model evaluation
        :return: log likelihood
            if reduce: (1)
            else: (batch)
        """

        # standard gaussian prior p(z) = N(0, I)
        if self.prior is None:
            log_p_z = log_gaussian(z).sum(-1)
        # VAMP Prior
        else:
            if cache_params:
                if hasattr(self, "prior_params"):
                    prior_mus, prior_logvars = self.prior_params
                else:
                    U = self.prior.get_U()
                    prior_mus, prior_logvars, _h = self.encoder(U)
                    self.prior_params = prior_mus, prior_logvars
            else:
                U = self.prior.get_U()
                prior_mus, prior_logvars, _h = self.encoder(U)

            # compute probability under mixture
            log_p_z = self.prior(z, prior_mus, prior_logvars)

        if reduce:
            log_p_z = log_p_z.mean()

        return log_p_z

    def log_q_z0(self, z_0, mu, logvar, reduce=True):
        """
        Get the probability of z0 under the base posterior.
        :param z0: (batch, latent)
        :param mu: (batch, latent)
        :param logvar: (batch, latent)
        :return:
            if reduce: (1)
            else: (batch)
        """
        log_q_z0 = log_gaussian(z_0, mu, logvar).sum(-1)
        if reduce:
            log_q_z0 = log_q_z0.mean()
        return log_q_z0

    def likelihood(self, x_sample, x, reduce=True):
        """
        Log likelihood of sample p(x|z)
        :param x_sample: (batch, input_dim)
        :param x: (batch, input_dim)
        :return: log p(x|z)
            if reduce: (1)
            else: (batch)
        """
        log_likelihood = F.binary_cross_entropy_with_logits(x_sample, x, reduction="none")
        log_likelihood = log_likelihood.sum(-1)  # (batch)
        if reduce:
            log_likelihood = log_likelihood.mean()
        return log_likelihood

    def infer(self, x):
        """
        Infer latent parameters from `x`.
        :param x: (batch, input_dim)
        :return:
            z_K: transformed posterior sample (batch, latent)
            z_0: base posterior sample (batch, latent)
            mu: (batch, latent)
            logvar: (batch, latent)
            ln_det: (batch)
        """
        mu, logvar, h = self.encoder(x)
        z_0 = reparameterize(mu, logvar)
        z_K = z_0

        # if using flows, transform the base sample and account
        # for the volume change.
        ln_det = torch.zeros(mu.size(0), device=mu.device)
        if self.posterior_flow is not None:
            z_K, ln_det = self.posterior_flow(z_K, h)
            ln_det = ln_det.sum(-1)  # (batch)

        return z_K, z_0, mu, logvar, ln_det

    def sample_prior(self, batch_size=1, mode=None, mean_only=False):
        """
        Draw sample from the generative model's prior.
        :param batch_size:
        :param mode: which modes of the prior to sample from np.array (batch_size)
        :param mean_only: for vamp-prior, takes only the mean of the selected component
        :return: (batch_size, latent_dim)
        """

        # note: ensure `z` is on same device as model
        device = self.params[0].device

        # gaussian prior
        if self.prior is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)
            return z

        # get pseudo-embedding matrix
        U = self.prior.get_U()

        # map to latent params (nb_inputs, latent)
        mus, logvars, *_ = self.encoder(U)

        # sample the mode uniform randomly
        if mode is None:
            nb_pseudos = mus.size(0)
            uniform = torch.ones(batch_size, nb_pseudos, device=device)
            mode = torch.multinomial(uniform).squeeze(1)  # (batch)
        else:
            mode = torch.from_numpy(mode).long().to(device)

        # select modes
        mu = mus[mode]
        z = mu
        if not mean_only:
            logvar = logvars[mode]
            z = reparameterize(mu, logvar)

        return z
