import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, nb_layers=2, deterministic=False, dropout_p=0.0):
        """
        A simple MLP encoder with gated activations.
        :param input_dim: input features
        :param hidden_dim: hidden features
        :param latent_dim: latent feature size OR number of parameters for the posterior_flow distribution
        :param nb_layers: excluding the output projection
        :param deterministic:
            True: return a single deterministic latent variable (e.g. for autoencoder, WAE, AAE)
            False: return parameters of a gaussian base posterior_flow, along with the final hidden representation,
                which can serve as context for a normalising flow.
        :param dropout_p:
        """
        super().__init__()

        self.deterministic = deterministic

        layers = []
        for i in range(nb_layers):
            input_dim = hidden_dim if i > 0 else input_dim
            layers += [nn.Linear(input_dim, hidden_dim * 2), nn.GLU(dim=1), nn.Dropout(dropout_p)]

        self.layers = nn.Sequential(*layers)

        # projection to latent OR posterior_flow parameters
        output_dim = latent_dim
        if not deterministic:
            output_dim *= 2
        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        :param x: (batch, input_dim)
        :return:
            if deterministic:
                z: (batch, latent_dim)
            else: (mu, logvar, h)
                mu: posterior_flow mean (batch, latent_dim)
                logvar: posterior_flow log-variance (batch, latent_dim)
                h: final hidden state / 'context' vector (batch, hidden_dim)
        """

        h = self.layers(x)  # (batch, hidden_dim)
        params = self.final(h)

        if self.deterministic:
            return params

        mu, logvar = torch.chunk(params, 2, dim=1)
        logvar = F.hardtanh(logvar, min_val=-6.0, max_val=2.0)

        return mu, logvar, h
