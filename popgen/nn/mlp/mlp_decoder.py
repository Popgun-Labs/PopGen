import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, input_dim=784, nb_layers=2, dropout_p=0.0):
        """
        A simple MLP encoder with gated activations.
        :param latent_dim: input features
        :param hidden_dim: hidden features
        :param input_dim: nb output features OR parameters for the likelihood distribution
        :param nb_layers: excluding the output projection
        :param dropout_p:
        """
        super().__init__()

        layers = []
        for i in range(nb_layers):
            inter_dim = latent_dim if i == 0 else hidden_dim
            layers += [nn.Linear(inter_dim, hidden_dim * 2), nn.GLU(dim=1), nn.Dropout(dropout_p)]

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (batch, input_dim)
        :return params: parameters of the posterior_flow latent distribution
        """
        params = self.layers(x)
        return params
