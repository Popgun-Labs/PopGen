import torch
import torch.nn as nn
import torch.nn.functional as F


class Jitter(nn.Module):
    def __init__(self, p=0.12):
        """
        Applies the jitter regularisation, suggested for VQ-VAE regularisation in https://arxiv.org/pdf/1901.08810.pdf
        :param p: jitter probability
        """
        super().__init__()

        self.p = p

    @torch.no_grad()
    def sample_copy_mask(self, x):
        _, length, _ = x.size()
        copy_probs = x.new_full((1, length, 1), self.p)
        copy_mask = torch.bernoulli(copy_probs).expand_as(x).byte()
        return copy_mask

    def forward(self, x):
        """
        :param x: (batch, length, features)
        """

        if not self.training or self.p == 0:
            return x

        # sample copy masks
        replace_with_left = self.sample_copy_mask(x)
        replace_with_left[:, 0] = 0.0  # prevent left-most item being swapped for an item on the left
        replace_with_right = self.sample_copy_mask(x)
        replace_with_right[:, -1] = 0.0

        # create two copies of the input
        x_left = F.pad(x, (0, 0, 1, 0))[:, :-1]
        x_right = F.pad(x, (0, 0, 0, 1))[:, 1:]

        # apply jitter
        x = torch.where(replace_with_left, x_left, x)
        x = torch.where(replace_with_right, x_right, x)

        return x
