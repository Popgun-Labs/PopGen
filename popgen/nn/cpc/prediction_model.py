import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_


class PredictionModel(nn.Module):
    def __init__(self, latent_dim=512, context_dim=256, K=12, **kwargs):
        """
        The prediction model for Contrastive Predictive Coding (van den Oorde et al, 2019).

        Defines the score function `f_k(x[k+t], c[t])`, which models the ratio `p(x|c) / p(x)`.
        During training `f_k` is optimised to assign high scores to the true context-latent pairings,
        and low scores to negative samples. Negative samples are drawn from random sequence and batch indices.

        This module should be used in conjunction with:
        1. a latent encoder: any deep neural network with sequence data as an input.
        2. a context model: any autoregressive function of the latent variables

        Since the design of 1. and 2. is domain specific, they have been omitted from this library.

        To compute the noise-contrastive loss on the returned score matrix:
        ```
        scores = pred_model(context, latents)
        positive_scores = scores[:, 0]
        loss = positive_scores - torch.logsumexp(scores, dim=1, keepdim=False)
        loss = -loss.mean()
        ```

        TODO: provide example notebook

        References:
            - https://arxiv.org/pdf/1807.03748v1.pdf

        :param latent_dim: dimension of `z`
        :param context_dim: dimension of `c`
        :param K: number of steps to predict
        :return: (batch, batch+1, L-K, K)
        """
        super().__init__()

        # create a different projection matrix for each step `k`
        Wk = torch.zeros(K, latent_dim, context_dim)
        xavier_normal_(Wk)
        self.Wk = nn.Parameter(Wk)  # (k, latent, context)

    def negative_samples(self, z_windows):
        """
        Draw negative samples by permuting `z_windows` according to the
        negative sampling strategy
        :param z_windows: (batch, L-K, K, latent_dim)
        :return: (batch, N, L-K, K, latent_dim)
            where N=0 corresponds to the positive samples
            N=1 onwards corresponds to the negative samples
        """

        B, L, *_ = z_windows.size()

        # draw `batch` negative samples from different indices of the same sequence
        z_shuffle = []
        for i in range(B):
            L_idx = torch.randperm(L, device=z_windows.device)
            z_shuffle.append(z_windows[:, L_idx])
        z_shuffle_length = torch.stack(z_shuffle, dim=1)  # (batch, -ve samples, L-K, K, latent_dim)

        # generate random negative samples from any random tensor position, first by shuffling on the length dimension,
        # and then taking a random permutation of the batch dimension.
        L_idx = torch.randperm(L, device=z_windows.device)
        B_idx = torch.randperm(B, device=z_windows.device)

        # shuffle z_windows randomly on the length dimension
        z_shuffle = z_windows[B_idx][:, L_idx]  # (-ve samples, L-K, K, latent_dim)
        z_shuffle = z_shuffle[
            None,
        ]  # (1, -ve samples, L-K, K, latent_dim)

        # combine the length-shuffled and fully shuffled samples
        z_shuffle = torch.cat([z_shuffle.expand_as(z_shuffle_length), z_shuffle_length], dim=1)

        return z_shuffle

    def forward(self, c, z):
        """
        :param c: context vectors. compute as an autoregressive function of the latents. (batch, length, context_dim)
        :param z: latents. (batch, length, latent_dim)
        """

        # take `k` projections of each context vector
        Wk = self.Wk[None, None, :, :, :]  # (1, 1, k, latent_dim, context_dim)
        c = c[:, :, None, :, None]  # (batch, length, 1, context_dim, 1)
        c_proj = (Wk @ c).squeeze(-1)  # (batch, length, K, latent_dim)

        # drop the final `k` context vectors (on the length dim), since there
        # is insufficient trailing `z` values to predict
        K = c_proj.size(2)
        c_proj = c_proj[:, :-K]  # (batch, L-K, K, latent_dim)

        # extract all windows of length `K` from the latents
        # the first window is dropped (no corresponding context)
        z_windows = z.unfold(1, K, 1)  # (batch, L-K+1, latent_dim, K)
        z_windows = z_windows[:, 1:].transpose(2, 3)  # (batch, L-K, K, latent_dim)

        # `z_windows` explanation:
        # every index on dimension 1 has a corresponding context vector, whose `K`
        # projections will be optimised to match the entries along dimension 2

        # compute positive samples by taking dot product of corresponding vectors
        # in `c_proj` and `z_windows`
        pos_scores = (c_proj * z_windows).sum(-1)  # (batch, L-K, K)

        # draw negative samples
        z_shuffle = self.negative_samples(z_windows)
        c_proj = c_proj[:, None]  # (batch, 1, L-K, K, latent_dim)

        # generate negative scores
        neg_scores = (c_proj * z_shuffle).sum(-1)  # (batch, -ve samples, L-K, K, latent_dim)

        # stack all scores on dim=1
        scores = torch.cat([pos_scores[:, None], neg_scores], dim=1)  # (batch, N, L-K, K)

        # `scores` explanation:
        # for each batch item in `scores`, we have `N` sets of score matrices
        # the N=0 score matrix contains the correct pairings of each `L-K` context vector with
        # the subsequent `K` latent vectors. the final `N-1` score matrices contain negative examples

        return scores
