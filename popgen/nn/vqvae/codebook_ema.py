import torch
import torch.nn as nn


class CodebookEMA(nn.Module):
    def __init__(self, nb_codes, embedding_dim, decay=0.99, epsilon=1e-5, batch_debias=False):
        """
        An implementation of the Exponential Moving Average code book from VQ-VAE. Described in
        Appendix A.1 of "Neural Discrete Representation Learning". This is a port of DeepMind's tensorflow
        implementation.

        Can be thought of as an online version of K-means, where each discrete code represents a cluster. The
        input batch `z_e` represents a sample of points in the embedding space and the
        corresponding quantised indices are the cluster assignments.

        References:
            paper: https://arxiv.org/pdf/1711.00937v2.pdf
            tensorflow: https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
            zero de-biasing trick: https://arxiv.org/abs/1412.6980

        Example usage:
            TODO

        :param nb_codes: number of discrete codes
        :param embedding_dim: dimension of vectors
        :param decay: moving average decay
        """
        super().__init__()

        self.nb_codes = nb_codes
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.batch_debias = batch_debias

        # the code vectors are defined as e = m / N where,
        # N = EMA of # points assigned to each code
        # m = EMA of assignment weights
        # see Appendix A.1 https://arxiv.org/pdf/1711.00937v2.pdf
        e = torch.empty(nb_codes, embedding_dim).normal_()
        m = torch.zeros(nb_codes, embedding_dim).float()
        N = torch.zeros(nb_codes, 1).float()
        t = torch.ones(1).long()

        self.register_buffer('e', e)
        self.register_buffer('m', m)
        self.register_buffer('N', N)
        self.register_buffer('t', t)

    def set_codebook(self, e):
        """
        :param e:
        :return:
        """
        self.e = e
        self.m.fill_(0.)
        self.N.fill_(0.)
        self.t.fill_(1)

    def get_codebook(self):
        return self.e

    @torch.no_grad()
    def update(self, z_e, idxs):
        """
        Update the moving average.
        :param z_e: Un-quantised encoder outputs. torch.FloatTensor (*, embedding_dim)
        :param idxs: Corresponding cluster assignments (i.e. discrete indices after quantisation)
            torch.LongTensor (*) in [0, nb_codes)
        """
        # ensure `z_e` is the correct size
        embedding_dim = self.embedding_dim
        assert z_e.size(-1) == embedding_dim, \
            "z_e must have embedding_dim={} features on the final dimension".format(embedding_dim)

        # extract params
        m = self.m
        N = self.N
        decay = self.decay
        t = self.t.item()

        # one-hot encode the indices / codes
        batch = z_e.size(0)
        one_hot = z_e.new(self.nb_codes, batch).fill_(0.)
        one_hot.scatter_(0, idxs.unsqueeze(0), 1)

        # compute the number of items assigned to each code
        n_i = one_hot.sum(1, keepdim=True)  # (nb_codes, 1)

        # update the cluster assignment moving average, and
        # correct for zero bias as per https://arxiv.org/pdf/1412.6980.pdf
        zero_bias_correction = 1. - (decay ** t)
        N_update = (N * decay) + n_i * (1. - decay)
        N_update_debiased = N_update / zero_bias_correction

        # Adjustment for batch sizes. Not explained in paper, but present in reference implementation.
        # Seems to works without?
        if self.batch_debias:
            n = N_update.sum()  # this is equal to the batch size / number of latents
            N_update_debiased = (N_update_debiased + self.epsilon) / (n + self.nb_codes * self.epsilon) * n

        # create a (nb_codes, emb_dim) matrix, where the i-th
        # row contain the sum of encoder outputs assigned to cluster `i`
        w = one_hot @ z_e
        m_update = (m * decay) + w * (1. - decay)
        m_update_debiased = m_update / zero_bias_correction

        # update state
        self.e = m_update_debiased / (N_update_debiased + self.epsilon)
        self.m = m_update
        self.N = N_update
        self.t += 1