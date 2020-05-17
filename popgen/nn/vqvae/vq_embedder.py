import torch
from torch.autograd import Function


class VQEmbedder(Function):
    """
    Given an embedding matrix and a batch of latent vectors,
    each vector is snapped to the nearest entry in the embedding table. On the backwards
    pass the "straight-through" estimator is used (i.e. gradients pass as though no discretisation occurred).
    Ref: "Neural Discrete Representation Learning" https://arxiv.org/abs/1711.00937v2
    """
    @staticmethod
    def forward(ctx, z_e, codebook):
        """
        :param ctx:
        :param z_e: (batch, latent_dimension)
        :param codebook: (nb_codes, latent_dimension)
        :return (z_q, idxs)
            z_q: torch.FloatTensor snapped vectors (batch, latent_dimension)
            idxs: corresponding discrete codes torch.LongTensor (batch)
        """

        # get the l2 distances between each feature vector `z_e`,
        # and each entry in the codebook / embedding table
        # (1, nb_classes, features) - (batch, 1, features) -> (batch, nb_classes, features)
        diff = codebook.unsqueeze(0) - z_e.unsqueeze(1)
        l2_dist = torch.sum(diff ** 2, 2)

        # map each latent vector to the row-index of nearest-neighbor in the embedding table
        _, idxs = torch.min(l2_dist, 1)

        # map `z_e` -> `z_d` with embedding lookup
        z_d = codebook[idxs]
        return z_d, idxs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Straight-through estimation.
        """
        z_q_grad = grad_outputs[0]
        return z_q_grad, None