import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineCentroidLoss(nn.Module):
    def __init__(self, reduce=True, fast=False):
        """
        Implementation of the metric learning loss described in https://arxiv.org/abs/1710.10467
        :param reduce:
        """
        super().__init__()

        self.reduce = reduce
        self.fast = fast
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, *args, **kwargs):
        if self.fast:
            return self.fast_ge2e(*args, **kwargs)

        return self.stable_ge2e(*args, **kwargs)

    def stable_ge2e(self, x, w, b):
        """
        When computing positive similarities, excludes the positive match itself from the centroid
        calculation. This should be more stable, but requires some expensive matrix ops.

        :param x: L2 normalised embeddings (nb_classes, nb_samples, embedding_dim)
        :param w: a learnable weight (1)
        :param b: a learnable bias (1)
        :return: (nb_classes * nb_examples) OR (1)
        """
        nb_classes, nb_samples, emb_dim = x.shape

        # ensure positivity of the weight term
        w = F.softplus(w)

        # compute a set of `nb_samples` centroids, where each centroid excludes one sample
        # this corresponds to equation 9. in the paper, and helps to prevent trivial solutions.
        # (nb_samples, nb_classes, nb_samples, embedding_dim)
        pos_centroids = x.unsqueeze(0).repeat(nb_samples, 1, 1, 1)
        # (nb_samples, 1, nb_samples, 1)
        mask = 1.0 - torch.eye(nb_samples, nb_samples, device=w.device)[:, None, :, None]
        pos_centroids = (pos_centroids * mask).sum(2) / (nb_samples - 1.0)  # (nb_samples, nb_classes, emb_dim)

        # embed the positive centroids on the diagonal of the similarity matrix
        pos_centroids = pos_centroids.transpose(1, 2)  # (nb_samples, emb_dim, nb_classes)
        pos_centroids = torch.diag_embed(pos_centroids)  # (nb_samples, emb_dim, nb_classes, nb_classes)
        pos_centroids = pos_centroids.permute(2, 0, 3, 1)  # (nb_classes, nb_samples, nb_classes, emb_dim)

        # compute the negative centroids (mean of each class), which will form
        # all elements of the off-diagonal
        neg_centroids = x.mean(1, keepdim=False)[None, None, :, :]  # (1, 1, nb_classes, emb_dim)
        # (nb_classes, nb_samples, nb_classes, emb_dim)
        neg_centroids = neg_centroids.repeat(nb_classes, nb_samples, 1, 1)

        # create the centroids matrix, with the true cluster means on the off-diagonal,
        # and the adjusted positive centroids on the diagonal.
        # (nb_classes, 1, nb_classes, 1)
        mask = 1.0 - torch.eye(nb_classes, nb_classes, device=w.device)[:, None, :, None]
        centroids = (mask * neg_centroids) + pos_centroids

        # expand `x` for comparison with each centroid
        x = x[:, :, None, :]  # (nb_classes, nb_samples, 1, emb_dim)

        # compute cosine similarity between each embedding and centroid
        S = w * self.similarity(x, centroids) + b  # (nb_classes, nb_samples, nb_classes)
        S = F.log_softmax(S, dim=-1)  # (nb_classes, nb_samples, nb_classes)

        # extract diagonal entries of similarity matrix (j == k)
        Sji_j = S.diagonal(dim1=0, dim2=2).transpose(0, 1)  # (nb_classes, nb_samples)

        # normalize, by subtracting total energy assigned to all centroids
        Sji_k = torch.logsumexp(S, dim=-1, keepdim=False)
        loss = Sji_j - Sji_k

        # flip the sign and flatten into (nb_classes * nb_samples)
        loss = -loss.view(-1)

        if self.reduce:
            return loss.mean()

        return loss

    def fast_ge2e(self, x, w, b):
        """
        Computes GE2E using a single cluster centroid for each class.
        This is fast and easy to understand, but might be unstable or have some pathological solutions ?
        :param x: L2 normalised embeddings (nb_classes, nb_samples, embedding_dim)
        :param w: a learnable weight (1)
        :param b: a learnable bias (1)
        :return: (nb_classes * nb_examples) OR (1)
        """
        nb_classes, nb_samples, emb_dim = x.size()

        # ensure positivity of the weight term
        w = F.softplus(w)

        # compute centroid (mean of each class)
        centroids = x.mean(1).unsqueeze(0)  # 1 x nb_classes x emb_dim

        # compute cosine similarity between each embedding and centroid
        x = x.view(-1, 1, emb_dim)  # (nb_classes * nb_samples, 1, embedding_dim)
        S = w * self.similarity(x, centroids) + b  # (nb_classes * nb_samples, nb_classes)
        S = F.log_softmax(S, dim=-1)
        S = S.view(nb_classes, nb_samples, nb_classes)

        # extract diagonal entries of similarity matrix (j == k)
        Sji_j = S.diagonal(dim1=0, dim2=2).transpose(0, 1)  # (nb_classes, nb_samples)

        # normalize, by subtracting total energy assigned to all centroids
        Sji_k = torch.logsumexp(S, dim=-1, keepdim=False)
        loss = Sji_j - Sji_k

        # flip the sign and flatten into (nb_classes * nb_samples)
        loss = -loss.view(-1)

        if self.reduce:
            return loss.mean()

        return loss
