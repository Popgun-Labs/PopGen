import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineCentroidLoss(nn.Module):
    def __init__(self, reduce=True):
        """
        Implementation of the metric learning loss described in https://arxiv.org/abs/1710.10467
        :param reduce:
        """
        super().__init__()

        self.reduce = reduce
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, x, w, b):
        """
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
