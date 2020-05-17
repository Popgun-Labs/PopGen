import torch
import torch.nn as nn
import torch.nn.functional as F


class HouseholderSylvesterFlow(nn.Module):
    def __init__(self, latent_dim=32, K=8, H=8, hidden_dim=300, **kwargs):
        """
        Householder sylvester flow. (van den Berg et al, 2019)

        API Explanation:
            The final encoder hidden vector is passed to the HSNF instance. All parameters of the flow
            are computed as a linear projection of this hidden state. All invertibility constraints are satisfied
            internally.

        References:
            - Paper: https://arxiv.org/pdf/1803.05649.pdf
            - Author's implementation

        TODO: Add Notebook Example

        :param latent_dim: dimension of latent space
        :param K: number of flows (note: accuracy of ln_det seems to degrade after about 6 flows)
        :param H: number of householder reflections to create the orthogonal matrix
        :param hidden_dim: dimension of hidden state input
        :param verbose: print convergence info.
        :param kwargs:
        """
        super().__init__()

        self.K = K
        self.H = H
        L = latent_dim
        self.L = latent_dim

        # create linear projection from hidden -> flow params
        nb_params = (H * L) + 2 * (L ** 2) + L
        nb_params = nb_params * K
        print("{} flow params (excluding projection)".format(nb_params))
        self.linear = nn.Linear(hidden_dim, nb_params)

        # store identity matrix as buffer (1, 1, L, L)
        # note: doing it this way ensures correct GPU
        self.register_buffer('I', torch.eye(L).float().unsqueeze(0).unsqueeze(0))

    def _get_flow_params(self, h, eps=1e-2):
        """
        :param h: (batch, hidden)
        :return: (R, R_tilde, V, B)
            R: (K, B, L, L)
            R_tilde: (K, B, L, L)
            V: (K, B, H, L)
            B: (K, B, L)
        """

        # get dimensions
        K = self.K
        L = self.L
        H = self.H
        batch = h.size(0)

        # fail on batch == 1
        if batch == 1:
            raise Exception(
                "Batch size 1 is invalid for HSNF, due to PyTorch `triu` bug! Try `.repeat()` on the batch dimension.")

        # get raw params
        params = self.linear(h)

        # create R
        used = 0
        R_params = K * L * L
        R = params[:, used:R_params]  # (batch, R_params)
        R = R.contiguous().view(batch, K, L, L).transpose(0, 1)
        used += R_params

        # create R_tilde
        R_tilde_params = K * L * L
        R_tilde = params[:, used:used + R_tilde_params]  # (batch, R_params)
        R_tilde = R_tilde.contiguous().view(batch, K, L, L).transpose(0, 1)
        used += R_tilde_params

        # enforce tr(R_tilde) > 0 + eps, which is a sufficient condition for invertibility
        # of an upper triangular matrix (i.e no zero entries on diagonal)
        R_tilde_diag_o = R_tilde.diagonal(dim1=-2, dim2=-1)
        R_tilde_diag = R_tilde_diag_o * 1  # get a copy
        R_tilde_diag = F.softplus(R_tilde_diag) + eps
        R_tilde = R_tilde - R_tilde_diag_o.diag_embed(dim1=-2, dim2=-1) \
            + R_tilde_diag.diag_embed(dim1=-2, dim2=-1)

        # enforce R and R~ are upper triangular
        R = R.triu()
        R_tilde = R_tilde.triu()

        # create the matrix of householder vectors
        V_params = K * H * L
        V = params[:, used:used + V_params]  # (batch, R_params)
        V = V.contiguous().view(batch, K, H, L).transpose(0, 1)
        used += V_params

        # create B
        B_params = K * L
        B = params[:, used:used + B_params]  # (batch, R_params)
        B = B.contiguous().view(batch, K, L).transpose(0, 1)

        return R, R_tilde, V, B

    def forward(self, z_0, h):
        """
        :param z_0: latent sample from base posterior_flow (batch, latent_dim)
        :param h: flow context (batch, hidden_dim)
        :return: (z_K, ln_det)
            z_K: transformed sample (batch, latent_dim)
            ln_det: per-dimension log-var (batch, latent_dim)
        """
        z_k = z_0
        ln_det = 0.

        R, R_tilde, V, B = self._get_flow_params(h)

        # construct Q using householder reflections
        Q = self.I  # (1, 1, L, L)
        I = self.I
        for i in range(self.H):
            # extract vector for this reflection
            V_i = V[:, :, i, :]  # (K, B, 1, L)

            # compute norm and outer product
            V_i_outer = V_i.unsqueeze(-1) @ V_i.unsqueeze(-2)  # (K, B, L, L)
            V_i_norm = (V_i ** 2).sum(-1, keepdim=True).unsqueeze(-1)  # (K, B, 1, 1)

            # compute householder reflection
            H = I - 2 * (V_i_outer / V_i_norm)

            # update Q
            Q = Q @ H

        # Theorem 2: Invertibility constraint
        # ensure r_ii * r_tilde_ii is larger than -1 / h_prime_infinite
        # since 0 <= h_prime <= 1, it is sufficient to ensure r_ii * r_tilde_ii >= -1
        # note: see derivation for planar flows in appendix of https://arxiv.org/pdf/1505.05770.pdf
        R_tilde_diag = R_tilde.diagonal(dim1=-2, dim2=-1)
        R_diag = R.diagonal(dim1=-2, dim2=-1)
        R_o_diag = R_diag * 1.  # get a copy
        R_R_tilde = R_diag * R_tilde_diag
        M_R_R_tilde = -1 + F.softplus(R_R_tilde)
        R_diag = R_diag + (M_R_R_tilde - R_R_tilde) * (1. / R_tilde_diag)

        # overwrite diagonal in R
        R_o = R_o_diag.diag_embed(dim1=-2, dim2=-1)
        R_new = R_diag.diag_embed(dim1=-2, dim2=-1)
        R = R - R_o + R_new

        # compute all matrix products that don't depend on `z_k`
        # (K, B, M, M) @ (K, B, M, L) -> (K, B, M, L)
        W = R_tilde @ Q.transpose(2, 3)
        # (K, B, L, M) @ (K, B, M, M) -> (K, B, L, M)
        Q_r = Q @ R
        # (K, B, M, M) @ (K, B, M, M) -> (K, B, M, M)
        R_tilde_R = R_tilde @ R

        for i in range(self.K):
            # collect flow params
            r_tilde_r = R_tilde_R[i]  # (B, M, M)
            q_r = Q_r[i]  # (B, L, M)
            w = W[i]  # (B, M, L)
            b = B[i]  # (B, M)

            # Eq. (13)
            # (B, M, L) @ (B, L, 1) -> (B, M, 1)
            dot_product = w @ z_k.unsqueeze(2) + b.unsqueeze(2)

            # (B, L, M) @ (B, M, 1) -> (B, L, 1)
            flow = q_r @ torch.tanh(dot_product)
            z_k = z_k + flow.squeeze()

            # update ln_det
            ln_det = ln_det + self.ln_det(dot_product, r_tilde_r)

        return z_k, ln_det

    def h_prime(self, x):
        return 1. - torch.tanh(x) ** 2

    def ln_det(self, dot_product, r_tilde_r, eps=1e-8):
        """
        :param dot_product: (B, M, 1)
        :param r_tilde_r: (B, M, M)
        :return: (B, M)
        """
        # get identity matrix (1, M, M)
        I = self.I.squeeze(0)

        # get jacobian matrix
        h_prime = self.h_prime(dot_product)  # (B, M, 1)
        jacobian = I + (h_prime * r_tilde_r)  # (batch, M, M)

        # extract the diagonal
        diag = jacobian.diagonal(dim1=-2, dim2=-1)

        # compute determinant
        ln_det = diag.abs().add(eps).log()

        return ln_det
