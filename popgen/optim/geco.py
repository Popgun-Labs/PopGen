import numpy as np


class GECO:
    def __init__(self, decay=0.99, prop=1e-6, initial_lagrange=1.0):
        """
        GECO for constrained optimisation

        References:
            - https://arxiv.org/pdf/1810.00597.pdf

        Example usage:
            TODO

        :param decay:
        """
        self.decay = decay
        self.t = 0
        self.lagrange = initial_lagrange
        self.prop = prop
        self.C_ma = None

    def update(self, C_batch):
        """
        :param C_batch: float value containing `loss - target_loss`
        :return:
        """

        # define moving average
        if self.t == 0:
            self.C_ma = C_batch
        else:
            self.C_ma = self.decay * self.C_ma + (1 - self.decay) * C_batch

        C_dif = self.C_ma - C_batch

        # compute Lagrange multiplier (essentially 1/beta)
        C = C_batch + C_dif

        self.lagrange = self.lagrange * np.exp(self.prop * C)
        self.t = self.t + 1

    def state_dict(self):
        return {"t": self.t, "lagrange": self.lagrange, "C_ma": self.C_ma}

    def load_state_dict(self, state_dict):
        self.t = state_dict["t"]
        self.lagrange = state_dict["lagrange"]
        self.C_ma = state_dict["C_ma"]
