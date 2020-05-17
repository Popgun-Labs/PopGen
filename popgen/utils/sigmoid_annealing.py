import numpy as np


def sigmoid_annealing(iter_nb: int, temp: float = 1e-4) -> float:
    """
    Defines a sigmoid annealing schedule, such as the one proposed in:
    "Generating Sentences from a Continuous Space" (Bowman et. al, 2015)
    https://arxiv.org/abs/1511.06349

    :param iter_nb: the training iteration
    :param temp: rate of annealing (higher is faster), default converges in ~50k gradient updates.
    :return: a value in [0, 1], approaching 1. for larger values of `iter_nb`
    """
    sigmoid = 1. / (1. + np.exp(-iter_nb * temp))
    return (2 * (sigmoid - 0.5)).item()
