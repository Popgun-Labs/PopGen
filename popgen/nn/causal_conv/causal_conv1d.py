import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, groups=1, init=None):
        """
        Causal 1d convolutions with caching mechanism for O(L) generation,
        as described in the ByteNet paper (Kalchbrenner et al, 2016) and "Fast Wavenet" (Paine, 2016)

        Usage:
            At train time, API is same as regular convolution. `conv = CausalConv1d(...)`
            At inference time, set `conv.sequential = True` to enable activation caching, and feed
            sequence through step by step. Recurrent state is managed internally.

        References:
            - Neural Machine Translation in Linear Time: https://arxiv.org/abs/1610.10099
            - Fast Wavenet: https://arxiv.org/abs/1611.09482

        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param init: optional initialisation function for nn.Conv1d module (e.g xavier)
        """
        super(CausalConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups

        # if `true` enables fast generation
        self.sequential = False

        # compute required amount of padding to preserve the length
        self.zeros = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)

        # use supplied initialization function
        if init:
            init(self.conv)

    def forward(self, x):
        """
        :param x: (batch, in_channels, length)
        :return: (batch, out_channels, length)
        """
        # training mode
        if not self.sequential:
            # no padding for kw=1
            if self.kernel_size == 1:
                return self.conv(x)

            # left-pad + conv.
            out = F.pad(x, [self.zeros, 0])
            return self.conv(out)

        # sampling mode
        else:
            # note: x refers to a single timestep (batch, features, 1)
            if not hasattr(self, "recurrent_state"):
                batch_size = x.size(0)
                self._init_recurrent_state(batch_size)

            return self._generate(x)

    def clear_cache(self):
        """
        Delete the recurrent state. Note: this should be called between runs, to prevent
        leftover state bleeding into future samples. Note that we delete state (instead of zeroing) to support
        changes in the inference time batch size.
        """
        if hasattr(self, "recurrent_state"):
            del self.recurrent_state

    def _init_recurrent_state(self, batch):
        """
        Initialize the recurrent state for fast generation.
        :param batch: the batch size to generate
        """

        # extract weights and biases from nn.Conv1d module
        state = self.conv.state_dict()
        self.weight = state["weight"]
        self.bias = state["bias"]

        # initialize the recurrent states to zeros
        self.recurrent_state = torch.zeros(batch, self.in_channels, self.zeros, device=self.bias.device)

    def _generate(self, x_i):
        """
        Generate a single output activations, from the input activation
        and the cached recurrent state activations from previous steps.

        :param x_i: features of a single timestep (batch, in_channels, 1)
        :return: the next output value in the series (batch, out_channels, 1)
        """

        # if the kernel_size is greater than 1, use recurrent state.
        if self.kernel_size > 1:
            # extract the recurrent state and concat with input column
            recurrent_activations = self.recurrent_state[:, :, : self.zeros]
            f = torch.cat([recurrent_activations, x_i], 2)

            # update the cache for this layer
            self.recurrent_state = torch.cat([self.recurrent_state[:, :, 1:], x_i], 2)
        else:
            f = x_i

        # perform convolution
        activations = F.conv1d(f, self.weight, self.bias, dilation=self.dilation, groups=self.groups)

        return activations
