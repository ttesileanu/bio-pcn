""" Define the linear constrained-predictive coding network. """

from typing import Sequence, Union, Optional

import torch

import numpy as np


class LinearCPCNetwork:
    """Linear constrained-predictive coding network."""

    def __init__(
        self,
        pyr_dims: Sequence,
        inter_dims: Optional[Sequence] = None,
        z_it: int = 100,
        z_lr: float = 0.2,
        g_a: Union[Sequence, float] = 1.0,
        g_b: Union[Sequence, float] = 1.0,
        l_s: Union[Sequence, float] = 1.0,
        c_m: Union[Sequence, float] = 1.0,
        bias_a: bool = True,
        bias_b: bool = True,
    ):
        """Initialize the network.

        :param pyr_dims: number of pyramidal neurons in each layer
        :param inter_dims: number of interneurons per hidden layer; default: same as
            `pyr_dims[1:-1]`
        :param z_it: maximum number of iterations for fast (z) dynamics
        :param z_lr: starting learning rate for fast (z) dynamics
        :param g_a: apical conductances
        :param g_b: basal conductances
        :param l_s: leak conductance
        :param c_m: strength of lateral connections
        :param bias_a: whether to have bias terms at the apical end
        :param bias_b: whether to have bias terms at the basal end
        """
        self.training = True

        self.pyr_dims = np.copy(pyr_dims)
        assert len(self.pyr_dims) > 2

        if inter_dims is None:
            inter_dims = self.pyr_dims[1:-1]
        self.inter_dims = np.copy(inter_dims)

        assert len(self.pyr_dims) == len(self.inter_dims) + 2

        self.z_it = z_it
        self.z_lr = z_lr

        self.g_a = self._expand_per_layer(g_a)
        self.g_b = self._expand_per_layer(g_b)
        self.l_s = self._expand_per_layer(l_s)
        self.c_m = self._expand_per_layer(c_m)

        self.bias_a = bias_a
        self.bias_b = bias_b

        # create network parameters
        # weights and biases
        self.W_a = []
        self.W_b = []
        self.h_a = [] if self.bias_a else None
        self.h_b = [] if self.bias_b else None
        self.Q = []
        self.M = []

        D = len(self.pyr_dims) - 2
        for i in range(D):
            self.W_a.append(torch.Tensor(self.pyr_dims[i + 2], self.pyr_dims[i + 1]))
            if self.bias_a:
                self.h_a.append(torch.Tensor(self.pyr_dims[i + 2]))

            self.W_b.append(torch.Tensor(self.pyr_dims[i + 1], self.pyr_dims[i]))
            if self.bias_b:
                self.h_b.append(torch.Tensor(self.pyr_dims[i + 1]))

            self.Q.append(torch.Tensor(self.inter_dims[i], self.pyr_dims[i + 1]))
            self.M.append(torch.Tensor(self.pyr_dims[i + 1], self.pyr_dims[i + 1]))

        # initialize weights and biases
        self._initialize_interlayer_weights()
        self._initialize_intralayer_weights()
        self._initialize_biases()

        # self.W = [
        #     torch.Tensor(self.dims[i + 1], self.dims[i])
        #     for i in range(len(self.dims) - 1)
        # ]
        # self.b = [
        #     torch.zeros(self.dims[i + 1], requires_grad=True)
        #     for i in range(len(self.dims) - 1)
        # ]
        # for W in self.W:
        #     nn.init.xavier_uniform_(W)
        #     W.requires_grad = True

        # create neural variables
        self.a = [None for _ in self.inter_dims]
        self.b = [None for _ in self.inter_dims]
        self.n = [None for _ in self.inter_dims]
        self.z = [None for _ in self.pyr_dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass with unconstrained output.

        This uses the basal weights and biases to propagate the input through the net
        up to the last hidden layer, then the apical weights to generate the output.

        :param x: input sample
        :return: activation at the output layer
        """
        z = [x]
        n = len(self.pyr_dims)
        D = n - 2
        for i in range(D):
            W = self.W_b[i]
            h = self.h_b[i] if self.bias_b else 0

            x = x @ W.T + h
            z.append(x)

        W = self.W_a[-1]
        h = self.h_a[-1] if self.bias_a else 0
        x = x @ W.T + h
        z.append(x)

        self.z = z
        return x

    def _initialize_interlayer_weights(self):
        pass

    def _initialize_intralayer_weights(self):
        pass

    def _initialize_biases(self):
        pass

    def _expand_per_layer(self, theta) -> torch.Tensor:
        """Expand a quantity to per-layer, if needed, and convert to tensor."""
        n = len(self.pyr_dims) - 1

        if np.size(theta) > 1:
            assert len(theta) == n
            theta = torch.from_numpy(theta)
        else:
            theta = theta * torch.ones(n)

        return theta
