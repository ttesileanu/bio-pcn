""" Define the linear constrained-predictive coding network. """

from typing import Sequence, Union, Optional, Callable

import torch
import torch.nn as nn

import numpy as np


class LinearCPCNetwork:
    """Linear constrained-predictive coding network."""

    def __init__(
        self,
        pyr_dims: Sequence,
        inter_dims: Optional[Sequence] = None,
        z_it: int = 100,
        z_lr: float = 0.01,
        g_a: Union[Sequence, float] = 1.0,
        g_b: Union[Sequence, float] = 1.0,
        l_s: Union[Sequence, float] = 1.0,
        c_m: Union[Sequence, float] = 1.0,
        bias_a: bool = True,
        bias_b: bool = True,
        fast_optimizer: Optional[Callable] = None,
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
        :param fast_optimizer: constructor for the optimizer used for the fast dynamics
            in `forward_constrained`; by default this is `SGD`
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

        if fast_optimizer is not None:
            self.fast_optimizer = fast_optimizer
        else:
            self.fast_optimizer = torch.optim.SGD

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

    def forward_constrained(
        self, x: torch.Tensor, y: torch.Tensor, loss_profile: bool = False
    ) -> Optional[Sequence]:
        """Do a forward pass where both input and output values are fixed.

        This runs a number of iterations (as set by `self.z_it`) of the fast optimizer,
        starting with an initialization where the input is propagated forward without an
        output constraint (using `self.forward`).

        :param x: input sample
        :param y: output sample
        :param loss_profile: if true, evaluates and returns the loss at every step; see
            `LinearCPCNetwork.loss`
        :return: if `loss_profile` is true, loss evaluated before every optimization
            step; otherwise, `None`
        """
        # we calculate gradients manually
        with torch.no_grad():
            # start with a simple forward pass to initialize the layer values
            self.forward(x)

            # fix the output layer values
            # noinspection PyTypeChecker
            self.z[-1] = y

            # create an optimizer for the fast parameters
            fast_optimizer = self.fast_optimizer(self.fast_parameters(), lr=self.z_lr)

            # iterate until convergence
            if loss_profile:
                losses = np.zeros(self.z_it)
            for i in range(self.z_it):
                self.calculate_currents()
                self.calculate_z_grad()
                fast_optimizer.step()

                if loss_profile:
                    losses[i] = self.loss().item()

        if loss_profile:
            return losses

    def calculate_weight_grad(self):
        """Calculate gradients for slow (weight) variables.

        This assumes that the fast variables have been calculated, using
        `forward_constrained`. The calculated gradients are assigned the `grad`
        attribute of each weight tensor.
        """
        D = len(self.pyr_dims) - 2
        for i in range(D):
            # apical
            pre = self.z[i + 1]
            if self.bias_a:
                post = self.z[i + 2] - self.h_a[i]
            else:
                post = self.z[i + 2]
            self.W_a[i].grad = self.g_a[i] * (self.W_a[i] - torch.outer(post, pre))

            # basal
            plateau = self.g_a[i] * self.a[i]
            hebbian_self = (self.l_s[i] - self.g_b[i]) * self.z[i + 1]
            hebbian_lateral = self.c_m[i] * self.M[i] @ self.z[i + 1]
            hebbian = hebbian_self + hebbian_lateral

            pre = self.z[i]
            post = plateau - hebbian
            self.W_b[i].grad = -torch.outer(post, pre)

            # inter
            pre = self.z[i + 1]
            post = self.n[i]
            self.Q[i].grad = self.g_a[i] * (self.Q[i] - torch.outer(post, pre))

            # lateral
            pre = self.z[i + 1]
            post = pre
            self.M[i].grad = self.c_m[i] * (self.M[i] - torch.outer(post, pre))

    def calculate_z_grad(self):
        """Calculate gradients for fast (z) variables.

        This assumes that the currents were calculated using `calculate_currents`. The
        calculated gradients are assigned to the `grad` attribute of each tensor in
        `self.z`.
        """
        D = len(self.pyr_dims) - 2
        for i in range(D):
            grad_apical = self.g_a[i] * self.a[i]
            grad_basal = self.g_b[i] * self.b[i]
            grad_lateral = self.c_m[i] * self.M[i] @ self.z[i + 1]
            grad_leak = self.l_s[i] * self.z[i + 1]

            self.z[i + 1].grad = grad_lateral + grad_leak - grad_apical - grad_basal

        self.z[0].grad = None
        self.z[-1].grad = None

    def calculate_currents(self):
        """Calculate apical, basal, and interneuron currents in all layers.

        Note that this relies on valid `z` values being available in `self.z`. This
        might require running `self.forward`.
        """
        D = len(self.pyr_dims) - 2
        for i in range(D):
            self.n[i] = self.Q[i] @ self.z[i + 1]

            if self.bias_b:
                self.b[i] = self.W_b[i] @ self.z[i] + self.h_b[i]
            else:
                self.b[i] = self.W_b[i] @ self.z[i]

            if self.bias_a:
                a_feedback = self.W_a[i].T @ (self.z[i + 2] - self.h_a[i])
            else:
                a_feedback = self.W_a[i].T @ self.z[i + 2]
            a_inter = self.Q[i].T @ self.n[i]
            self.a[i] = a_feedback - a_inter

    def loss(self) -> torch.Tensor:
        """Estimate loss given current activation values.

        This is defined as the predictive-coding loss with duplicated weights connecting
        the hidden layers. Schematically,

            loss = 0.5 * sum(g_a * (z[l + 1] - mu_a[l + 1]) ** 2 +
                             g_b * (z[l] - mu_b[l]) ** 2))

        where the sum is over the hidden layers, and the predictions `mu_a` and `mu_b`
        are calculated using the apical and basal weights and biases, respectively:

            mu_x[l + 1] = W_x[l] @ z[l] + h_x[l]

        with `x` either `a` or `b`.

        This loss is minimized whenever the predictive-coding loss is minimized. (That
        is, at the minimum, `W_a == W_b`.)
        """
        res = torch.FloatTensor([0])

        D = len(self.pyr_dims) - 2
        norm = torch.linalg.norm
        for i in range(D):
            mu_a = self.W_a[i] @ self.z[i + 1]
            if self.bias_a:
                mu_a += self.h_a[i]
            apical = self.g_a[i] * norm(self.z[i + 2] - mu_a) ** 2

            mu_b = self.W_b[i] @ self.z[i]
            if self.bias_b:
                mu_b += self.h_b[i]
            basal = self.g_b[i] * norm(self.z[i + 1] - mu_b) ** 2

            res += apical + basal

        res /= 2

        return res

    def fast_parameters(self) -> list:
        """Create list of parameters to optimize in the fast phase.

        These are the activations in all the hidden layers.
        """
        return self.z[1:-1]

    def _initialize_interlayer_weights(self):
        for lst in [self.W_a, self.W_b]:
            for W in lst:
                nn.init.xavier_uniform_(W)
                W.requires_grad = True

    def _initialize_intralayer_weights(self):
        for lst in [self.Q, self.M]:
            for W in lst:
                nn.init.xavier_uniform_(W)
                W.requires_grad = True

    def _initialize_biases(self):
        all = []
        if self.bias_a:
            all += self.h_a
        if self.bias_b:
            all += self.h_b

        for h in all:
            h.zero_()
            h.requires_grad = True

    def _expand_per_layer(self, theta) -> torch.Tensor:
        """Expand a quantity to per-layer, if needed, and convert to tensor."""
        D = len(self.pyr_dims) - 2

        if np.size(theta) > 1:
            assert len(theta) == D
            theta = torch.from_numpy(np.asarray(theta))
        else:
            theta = theta * torch.ones(D)

        return theta
