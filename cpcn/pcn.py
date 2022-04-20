""" Implement the predictive-coding network from Whittington & Bogacz. """

from types import SimpleNamespace
from typing import Sequence, Union, Callable

import torch
import torch.nn as nn

import numpy as np


class PCNetwork(object):
    """ An implementation of the predictive coding network from Whittington&Bogacz. """

    def __init__(
        self,
        dims: Sequence,
        activation: Union[Sequence, Callable] = torch.tanh,
        it_inference: int = 100,
        lr_inference: float = 0.2,
        variances: Union[Sequence, float] = 1.0,
    ):
        """ Initialize the network.

        :param dims: number of units in each layer
        :param activation: activation function(s) to use for each layer
        :param it_inference: number of iterations per inference step
        :param lr_inference: learning rate for inference step
        :param variances: variance(s) to use for each layer after the first
        """
        self.training = True

        self.dims = np.copy(dims)
        self.activation = (
            (len(self.dims) - 1) * [activation]
            if not hasattr(activation, "__len__")
            else list(activation)
        )

        assert len(self.activation) == len(self.dims) - 1

        self.it_inference = it_inference
        self.lr_inference = lr_inference
        self.variances = torch.from_numpy(
            np.copy(variances)
            if np.size(variances) > 1
            else np.repeat(variances, len(self.dims) - 1)
        )

        assert len(self.variances) == len(self.dims) - 1

        # create and initialize the network parameters
        # weights and biases
        self.W = [
            torch.Tensor(self.dims[i + 1], self.dims[i])
            for i in range(len(self.dims) - 1)
        ]
        self.b = [
            torch.zeros(self.dims[i + 1], requires_grad=True)
            for i in range(len(self.dims) - 1)
        ]
        for W in self.W:
            nn.init.xavier_uniform_(W)
            W.requires_grad = True

        self.z = [torch.zeros(dim) for dim in self.dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Do a forward pass with unconstrained output.

        This sets each layer's variables to the most likely values given the previous
        layer values. This ends up being the same as a vanilla artificial neural net.

        :param x: input sample
        :returns: activation at output (last) layer
        """
        self.z[0] = x
        for i in range(len(self.dims) - 1):
            x = self.activation[i](x)
            x = x @ self.W[i].T + self.b[i]

            self.z[i + 1] = x

        return x

    def forward_constrained(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pc_loss_profile: bool = False,
        latent_profile: bool = False,
    ) -> SimpleNamespace:
        """ Do a forward pass where both input and output values are fixed.

        This runs a number of iterations (as set by `self.it_inference`) of the fast
        optimizer, starting with an initialization where the input is propagated forward
        without an output constraint (using `self.forward`).

        :param x: input sample
        :param y: output sample
        :param pc_loss_profile: if true, the evolution of the predictive-coding loss
            during the optimization is returned in the output namespace, under the name
            `pc_loss`
        :param latent_profile: if true, the evolution of the latent variables during the
            optimization is returned in the output namespace, under the name `latent.z`;
            the values are stored after each optimizer step, and they are stored as a
            list of tensors, one for each layer, of shape `[n_it, batch_size, n_units]`;
            note that this output will always have a batch index, even if the input and
            output samples do not
        :return: namespace with results; this is empty if both `loss_profile` and
            `latent_profile` are false
        """
        assert x.ndim == y.ndim
        if x.ndim > 1:
            assert x.shape[0] == y.shape[0]

        # start with a simple forward pass to initialize the layer values
        with torch.no_grad():
            self.forward(x)

        # fix the output layer values
        # noinspection PyTypeChecker
        self.z[-1] = y

        # ensure the variables in the hidden layers require grad
        for x in self.z[1:-1]:
            x.requires_grad = True

        # create an optimizer for the fast parameters
        fast_optimizer = torch.optim.SGD(self.fast_parameters(), lr=self.lr_inference)

        # ensure we're not calculating unneeded gradients
        # this improves speed by about 15% in the Whittington&Bogacz XOR example
        for W, b in zip(self.W, self.b):
            W.requires_grad = False
            b.requires_grad = False

        if latent_profile:
            batch_size = x.shape[0] if x.ndim > 1 else 1
            latent = [
                torch.zeros((self.it_inference, batch_size, dim)) for dim in self.dims
            ]

        # iterate until convergence
        losses = torch.zeros(self.it_inference)
        for i in range(self.it_inference):
            # this is about 10% faster than fast_optimizer.zero_grad()
            for param in self.fast_parameters():
                param.grad = None

            loss = self.loss()
            loss.backward()

            fast_optimizer.step()

            losses[i] = loss.item()

            if latent_profile:
                for k, crt_z in enumerate(self.z):
                    latent[k][i, :, :] = crt_z

        # reset requires_grad
        for W, b in zip(self.W, self.b):
            W.requires_grad = True
            b.requires_grad = True

        ns = SimpleNamespace()
        if pc_loss_profile:
            ns.pc_loss = losses
        if latent_profile:
            ns.latent = SimpleNamespace(z=latent)

        return ns

    def loss(self) -> torch.Tensor:
        """ Calculate the loss given the current values of the random variables. """
        loss = torch.FloatTensor([0])
        x = self.z[0]
        for i in range(len(self.dims) - 1):
            x_pred = self.activation[i](x)
            x_pred = x_pred @ self.W[i].T + self.b[i]

            x = self.z[i + 1]
            # noinspection PyUnresolvedReferences
            loss += torch.sum((x - x_pred) ** 2) / self.variances[i]

        loss *= 0.5
        return loss

    def pc_loss(self) -> torch.Tensor:
        """ An alias of `self.loss()`, for consistency with CPCN classes."""
        return self.loss()

    def calculate_weight_grad(self):
        """Calculate gradients for slow (weight) variables.

        This is equivalent to using `backward()` on the output from `self.loss()`
        (after zeroing all the gradients) and is only provided here for consistency with
        constrained predictive-coding networks, where the gradients are calculated
        manually.
        """
        for param in self.slow_parameters():
            param.grad = None

        loss = self.loss()
        loss.backward()

    def train(self):
        """ Set in training mode. """
        self.training = True

    def eval(self):
        """ Set in evaluation mode. """
        self.training = False

    def to(self, *args, **kwargs):
        """ Moves and/or casts the parameters and buffers. """
        with torch.no_grad():
            for i in range(len(self.W)):
                self.W[i] = self.W[i].to(*args, **kwargs).requires_grad_()
                self.b[i] = self.b[i].to(*args, **kwargs).requires_grad_()

            for i in range(len(self.z)):
                self.z[i] = self.z[i].to(*args, **kwargs)

        return self

    def slow_parameters(self) -> list:
        """ Create list of parameters to optimize in the slow phase.

        These are the weights and biases.
        """
        return self.W + self.b

    def fast_parameters(self) -> list:
        """ Create list of parameters to optimize in the fast phase.

        These are the random variables in all but the input and output layers.
        """
        return self.z[1:-1]

    def __str__(self) -> str:
        s = f"PCNetwork(dims={str(self.dims)}, activation={str(self.activation)})"
        return s

    def __repr__(self) -> str:
        s = (
            f"PCNetwork("
            f"dims={repr(self.dims)}, "
            f"activation={repr(self.activation)}, "
            f"it_inference={self.it_inference}, "
            f"lr_inference={self.lr_inference}, "
            f"variances={self.variances}"
            f")"
        )
        return s
