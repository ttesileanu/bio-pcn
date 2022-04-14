""" Implement the predictive-coding network from Whittington & Bogacz. """

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

        self.x = [None for _ in self.dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Do a forward pass with unconstrained output.

        This sets each layer's variables to the most likely values given the previous
        layer values. This ends up being the same as a vanilla artificial neural net.

        :param x: input sample
        :returns: activation at output (last) layer
        """
        xs = [x]
        for i in range(len(self.dims) - 1):
            x = self.activation[i](x)
            x = x @ self.W[i].T + self.b[i]

            xs.append(x)

        self.x = xs
        return x

    def forward_constrained(self, x: torch.Tensor, y: torch.Tensor) -> Sequence:
        """ Do a forward pass where both input and output values are fixed.

        This runs a number of iterations (as set by `self.it_inference`) of the fast
        optimizer, starting with an initialization where the input is propagated forward
        without an output constraint (using `self.forward`).

        :param x: input sample
        :param y: output sample
        :return: loss evaluated before every optimization step
        """
        # start with a simple forward pass to initialize the layer values
        with torch.no_grad():
            self.forward(x)

        # fix the output layer values
        # noinspection PyTypeChecker
        self.x[-1] = y

        # ensure the variables in the hidden layers require grad
        for x in self.x[1:-1]:
            x.requires_grad = True

        # create an optimizer for the fast parameters
        fast_optimizer = torch.optim.SGD(self.fast_parameters(), lr=self.lr_inference)

        # ensure we're not calculating unneeded gradients
        # this improves speed by about 15% in the Whittington&Bogacz XOR example
        for W, b in zip(self.W, self.b):
            W.requires_grad = False
            b.requires_grad = False

        # iterate until convergence
        losses = np.zeros(self.it_inference)
        for i in range(self.it_inference):
            # this is about 10% faster than fast_optimizer.zero_grad()
            for param in self.fast_parameters():
                param.grad = None

            loss = self.loss()
            loss.backward()

            fast_optimizer.step()

            losses[i] = loss.item()

        # reset requires_grad
        for W, b in zip(self.W, self.b):
            W.requires_grad = True
            b.requires_grad = True

        return losses

    def loss(self) -> torch.Tensor:
        """ Calculate the loss given the current values of the random variables. """
        loss = torch.FloatTensor([0])
        x = self.x[0]
        for i in range(len(self.dims) - 1):
            x_pred = self.activation[i](x)
            x_pred = x_pred @ self.W[i].T + self.b[i]

            x = self.x[i + 1]
            # noinspection PyUnresolvedReferences
            loss += torch.sum((x - x_pred) ** 2) / self.variances[i]

        loss *= 0.5
        return loss

    def train(self):
        """ Set in training mode. """
        self.training = True

    def eval(self):
        """ Set in evaluation mode. """
        self.training = False

    def to(self, *args, **kwargs):
        """ Moves and/or casts the parameters and buffers. """
        for parameters in [self.slow_parameters(), self.fast_parameters()]:
            for param in parameters:
                param.to(*args, **kwargs)

    def slow_parameters(self) -> list:
        """ Create list of parameters to optimize in the slow phase.

        These are the weights and biases.
        """
        return self.W + self.b

    def fast_parameters(self) -> list:
        """ Create list of parameters to optimize in the fast phase.

        These are the random variables in all but the input and output layers.
        """
        return self.x[1:-1]
