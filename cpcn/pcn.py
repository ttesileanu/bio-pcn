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
        whitening: bool = False,
        rho: Union[Sequence, float] = 0.0,
        bias: bool = True,
        fast_optimizer: Callable = torch.optim.Adam,
    ):
        """ Initialize the network.

        :param dims: number of units in each layer
        :param activation: activation function(s) to use for each layer
        :param it_inference: number of iterations per inference step
        :param lr_inference: learning rate for inference step
        :param variances: variance(s) to use for each layer after the first
        :param whitening: if true, use a whitening *in*equality,
            cov_matrix(z) <= rho * identity_matrix
        :param rho: parameter(s) for the whitening inequality
        :param bias: whether to include a bias term
        :param fast_optimizer: constructor for the optimizer used for the fast dynamics
            in `forward_constrained`
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
        self.rho = torch.from_numpy(
            np.copy(rho) if np.size(rho) > 1 else np.repeat(rho, len(self.dims) - 1)
        )
        self.whitening = whitening
        self.bias = bias
        self.fast_optimizer = fast_optimizer

        assert len(self.variances) == len(self.dims) - 1

        # create and initialize the network parameters
        # weights and biases
        self.W = [
            torch.Tensor(self.dims[i + 1], self.dims[i])
            for i in range(len(self.dims) - 1)
        ]
        if self.bias:
            self.b = [
                torch.zeros(self.dims[i + 1], requires_grad=True)
                for i in range(len(self.dims) - 1)
            ]
        else:
            self.b = []
        for W in self.W:
            nn.init.xavier_uniform_(W)
            W.requires_grad = True

        if self.whitening:
            self.Q = [
                torch.Tensor(self.dims[i + 1], self.dims[i + 1])
                for i in range(len(self.dims) - 1)
            ]
            for Q in self.Q:
                nn.init.xavier_uniform_(Q)
                Q.requires_grad = True

        self.z = [torch.zeros(dim) for dim in self.dims]

    def forward(self, x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
        """ Do a forward pass with unconstrained output.

        This sets each layer's variables to the most likely values given the previous
        layer values. This ends up being the same as a vanilla artificial neural net.

        :param x: input sample
        :param inplace: whether to update the latent-state values in-place; if false,
            the values are returned instead of the last-layer activation
        :returns: list of layer activations after the forward pass; if `inplace` is
            true (the default), this is the same as `self.z`; if `inplace` is false, the
            returned activations will be the same as if `inplace` were true, but
            `self.z` will be untouched
        """
        if inplace:
            z = self.z
        else:
            z = [None for _ in self.z]

        z[0] = x
        for i in range(len(self.dims) - 1):
            x = self.activation[i](x)

            if self.bias:
                x = x @ self.W[i].T + self.b[i]
            else:
                x = x @ self.W[i].T

            z[i + 1] = x

        return z

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
        fast_optimizer = self.fast_optimizer(
            self.fast_parameters(), lr=self.lr_inference
        )

        # ensure we're not calculating unneeded gradients
        # this improves speed by about 15% in the Whittington&Bogacz XOR example
        if self.bias:
            for W, b in zip(self.W, self.b):
                W.requires_grad = False
                b.requires_grad = False
        else:
            for W in self.W:
                W.requires_grad = False

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
        if self.bias:
            for W, b in zip(self.W, self.b):
                W.requires_grad = True
                b.requires_grad = True
        else:
            for W in self.W:
                W.requires_grad = True

        ns = SimpleNamespace()
        if pc_loss_profile:
            ns.pc_loss = losses
        if latent_profile:
            ns.latent = SimpleNamespace(z=latent)

        return ns

    def loss(
        self, reduction: str = "mean", ignore_whitening: bool = False
    ) -> torch.Tensor:
        """ Calculate the loss given the current values of the random variables.
        
        :param reduction: reduction to apply to the output: `"none" | "mean" | "sum"`
        :param ignore_whitening: if true, a whitening term is not included even if
            `self.whitening` is true; this does nothing if `self.whitening` is false;
            note also that the constraint term should vanish when the whitening
            inequality is satisfied, so this parameter should not make a big difference
            after training has converged
        """
        x = self.z[0]

        batch_size = 1 if x.ndim == 1 else len(x)
        loss = torch.zeros(batch_size)

        for i in range(len(self.dims) - 1):
            x_pred = self.activation[i](x)
            if self.bias:
                x_pred = x_pred @ self.W[i].T + self.b[i]
            else:
                x_pred = x_pred @ self.W[i].T

            x = self.z[i + 1]
            # noinspection PyUnresolvedReferences
            loss += ((x - x_pred) ** 2).sum(dim=-1) / self.variances[i]

        if not ignore_whitening and self.whitening:
            batch_outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
            for i in range(len(self.dims) - 1):
                fz = self.activation[i](self.z[i + 1])
                cons = batch_outer(fz, fz) - self.rho[i] * torch.eye(fz.shape[-1])
                q_prod = self.Q[i].T @ self.Q[i]
                constraint0 = (q_prod @ cons).diagonal(dim1=-1, dim2=-2).sum(dim=-1)
                loss += constraint0 / self.variances[i]

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction != "none":
            raise ValueError("unknown reduction type")

        loss *= 0.5
        return loss

    def pc_loss(self) -> torch.Tensor:
        """ An alias of `self.loss()` with `ignore_whitening` set to true. This is
        mostly useful for consistency with CPCN classes.
        """
        return self.loss(ignore_whitening=True)

    def calculate_weight_grad(self, reduction: str = "mean"):
        """Calculate gradients for slow (weight) variables.

        This is equivalent to using `backward()` on the output from `self.loss()`
        (after zeroing all the gradients) *and flipping the sign for the gradient of
        the `Q` parameters* (if `self.whitening` is true). The latter is because the
        optimizing needs to maximize over `Q` while minimizing over all the other
        paramters.

        :param reduction: reduction to apply to the gradients: `"mean" | "sum"`
        """
        for param in self.slow_parameters():
            param.grad = None

        loss = self.loss(reduction=reduction)
        loss.backward()

        # need to flip the sign of the whitening Lagrange multipliers, if any
        if self.whitening:
            for Q in self.Q:
                Q.grad = -Q.grad

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
                if self.bias:
                    self.b[i] = self.b[i].to(*args, **kwargs).requires_grad_()

            for i in range(len(self.z)):
                self.z[i] = self.z[i].to(*args, **kwargs)

            if self.whitening:
                for i in range(len(self.Q)):
                    self.Q[i] = self.Q[i].to(*args, **kwargs).requires_grad_()

        return self

    def slow_parameters(self) -> list:
        """ Create list of parameters to optimize in the slow phase.

        These are the weights and biases.
        """
        params = self.W

        if self.bias:
            params.extend(self.b)
        if self.whitening:
            params.extend(self.Q)

        return params

    def fast_parameters(self) -> list:
        """ Create list of parameters to optimize in the fast phase.

        These are the random variables in all but the input and output layers.
        """
        return self.z[1:-1]

    def __str__(self) -> str:
        s = (
            f"PCNetwork("
            f"dims={str(self.dims)}, "
            f"activation={str(self.activation)}, "
            f"bias={str(self.bias)}"
            f")"
        )
        return s

    def __repr__(self) -> str:
        s = (
            f"PCNetwork("
            f"dims={repr(self.dims)}, "
            f"activation={repr(self.activation)}, "
            f"bias={str(self.bias)}, "
            f"it_inference={self.it_inference}, "
            f"lr_inference={self.lr_inference}, "
            f"variances={self.variances}"
            f")"
        )
        return s
