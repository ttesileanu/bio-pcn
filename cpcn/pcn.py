"""Implement the predictive-coding network from Whittington & Bogacz."""

from types import SimpleNamespace
from typing import Sequence, Union, Callable, Tuple

import torch
import torch.nn as nn

import numpy as np


# mapping from strings to activation functions and their derivatives
_zero = torch.FloatTensor([0.0])
_activation_map = {
    "none": (lambda _: _, lambda _: torch.ones_like(_)),
    "relu": (torch.relu, lambda _: torch.heaviside(_, _zero).to(_.device)),
    "tanh": (torch.tanh, lambda _: 1.0 / torch.cosh(_) ** 2),
}


class PCNetwork(object):
    """An implementation of the predictive coding network from Whittington&Bogacz."""

    def __init__(
        self,
        dims: Sequence,
        activation: Union[Sequence, Callable, str] = "tanh",
        z_it: int = 100,
        z_lr: float = 0.2,
        variances: Union[Sequence, float] = 1.0,
        constrained: bool = False,
        rho: Union[Sequence, float] = 0.0,
        bias: bool = True,
        fast_optimizer: Callable = torch.optim.Adam,
    ):
        """Initialize the network.

        :param dims: number of units in each layer
        :param activation: activation function(s) to use for each layer; they can be
            callables or, preferrably, one of the following strings: `"none"` for linear
            (no activation function); `"relu"`; `"tanh"`
        :param z_it: number of iterations per inference step
        :param z_lr: learning rate for inference step
        :param variances: variance(s) to use for each layer after the first
        :param constrained: if true, use an inequality constraint,
            cov_matrix(z) <= rho * identity_matrix
        :param rho: parameter(s) for the constraint
        :param bias: whether to include a bias term
        :param fast_optimizer: constructor for the optimizer used for the fast dynamics
            in `relax`
        """
        self.training = True

        self.dims = np.copy(dims)
        self.activation = (
            (len(self.dims) - 1) * [activation]
            if isinstance(activation, str) or not hasattr(activation, "__len__")
            else list(activation)
        )

        assert len(self.activation) == len(self.dims) - 1

        self.z_it = z_it
        self.z_lr = z_lr
        self.variances = self._extend(variances, len(self.dims) - 1)
        self.rho = self._extend(rho, len(self.dims) - 2)

        self.constrained = constrained
        self.bias = bias
        self.fast_optimizer = fast_optimizer

        # create and initialize the network parameters
        # weights and biases
        self.W = [
            torch.Tensor(self.dims[i + 1], self.dims[i])
            for i in range(len(self.dims) - 1)
        ]
        if self.bias:
            self.h = [
                torch.zeros(self.dims[i + 1], requires_grad=True)
                for i in range(len(self.dims) - 1)
            ]
        else:
            self.h = []
        for W in self.W:
            nn.init.xavier_uniform_(W)
            W.requires_grad = True

        if self.constrained:
            self.Q = [
                torch.Tensor(self.dims[i + 1], self.dims[i + 1])
                for i in range(len(self.dims) - 2)
            ]
            for Q in self.Q:
                nn.init.xavier_uniform_(Q)
                Q.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass with unconstrained output.

        This returns the layer activations in a setting where they are fully determined
        by the previous layer values, which is the minimal-loss solution if only the
        input layer is fixed.

        :param x: input sample
        :returns: list of layer activations after the forward pass
        """
        z = []
        z.append(x.detach())
        activation = self._get_activation_fcts()
        with torch.no_grad():
            for i in range(len(self.dims) - 1):
                x = activation[i](x)

                if self.bias:
                    x = x @ self.W[i].T + self.h[i]
                else:
                    x = x @ self.W[i].T

                z.append(x)

        return z

    def relax(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pc_loss_profile: bool = False,
        latent_profile: bool = False,
    ) -> SimpleNamespace:
        """Do a forward pass where both input and output values are fixed.

        This runs a number of iterations (as set by `self.z_it`) of the fast optimizer,
        starting with an initialization where the input is propagated forward without an
        output constraint (using `self.forward`).

        :param x: input sample
        :param y: output sample
        :param pc_loss_profile: if true, the evolution of the predictive-coding loss
            during the optimization is returned in the output namespace, under the name
            `profile.pc_loss`
        :param latent_profile: if true, the evolution of the latent variables during the
            optimization is returned in the output namespace, under `profile.z`; the
            values are stored after each optimizer step, and they are stored as a list
            of tensors, one for each layer, of shape `[n_it, batch_size, n_units]`; note
            that this output will always have a batch index, even if the input and
            output samples do not
        :return: namespace with results; this always contains the final layer
            activations, in a list called `z`; it also contains a `profile` member,
            which is either empty or populated as described above when discussing the
            `..._profile` arguments; unlike the `latent_profile` described above, the
            final `z` values obey the batch conventions from `x` and `y`: i.e., they
            only have a batch index if `x` and `y` do
        """
        assert x.ndim == y.ndim
        if x.ndim > 1:
            assert x.shape[0] == y.shape[0]

        # start with a simple forward pass to initialize the layer values
        z = self.forward(x)

        # fix the output layer values
        z[-1] = y.detach()

        # create an optimizer for the fast parameters
        fast_optimizer = self.fast_optimizer(z[1:-1], lr=self.z_lr)

        # create storage for output
        if latent_profile:
            batch_size = x.shape[0] if x.ndim > 1 else 1
            latent = [torch.zeros((self.z_it, batch_size, dim)) for dim in self.dims]
        if pc_loss_profile:
            losses = torch.zeros(self.z_it)

        # iterate
        for i in range(self.z_it):
            self.calculate_z_grad(z)

            if pc_loss_profile:
                with torch.no_grad():
                    losses[i] = self.pc_loss(z).item()

            fast_optimizer.step()

            if latent_profile:
                for k, crt_z in enumerate(z):
                    latent[k][i, :, :] = crt_z

        ns = SimpleNamespace(z=z, profile=SimpleNamespace())
        if pc_loss_profile:
            ns.profile.pc_loss = losses
        if latent_profile:
            ns.profile.z = latent

        return ns

    def loss(
        self, z: Sequence, reduction: str = "mean", ignore_constraint: bool = False
    ) -> torch.Tensor:
        """Calculate the loss given the current values of the latent variables.
        
        :param z: list of latent activations, one tensor per layer
        :param reduction: reduction to apply to the output: `"none" | "mean" | "sum"`
        :param ignore_constraint: if true, the constraint term is not included even if
            `self.constrained` is true; it does nothing if `self.constrained` is false;
            note also that the constraint term should vanish when the inequality is
            satisfied, so this parameter should not make a significant difference after
            training has converged
        """
        x = z[0]

        batch_size = 1 if x.ndim == 1 else len(x)
        loss = torch.zeros(batch_size).to(x.device)

        activation = self._get_activation_fcts()
        for i in range(len(self.dims) - 1):
            x_pred = activation[i](x)

            if i > 0 and not ignore_constraint and self.constrained:
                Q = self.Q[i - 1]
                qz = x_pred @ Q.T
                cons_z = (qz ** 2).sum(dim=-1)
                cons_q = self.rho[i - 1] * (Q ** 2).sum()
                constraint0 = cons_z - cons_q
                loss += constraint0 / self.variances[i - 1]

            if self.bias:
                x_pred = x_pred @ self.W[i].T + self.h[i]
            else:
                x_pred = x_pred @ self.W[i].T

            x = z[i + 1]
            loss += ((x - x_pred) ** 2).sum(dim=-1) / self.variances[i]

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction != "none":
            raise ValueError("unknown reduction type")

        loss *= 0.5
        return loss

    def pc_loss(self, z: Sequence) -> torch.Tensor:
        """An alias of `self.loss()` with `ignore_constraint` set to true. This is
        mostly useful for consistency with CPCN classes.
        """
        return self.loss(z, ignore_constraint=True)

    def calculate_z_grad(self, z: Sequence):
        """Calculate gradients for fast (z) variables.
        
        These gradients follow from backprop on `self.loss()`, but this manual
        implementation is more consistent with what we do for BioPCN, and is also
        significantly faster in this case.
        """
        # calculate activations and derivatives
        fz = []
        fz_der = []
        activation, der_activation = self._get_activation_fcts_and_ders()
        if activation is None:
            # need to calculate derivatives using autograd
            with torch.enable_grad():
                activation = self._get_activation_fcts()
                for i in range(len(self.dims) - 1):
                    f = activation[i]

                    crt_z = z[i].detach().requires_grad_()
                    crt_fz = f(crt_z)
                    crt_fz_der = torch.autograd.grad(
                        crt_fz,
                        crt_z,
                        grad_outputs=torch.ones_like(crt_z),
                        create_graph=True,
                    )[0]

                    fz.append(crt_fz.detach())
                    fz_der.append(crt_fz_der.detach())
        else:
            for i in range(len(self.dims) - 1):
                crt_z = z[i].detach()
                fz.append(activation[i](crt_z))
                fz_der.append(der_activation[i](crt_z))

        # calculate error nodes
        with torch.no_grad():
            eps = []
            for i in range(1, len(self.dims)):
                mu = fz[i - 1] @ self.W[i - 1].T
                if self.bias:
                    mu += self.h[i - 1]

                eps.append((z[i].detach() - mu) / self.variances[i - 1])

            # calculate the gradients
            for i in range(1, len(self.dims) - 1):
                grad0 = eps[i - 1] - fz_der[i] * (eps[i] @ self.W[i])
                if self.constrained:
                    v = self.variances[i - 1]
                    grad0 += fz_der[i] * (fz[i] @ self.Q[i - 1].T @ self.Q[i - 1]) / v

                if grad0.ndim == z[i].ndim + 1:
                    grad0 = grad0.mean(dim=0)
                z[i].grad = grad0

    def calculate_weight_grad(self, z: Sequence, reduction: str = "mean"):
        """Calculate gradients for slow (weight) variables.

        This is equivalent to using `backward()` on the output from `self.loss()`
        (after zeroing all the gradients) *and flipping the sign for the gradient of
        the `Q` parameters* (if `self.constrained` is true). The latter is because the
        optimization needs to maximize over `Q` while minimizing over all the other
        paramters.

        :param z: values of latent variables in each layer
        :param reduction: reduction to apply to the gradients: `"mean" | "sum"`
        """
        for param in self.slow_parameters():
            param.grad = None

        loss = self.loss(z, reduction=reduction)
        loss.backward()

        # need to flip the sign of the Lagrange multipliers, if any
        if self.constrained:
            for Q in self.Q:
                Q.grad = -Q.grad

    def train(self):
        """Set in training mode."""
        self.training = True

    def eval(self):
        """Set in evaluation mode."""
        self.training = False

    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers."""
        with torch.no_grad():
            for i in range(len(self.W)):
                self.W[i] = self.W[i].to(*args, **kwargs).detach().requires_grad_()
                if self.bias:
                    self.h[i] = self.h[i].to(*args, **kwargs).detach().requires_grad_()

            if self.constrained:
                for i in range(len(self.Q)):
                    self.Q[i] = self.Q[i].to(*args, **kwargs).detach().requires_grad_()

        return self

    def slow_parameters(self) -> list:
        """Create list of parameters to optimize in the slow phase.

        These are the weights and biases.
        """
        params = list(self.W)

        if self.bias:
            params.extend(self.h)
        if self.constrained:
            params.extend(self.Q)

        return params

    def slow_parameter_groups(self) -> list:
        """Create list of parameter groups to optimize in the slow phase.
        
        This is meant to allow for different learning rates for different parameters.
        The returned list is in the format accepted by optimizers -- a list of
        dictionaries, each of which contains `"params"` (an iterable of tensors in the
        group). Each dictionary also contains a `"name"` -- a string identifying the
        parameters.
        """
        groups = []
        groups.append({"name": "W", "params": self.W})
        if self.constrained:
            groups.append({"name": "Q", "params": self.Q})
        if self.bias:
            groups.append({"name": "h", "params": self.h})

        return groups

    @staticmethod
    def _extend(var: Sequence, n: int) -> np.ndarray:
        """Extend a scalar to a given number of elements, or convert a list/tensor of
        the correct length to a numpy array. The data type is also converted to float.
        """
        if torch.is_tensor(var):
            var = var.detach().numpy()
        res = (np.copy(var) if np.size(var) > 1 else np.repeat(var, n)).astype(float)

        assert len(res) == n
        return res

    def _get_activation_fcts(self) -> list:
        """Return list of activation functions for each layer."""
        activation = []

        for fct in self.activation:
            if not isinstance(fct, str):
                activation.append(fct)
            else:
                activation.append(_activation_map[fct][0])

        return activation

    def _get_activation_fcts_and_ders(self) -> Tuple[list, list]:
        """Return tuple of lists, one of activation functions and one of derivatives for
        each layer.
        
        This requires all layers to have the activation function specified by a string.
        In any other scenario, the function returns `(None, None)`.
        """
        activation = []
        der_activation = []

        for fct in self.activation:
            if not isinstance(fct, str):
                return None, None
            else:
                activation.append(_activation_map[fct][0])
                der_activation.append(_activation_map[fct][1])

        return activation, der_activation

    def __str__(self) -> str:
        s = (
            f"PCNetwork("
            f"dims={str(self.dims)}, "
            f"activation={str(self.activation)}, "
            f"bias={str(self.bias)}, "
            f"constrained={self.constrained}"
            f")"
        )
        return s

    def __repr__(self) -> str:
        s = (
            f"PCNetwork("
            f"dims={repr(self.dims)}, "
            f"activation={repr(self.activation)}, "
            f"bias={str(self.bias)}, "
            f"constrained={self.constrained}, "
            f"z_it={self.z_it}, "
            f"z_lr={self.z_lr}, "
            f"variances={self.variances}"
            f")"
        )
        return s
