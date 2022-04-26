""" Define the linear constrained-predictive coding network. """

from types import SimpleNamespace
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
        rho: Union[Sequence, float] = 1.0,
        bias_a: bool = True,
        bias_b: bool = True,
        fast_optimizer: Callable = torch.optim.Adam,
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
        :param rho: squared radius for whiteness constraint; that is, the constraint is
            cov_matrix(z) <= rho * identity_matrix
        :param bias_a: whether to have bias terms at the apical end
        :param bias_b: whether to have bias terms at the basal end
        :param fast_optimizer: constructor for the optimizer used for the fast dynamics
            in `relax`
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
        self.rho = self._expand_per_layer(rho)

        self.bias_a = bias_a
        self.bias_b = bias_b

        self.fast_optimizer = fast_optimizer

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
        self.a = [torch.zeros(dim) for dim in self.pyr_dims[1:-1]]
        self.b = [torch.zeros(dim) for dim in self.pyr_dims[1:-1]]
        self.n = [torch.zeros(dim) for dim in self.inter_dims]
        self.z = [torch.zeros(dim) for dim in self.pyr_dims]

    def forward(self, x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
        """Do a forward pass with unconstrained output.

        This uses the basal weights and biases to propagate the input through the net
        up to the last hidden layer, then the apical weights to generate the output.

        :param x: input sample
        :param inplace: whether to update the latent-state values in-place; if false,
            the values are returned instead of the last-layer activation
        :returns: list of layer activations after the forward pass; if `inplace` is
            true (the default), this is the same as `self.z`; if `inplace` is false, the
            returned activations will be the same as if `inplace` were true, but
            `self.z` will be untouched
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

        if inplace:
            self.z = z

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
            `pc_loss`; see `self.pc_loss()`
        :param latent_profile: if true, the evolution of the latent variables during the
            optimization is returned as `latent` in the output, with subfields for each
            variable, e.g., `latent.z`, `latent.n`, etc.; the values are stored after
            each optimizer step, and they are stored as a list of tensors, one for each
            layer, of shape `[n_it, batch_size, n_units]`; note that this output will
            always have a batch index, even if the input and output samples do not
        :return: if `pc_loss_profile` is true, the predictive-coding loss evaluated
            before every optimization step; otherwise, `None`
        """
        assert x.ndim == y.ndim
        if x.ndim > 1:
            assert x.shape[0] == y.shape[0]

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
            if pc_loss_profile:
                losses = torch.zeros(self.z_it)
            if latent_profile:
                batch_size = x.shape[0] if x.ndim > 1 else 1
                latent = SimpleNamespace()
                latent.z = [
                    torch.zeros((self.z_it, batch_size, dim)) for dim in self.pyr_dims
                ]
                latent.a = [
                    torch.zeros((self.z_it, batch_size, dim))
                    for dim in self.pyr_dims[1:-1]
                ]
                latent.b = [
                    torch.zeros((self.z_it, batch_size, dim))
                    for dim in self.pyr_dims[1:-1]
                ]
                latent.n = [
                    torch.zeros((self.z_it, batch_size, dim)) for dim in self.inter_dims
                ]

            self.calculate_currents()
            for i in range(self.z_it):
                self.calculate_z_grad()
                fast_optimizer.step()
                self.calculate_currents()

                if pc_loss_profile:
                    losses[i] = self.pc_loss().item()
                if latent_profile:
                    for k, crt_z in enumerate(self.z):
                        latent.z[k][i, :, :] = crt_z
                    for var in ["a", "b", "n"]:
                        crt_storage = getattr(latent, var)
                        for k, crt_values in enumerate(getattr(self, var)):
                            crt_storage[k][i, :, :] = crt_values

        ns = SimpleNamespace()
        if pc_loss_profile:
            ns.pc_loss = losses
        if latent_profile:
            ns.latent = latent

        return ns

    def calculate_weight_grad(self, reduction: str = "mean"):
        """Calculate gradients for slow (weight) variables.

        This assumes that the fast variables have been calculated, using `relax()`. The
        calculated gradients are assigned the `grad` attribute of each weight tensor.

        Note that these gradients do not follow from `self.pc_loss()`! While there is a
        modified loss that can generate both the latent- and weight-gradients in the
        linear case, this is no longer true for non-linear generalizations. We therefore
        use manual gradients in this case, as well, for consistency.

        :param reduction: reduction to apply to the gradients: `"mean" | "sum"`
        """
        D = len(self.pyr_dims) - 2
        batch_outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
        red_fct = {"mean": torch.mean, "sum": torch.sum}[reduction]
        for i in range(D):
            # apical
            pre = self.z[i + 1]
            if self.bias_a:
                post = self.z[i + 2] - self.h_a[i]
            else:
                post = self.z[i + 2]
            grad = self.g_a[i] * (self.W_a[i] - batch_outer(post, pre))
            if grad.ndim == self.W_a[i].ndim + 1:
                # this is a batch evaluation!
                grad = red_fct(grad, 0)
            self.W_a[i].grad = grad

            # basal
            plateau = self.g_a[i] * self.a[i]
            hebbian_self = (self.l_s[i] - self.g_b[i]) * self.z[i + 1]
            hebbian_lateral = self.c_m[i] * self.z[i + 1] @ self.M[i].T
            hebbian = hebbian_self + hebbian_lateral

            pre = self.z[i]
            post = hebbian - plateau
            grad = batch_outer(post, pre)
            if grad.ndim == self.W_b[i].ndim + 1:
                # this is a batch evaluation!
                grad = red_fct(grad, 0)
            self.W_b[i].grad = grad

            # inter
            pre = self.z[i + 1]
            post = self.n[i]
            grad = self.g_a[i] * (self.rho[i] * self.Q[i] - batch_outer(post, pre))
            if grad.ndim == self.Q[i].ndim + 1:
                # this is a batch evaluation!
                grad = red_fct(grad, 0)
            self.Q[i].grad = grad

            # lateral
            pre = self.z[i + 1]
            post = pre
            grad = self.c_m[i] * (self.M[i] - batch_outer(post, pre))
            if grad.ndim == self.M[i].ndim + 1:
                # this is a batch evaluation!
                grad = red_fct(grad, 0)
            self.M[i].grad = grad

            # biases
            if self.bias_a:
                mu = self.z[i + 1] @ self.W_a[i].T + self.h_a[i]
                grad = self.g_a[i] * (mu - self.z[i + 2])
                if grad.ndim == self.h_a[i].ndim + 1:
                    # this is a batch evaluation!
                    grad = red_fct(grad, 0)
                self.h_a[i].grad = grad
            if self.bias_b:
                mu = self.z[i] @ self.W_b[i].T + self.h_b[i]
                grad = self.g_b[i] * (mu - self.z[i + 1])
                if grad.ndim == self.h_b[i].ndim + 1:
                    # this is a batch evaluation!
                    grad = red_fct(grad, 0)
                self.h_b[i].grad = grad

    def calculate_z_grad(self):
        """Calculate gradients for fast (z) variables.

        This assumes that the currents were calculated using `calculate_currents`. The
        calculated gradients are assigned to the `grad` attribute of each tensor in
        `self.z`.

        Note that these gradients do not follow from `self.pc_loss()`! While there is a
        modified loss that can generate both the latent- and weight-gradients in the
        linear case, this is no longer true for non-linear generalizations. We therefore
        use manual gradients in this case, as well, for consistency.
        """
        D = len(self.pyr_dims) - 2
        for i in range(D):
            grad_apical = self.g_a[i] * self.a[i]
            grad_basal = self.g_b[i] * self.b[i]
            grad_lateral = self.c_m[i] * self.z[i + 1] @ self.M[i].T
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
            self.n[i] = self.z[i + 1] @ self.Q[i].T

            if self.bias_b:
                self.b[i] = self.z[i] @ self.W_b[i].T + self.h_b[i]
            else:
                self.b[i] = self.z[i] @ self.W_b[i].T

            if self.bias_a:
                a_feedback = (self.z[i + 2] - self.h_a[i]) @ self.W_a[i]
            else:
                a_feedback = self.z[i + 2] @ self.W_a[i]
            a_inter = self.n[i] @ self.Q[i]
            self.a[i] = a_feedback - a_inter

    def pc_loss(self, reduction: str = "mean") -> torch.Tensor:
        """Estimate predictive-coding loss given current activation values.

        Note that this loss does *not* generate either the latent-state gradients from
        `self.calculate_z_grad()`, or the weight gradients from
        `self.calculate_weight_grad()`!

        This is defined as the predictive-coding loss with duplicated weights connecting
        the hidden layers. Schematically,

            pc_loss = 0.5 * sum(g_a * (z[l + 1] - mu_a[l + 1]) ** 2 +
                                g_b * (z[l] - mu_b[l]) ** 2))

        where the sum is over the hidden layers, and the predictions `mu_a` and `mu_b`
        are calculated using the apical and basal weights and biases, respectively:

            mu_x[l + 1] = W_x[l] @ z[l] + h_x[l]

        with `x` either `a` or `b`.

        This loss is minimized whenever the predictive-coding loss is minimized. (That
        is, at the minimum, `W_a == W_b`.)

        :param reduction: reduction to apply to the output: `"none" | "mean" | "sum"`
        """
        batch_size = 1 if self.z[0].ndim == 1 else len(self.z[0])
        loss = torch.zeros(batch_size)

        D = len(self.pyr_dims) - 2
        for i in range(D):
            mu_a = self.z[i + 1] @ self.W_a[i].T
            if self.bias_a:
                mu_a += self.h_a[i]
            apical = self.g_a[i] * ((self.z[i + 2] - mu_a) ** 2).sum(dim=-1)

            mu_b = self.z[i] @ self.W_b[i].T
            if self.bias_b:
                mu_b += self.h_b[i]
            basal = self.g_b[i] * ((self.z[i + 1] - mu_b) ** 2).sum(dim=-1)

            loss += apical + basal

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction != "none":
            raise ValueError("unknown reduction type")

        loss /= 2
        return loss

    def fast_parameters(self) -> list:
        """Create list of parameters to optimize in the fast phase.

        These are the activations in all the hidden layers.
        """
        return self.z[1:-1]

    def slow_parameters(self) -> list:
        """ Create list of parameters to optimize in the slow phase.

        These are the weights and biases.
        """
        res = self.W_a + self.W_b + self.Q + self.M
        if self.bias_a:
            res += self.h_a
        if self.bias_b:
            res += self.h_b

        return res

    def to(self, *args, **kwargs):
        """ Moves and/or casts the parameters and buffers. """
        with torch.no_grad():
            for i in range(len(self.W_a)):
                self.W_a[i] = self.W_a[i].to(*args, **kwargs).requires_grad_()
                self.W_b[i] = self.W_b[i].to(*args, **kwargs).requires_grad_()
                if self.bias_a:
                    self.h_a[i] = self.h_a[i].to(*args, **kwargs).requires_grad_()
                if self.bias_b:
                    self.h_b[i] = self.h_b[i].to(*args, **kwargs).requires_grad_()

                self.Q[i] = self.Q[i].to(*args, **kwargs).requires_grad_()
                self.M[i] = self.M[i].to(*args, **kwargs).requires_grad_()

                self.a[i] = self.a[i].to(*args, **kwargs)
                self.b[i] = self.b[i].to(*args, **kwargs)
                self.n[i] = self.n[i].to(*args, **kwargs)

            for i in range(len(self.z)):
                self.z[i] = self.z[i].to(*args, **kwargs)

        return self

    def train(self):
        """ Set in training mode. """
        self.training = True

    def eval(self):
        """ Set in evaluation mode. """
        self.training = False

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

        if torch.is_tensor(theta):
            assert theta.ndim == 1
            if len(theta) > 1:
                assert len(theta) == D
                theta = theta.clone()
            else:
                theta = theta * torch.ones(D)
        elif hasattr(theta, "__len__") and len(theta) == D:
            assert len(theta) == D
            theta = torch.from_numpy(np.asarray(theta))
        elif np.size(theta) == 1:
            theta = theta * torch.ones(D)
        else:
            raise ValueError("parameter has wrong size")

        return theta

    def __str__(self) -> str:
        s = (
            f"LinearCPCNetwork(pyr_dims={str(self.pyr_dims)}, "
            f"inter_dims={str(self.inter_dims)}, "
            f"bias_a={self.bias_a}, "
            f"bias_b={self.bias_b}"
            f")"
        )
        return s

    def __repr__(self) -> str:
        s = (
            f"LinearCPCNetwork("
            f"pyr_dims={repr(self.pyr_dims)}, "
            f"inter_dims={repr(self.inter_dims)}, "
            f"bias_a={self.bias_a}, "
            f"bias_b={self.bias_b}, "
            f"fast_optimizer={repr(self.fast_optimizer)}, "
            f"z_it={self.z_it}, "
            f"z_lr={self.z_lr}, "
            f"g_a={repr(self.g_a)}, "
            f"g_b={repr(self.g_b)}, "
            f"l_s={repr(self.l_s)}, "
            f"c_m={repr(self.c_m)}"
            f")"
        )
        return s
