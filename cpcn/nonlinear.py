"""Define the constrained-predictive coding network."""

from types import SimpleNamespace
from typing import Sequence, Union, Optional, Callable, Tuple

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


class BioPCN:
    """Constrained-predictive coding network."""

    def __init__(
        self,
        dims: Sequence,
        inter_dims: Optional[Sequence] = None,
        activation: Union[Sequence, Callable, str] = "tanh",
        z_it: int = 100,
        z_lr: float = 0.01,
        g_a: Union[Sequence, float] = 1.0,
        g_b: Union[Sequence, float] = 1.0,
        l_s: Union[Sequence, float] = 1.0,
        c_m: Union[Sequence, float] = 0.0,
        rho: Union[Sequence, float] = 1.0,
        bias_a: bool = True,
        bias_b: bool = True,
        fast_optimizer: Callable = torch.optim.Adam,
        wa0_scale: Union[Sequence, float] = 1.0,
        wb0_scale: Union[Sequence, float] = 1.0,
        q0_scale: Union[Sequence, float] = 1.0,
        m0_scale: Union[Sequence, float] = 1.0,
        init_scale_type: str = "unif_out_only",
    ):
        """Initialize the network.

        :param dims: number of pyramidal neurons in each layer
        :param inter_dims: number of interneurons per hidden layer; default: same as
            `dims[1:-1]`
        :param activation: activation function(s) to use for each layer; they can be
            callables or, preferrably, one of the following strings: `"none"` for linear
            (no activation function); `"relu"`; `"tanh"`
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
        :param wa0_scale: scale(s) for the (random) initial values of W_a
        :param wb0_scale: scale(s) for the (random) initial values of W_b
        :param q0_scale: scale(s) for the (random) initial values of Q
        :param m0_scale: scale(s) for the (random) initial values of M
        :param init_scale_type: how to initialize weights; can be `"xavier_uniform"` or
            `"unif_out_only"`; both generate uniform values between `-a` and `a`, where
            `a ** 2` is `6 / (n_in + n_out)` for `"xavier_uniform"` and `6 / n_out` for
            `"unif_out_only"`; these are scaled by the relevant scale factors above
        """
        self.training = True

        self.dims = np.copy(dims)
        assert len(self.dims) > 2

        self.activation = (
            (len(self.dims) - 1) * [activation]
            if isinstance(activation, str) or not hasattr(activation, "__len__")
            else list(activation)
        )

        if inter_dims is None:
            inter_dims = self.dims[1:-1]
        self.inter_dims = np.copy(inter_dims)

        assert len(self.dims) == len(self.inter_dims) + 2

        self.z_it = z_it
        self.z_lr = z_lr

        self.g_a = self._expand_per_layer(g_a)
        self.g_b = self._expand_per_layer(g_b)
        self.l_s = self._expand_per_layer(l_s)
        self.c_m = self._expand_per_layer(c_m)
        self.rho = self._expand_per_layer(rho)

        wa0_scale = self._expand_per_layer(wa0_scale)
        wb0_scale = self._expand_per_layer(wb0_scale)
        q0_scale = self._expand_per_layer(q0_scale)
        m0_scale = self._expand_per_layer(m0_scale)

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

        D = len(self.dims) - 2
        for i in range(D):
            self.W_a.append(torch.Tensor(self.dims[i + 2], self.dims[i + 1]))
            if self.bias_a:
                self.h_a.append(torch.Tensor(self.dims[i + 2]))

            self.W_b.append(torch.Tensor(self.dims[i + 1], self.dims[i]))
            if self.bias_b:
                self.h_b.append(torch.Tensor(self.dims[i + 1]))

            self.Q.append(torch.Tensor(self.inter_dims[i], self.dims[i + 1]))
            self.M.append(torch.Tensor(self.dims[i + 1], self.dims[i + 1]))

        # initialize weights and biases
        self._initialize_interlayer_weights(wa0_scale, wb0_scale, init_scale_type)
        self._initialize_intralayer_weights(q0_scale, m0_scale, init_scale_type)
        self._initialize_biases()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass with unconstrained output.

        This uses the basal weights and biases to propagate the input through the net
        up to the last hidden layer. The values in the output layer are generated using
        the apical weights.

        :param x: input sample
        :returns: list of layer activations after the forward pass
        """
        z = [x.detach()]
        n = len(self.dims)
        D = n - 2
        activation = self._get_activation_fcts()
        with torch.no_grad():
            for i in range(D):
                x = activation[i](x)

                W = self.W_b[i]
                h = self.h_b[i] if self.bias_b else 0

                x = x @ W.T + h
                z.append(x)

            x = activation[-1](x)
            W = self.W_a[-1]
            h = self.h_a[-1] if self.bias_a else 0
            x = x @ W.T + h
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
            `profile.pc_loss`; see `self.pc_loss()`
        :param latent_profile: if true, the evolution of the latent variables during the
            optimization is returned as `profile.z`, `profile.a`, `profile.b`, and
            `profile.n` in the output namespace; the values are stored after each
            optimizer step, and they are stored as a list of tensors, one for each
            layer, of shape `[n_it, batch_size, n_units]`; note that this output will
            always have a batch index, even if the input and output samples do not
        :return: namespace with results; this always contains the final layer currents
            and activations, in lists called `a`, `b`, `n`, and `z`, and a prediction,
            `y_pred`, which is the same as the last-layer activation `z[-1]` from
            `self.forward()`; unlike in the case of the profile, these final values obey
            the batch conventions from `x` and `y`: i.e., they only have a batch index
            if `x` and `y` do; the returned namespace also contains a `profile` member,
            which is either empty, or populated as described above when discussing the
            `..._profile` arguments;
        """
        assert x.ndim == y.ndim
        if x.ndim > 1:
            assert x.shape[0] == y.shape[0]

        # we calculate gradients manually
        with torch.no_grad():
            # start with a simple forward pass to initialize the layer values
            z = self.forward(x)
            y_pred = z[-1].clone()

            # fix the output layer values
            z[-1] = y.detach()

            # create an optimizer for the fast parameters
            fast_optimizer = self.fast_optimizer(z[1:-1], lr=self.z_lr)

            # create storage for output
            if pc_loss_profile:
                losses = torch.zeros(self.z_it)
            if latent_profile:
                batch_size = x.shape[0] if x.ndim > 1 else 1
                latent = SimpleNamespace()
                latent.z = [
                    torch.zeros((self.z_it, batch_size, dim)) for dim in self.dims
                ]
                latent.a = [
                    torch.zeros((self.z_it, batch_size, dim)) for dim in self.dims[1:-1]
                ]
                latent.b = [
                    torch.zeros((self.z_it, batch_size, dim)) for dim in self.dims[1:-1]
                ]
                latent.n = [
                    torch.zeros((self.z_it, batch_size, dim)) for dim in self.inter_dims
                ]

            # iterate until convergence
            a, b, n = self.calculate_currents(z)
            for i in range(self.z_it):
                self.calculate_z_grad(z, a, b, n)

                if pc_loss_profile:
                    losses[i] = self.pc_loss(z).item()

                fast_optimizer.step()
                a, b, n = self.calculate_currents(z)

                if latent_profile:
                    for k, crt_z in enumerate(z):
                        latent.z[k][i, :, :] = crt_z
                    for k in range(len(self.inter_dims)):
                        latent.a[k][i, :, :] = a[k]
                        latent.b[k][i, :, :] = b[k]
                        latent.n[k][i, :, :] = n[k]

        ns = SimpleNamespace(z=z, a=a, b=b, n=n, y_pred=y_pred)
        if latent_profile:
            ns.profile = latent
        else:
            ns.profile = SimpleNamespace()
        if pc_loss_profile:
            ns.profile.pc_loss = losses

        return ns

    def calculate_weight_grad(self, fast: SimpleNamespace, reduction: str = "mean"):
        """Calculate gradients for slow (weight) variables.

        The calculated gradients are assigned to the `grad` attribute of each weight
        tensor.

        Note that these gradients do not follow from `self.pc_loss()`! While there is a
        modified loss that can generate both the latent- and weight-gradients in the
        linear case, this is no longer true for non-linear generalizations. We therefore
        use manual gradients in this case, as well, for consistency.

        :param fast: namespace of equilibrium values for the fast variables: the latents
            `z`; the apical current `a`; the basal current `b`; and the interneuron
            activities `n`
        :param reduction: reduction to apply to the gradients: `"mean" | "sum"`
        """
        D = len(fast.z) - 2
        batch_outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
        red_fct = {"mean": torch.mean, "sum": torch.sum}[reduction]

        fz, fz_der = self._calculate_activations_and_derivatives(fast.z, der=True)
        for i in range(D):
            # apical
            pre = fz[i + 1]
            if self.bias_a:
                post = fast.z[i + 2] - self.h_a[i]
            else:
                post = fast.z[i + 2]
            grad = self.g_a[i] * (self.rho[i] * self.W_a[i] - batch_outer(post, pre))
            if grad.ndim == self.W_a[i].ndim + 1:
                # this is a batch evaluation!
                grad = red_fct(grad, 0)
            self.W_a[i].grad = grad

            # basal
            plateau = self.g_a[i] * fz_der[i + 1] * fast.a[i]
            hebbian_self = (self.l_s[i] - self.g_b[i]) * fast.z[i + 1]
            if self.c_m[i] != 0:
                hebbian_lateral = (
                    self.c_m[i] * fz_der[i + 1] * (fz[i + 1] @ self.M[i].T)
                )
            else:
                hebbian_lateral = 0
            hebbian = hebbian_self + hebbian_lateral

            pre = fz[i]
            post = hebbian - plateau
            grad = batch_outer(post, pre)
            if grad.ndim == self.W_b[i].ndim + 1:
                # this is a batch evaluation!
                grad = red_fct(grad, 0)
            self.W_b[i].grad = grad

            # inter
            pre = fz[i + 1]
            post = fast.n[i]
            grad = self.g_a[i] * (self.rho[i] * self.Q[i] - batch_outer(post, pre))
            if grad.ndim == self.Q[i].ndim + 1:
                # this is a batch evaluation!
                grad = red_fct(grad, 0)
            self.Q[i].grad = grad

            # lateral
            if self.c_m[i] != 0:
                pre = fz[i + 1]
                post = pre
                grad = self.c_m[i] * (self.M[i] - batch_outer(post, pre))
                if grad.ndim == self.M[i].ndim + 1:
                    # this is a batch evaluation!
                    grad = red_fct(grad, 0)
                self.M[i].grad = grad
            else:
                self.M[i].grad = torch.zeros_like(self.M[i])

            # biases
            if self.bias_a:
                mu = fz[i + 1] @ self.W_a[i].T + self.h_a[i]
                grad = self.g_a[i] * (mu - fast.z[i + 2])
                if grad.ndim == self.h_a[i].ndim + 1:
                    # this is a batch evaluation!
                    grad = red_fct(grad, 0)
                self.h_a[i].grad = grad
            if self.bias_b:
                mu = fz[i] @ self.W_b[i].T + self.h_b[i]
                grad = self.g_b[i] * (mu - fast.z[i + 1])
                if grad.ndim == self.h_b[i].ndim + 1:
                    # this is a batch evaluation!
                    grad = red_fct(grad, 0)
                self.h_b[i].grad = grad

    def calculate_currents(self, z: Sequence) -> Tuple[list, list, list]:
        """Calculate apical, basal, and interneuron currents in all layers.

        :param z: values of latent variables in each layer
        :return: tuple of lists `(a, b, n)` of apical, basal, and interneuron currents
        """
        D = len(z) - 2
        a = []
        b = []
        n = []
        activation = self._get_activation_fcts()
        for i in range(D):
            f0 = activation[i]
            f1 = activation[i + 1]
            n.append((f1(z[i + 1]) @ self.Q[i].T).detach())

            if self.bias_b:
                b.append((f0(z[i]) @ self.W_b[i].T + self.h_b[i]).detach())
            else:
                b.append((f0(z[i]) @ self.W_b[i].T).detach())

            if self.bias_a:
                a_feedback = ((z[i + 2] - self.h_a[i]) @ self.W_a[i]).detach()
            else:
                a_feedback = (z[i + 2] @ self.W_a[i]).detach()
            a_inter = n[i] @ self.Q[i]
            a.append((a_feedback - a_inter).detach())

        return a, b, n

    def calculate_z_grad(self, z: Sequence, a: Sequence, b: Sequence, n: Sequence):
        """Calculate gradients for fast (z) variables.

        This uses the apical (`a`), basal (`b`), and interneuron (`n`) currents,
        presumably calculated using `calculate_currents`. The calculated gradients are
        assigned to the `grad` attribute of each tensor in `z`.

        Note that these gradients do not follow from `self.pc_loss()`! While there is a
        modified loss that can generate both the latent- and weight-gradients in the
        linear case, this is no longer true for non-linear generalizations. We therefore
        use manual gradients in this case, as well, for consistency.

        :param z: latent-state variables
        :param a: apical currents
        :param b: basal currents
        :param n: interneuron activities
        """
        fz, fz_der = self._calculate_activations_and_derivatives(z, der=True)
        with torch.no_grad():
            D = len(z) - 2
            for i in range(D):
                crt_f = fz[i + 1]
                crt_df = fz_der[i + 1]
                grad_apical = self.g_a[i] * crt_df * a[i]
                grad_basal = self.g_b[i] * b[i]
                grad_lateral = (self.c_m[i] * crt_df * (crt_f @ self.M[i].T)).detach()
                grad_leak = self.l_s[i] * z[i + 1]

                z[i + 1].grad = grad_lateral + grad_leak - grad_apical - grad_basal

    def pc_loss(self, z: Sequence, reduction: str = "mean") -> torch.Tensor:
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

            mu_x[l + 1] = W_x[l] @ f(z[l]) + h_x[l]

        with `x` either `a` or `b` and `f` the activation function for layer `l`.

        This loss is minimized whenever the predictive-coding loss is minimized. (That
        is, at the minimum, `W_a == W_b`.)

        :param z: values of latent variables in each layer
        :param reduction: reduction to apply to the output: `"none" | "mean" | "sum"`
        """
        batch_size = 1 if z[0].ndim == 1 else len(z[0])
        loss = torch.zeros(batch_size).to(z[0].device)

        D = len(z) - 2
        activation = self._get_activation_fcts()
        for i in range(D):
            f0 = activation[i]
            f1 = activation[i + 1]
            mu_a = f1(z[i + 1]) @ self.W_a[i].T
            if self.bias_a:
                mu_a += self.h_a[i]
            apical = self.g_a[i] * ((z[i + 2] - mu_a) ** 2).sum(dim=-1)

            mu_b = f0(z[i]) @ self.W_b[i].T
            if self.bias_b:
                mu_b += self.h_b[i]
            basal = self.g_b[i] * ((z[i + 1] - mu_b) ** 2).sum(dim=-1)

            loss += apical + basal

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction != "none":
            raise ValueError("unknown reduction type")

        loss /= 2
        return loss

    def parameters(self) -> list:
        """Create list of parameters to optimize in the slow phase.

        These are the weights and biases.
        """
        res = self.W_a + self.W_b + self.Q + self.M
        if self.bias_a:
            res += self.h_a
        if self.bias_b:
            res += self.h_b

        return res

    def parameter_groups(self) -> list:
        """Create list of parameter groups to optimize in the slow phase.
        
        This is meant to allow for different learning rates for different parameters.
        The returned list is in the format accepted by optimizers -- a list of
        dictionaries, each of which contains `"params"` (an iterable of tensors in the
        group). Each dictionary also contains a `"name"` -- a string identifying the
        parameters.
        """
        groups = []
        groups.extend(
            {"name": f"W_a:{i}", "params": [_]} for i, _ in enumerate(self.W_a)
        )
        groups.extend(
            {"name": f"W_b:{i}", "params": [_]} for i, _ in enumerate(self.W_b)
        )
        groups.extend({"name": f"Q:{i}", "params": [_]} for i, _ in enumerate(self.Q))
        groups.extend({"name": f"M:{i}", "params": [_]} for i, _ in enumerate(self.M))
        if self.bias_a:
            groups.extend(
                {"name": f"h_a:{i}", "params": [_]} for i, _ in enumerate(self.h_a)
            )
        if self.bias_b:
            groups.extend(
                {"name": f"h_b:{i}", "params": [_]} for i, _ in enumerate(self.h_b)
            )

        return groups

    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers."""
        with torch.no_grad():
            for i in range(len(self.W_a)):
                self.W_a[i] = self.W_a[i].to(*args, **kwargs).detach().requires_grad_()
                self.W_b[i] = self.W_b[i].to(*args, **kwargs).detach().requires_grad_()
                if self.bias_a:
                    self.h_a[i] = (
                        self.h_a[i].to(*args, **kwargs).detach().requires_grad_()
                    )
                if self.bias_b:
                    self.h_b[i] = (
                        self.h_b[i].to(*args, **kwargs).detach().requires_grad_()
                    )

                self.Q[i] = self.Q[i].to(*args, **kwargs).detach().requires_grad_()
                self.M[i] = self.M[i].to(*args, **kwargs).detach().requires_grad_()

        return self

    def clone(self) -> "BioPCN":
        new = BioPCN(
            self.dims,
            self.inter_dims,
            activation=self.activation,
            z_it=self.z_it,
            z_lr=self.z_lr,
            g_a=self.g_a,
            g_b=self.g_b,
            l_s=self.l_s,
            c_m=self.c_m,
            rho=self.rho,
            bias_a=self.bias_a,
            bias_b=self.bias_b,
            fast_optimizer=self.fast_optimizer,
        )

        for d in self.parameter_groups():
            name_full = d["name"]
            name, layer_str = name_full.split(":")
            layer = int(layer_str)

            value = d["params"][0]

            getattr(new, name)[layer] = value.detach().clone().requires_grad_()

        return new

    def train(self):
        """Set in training mode."""
        self.training = True

    def eval(self):
        """Set in evaluation mode."""
        self.training = False

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

    def _get_weight_init_fct(self, init_scale_type: str):
        """Return the function used to initialize weights."""
        if init_scale_type == "xavier_uniform":
            return nn.init.xavier_uniform_
        elif init_scale_type == "unif_out_only":

            def scaling_fct_(tensor: torch.Tensor, gain: float):
                a = gain * np.sqrt(1 / tensor.shape[1])
                torch.nn.init.uniform_(tensor, -a, a)

            return scaling_fct_
        else:
            raise ValueError(f"unknown init_scale_type, {init_scale_type}")

    def _calculate_activations_and_derivatives(
        self, z: Sequence, der: bool
    ) -> Tuple[list, list]:
        """Calculate activations and derivatives (if `der` is true)."""
        fz = []
        fz_der = []
        activation, der_activation = self._get_activation_fcts_and_ders()
        if der and activation is None:
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
                if der:
                    fz_der.append(der_activation[i](crt_z))

        return fz, fz_der

    def _initialize_interlayer_weights(
        self, wa0_scale: torch.Tensor, wb0_scale: torch.Tensor, init_scale_type: str
    ):
        init_fct = self._get_weight_init_fct(init_scale_type)
        for W, scale in zip(self.W_a, wa0_scale):
            init_fct(W, gain=scale)
            W.requires_grad = True
        for W, scale in zip(self.W_b, wb0_scale):
            init_fct(W, gain=scale)
            W.requires_grad = True

    def _initialize_intralayer_weights(
        self, q0_scale: torch.Tensor, m0_scale: torch.Tensor, init_scale_type: str
    ):
        init_fct = self._get_weight_init_fct(init_scale_type)
        for Q, scale in zip(self.Q, q0_scale):
            init_fct(Q, gain=scale)
            Q.requires_grad = True
        for i, (M, scale) in enumerate(zip(self.M, m0_scale)):
            init_fct(M, gain=scale)
            M.requires_grad = True

    def _initialize_biases(self):
        all = []
        if self.bias_a:
            all += self.h_a
        if self.bias_b:
            all += self.h_b

        for h in all:
            h.zero_()
            h.requires_grad = True

    def _expand_per_layer(self, theta) -> np.ndarray:
        """Expand a quantity to per-layer, if needed, and convert to numpy array."""
        D = len(self.dims) - 2

        if torch.is_tensor(theta):
            assert theta.ndim == 1
            if len(theta) > 1:
                assert len(theta) == D
                theta = np.copy(theta.detach().numpy())
            else:
                theta = theta.item() * np.ones(D)
        elif hasattr(theta, "__len__") and len(theta) == D:
            theta = np.copy(theta)
        elif np.size(theta) == 1:
            theta = theta * torch.ones(D)
        else:
            raise ValueError("parameter has wrong size")

        return theta

    def __str__(self) -> str:
        s = (
            f"BioPCN(dims={str(self.dims)}, "
            f"inter_dims={str(self.inter_dims)}, "
            f"activation={str(self.activation)}, "
            f"bias_a={self.bias_a}, "
            f"bias_b={self.bias_b}"
            f")"
        )
        return s

    def __repr__(self) -> str:
        s = (
            f"BioPCN("
            f"dims={repr(self.dims)}, "
            f"inter_dims={repr(self.inter_dims)}, "
            f"activation={repr(self.activation)}, "
            f"bias_a={self.bias_a}, "
            f"bias_b={self.bias_b}, "
            f"fast_optimizer={repr(self.fast_optimizer)}, "
            f"z_it={self.z_it}, "
            f"z_lr={self.z_lr}, "
            f"g_a={repr(self.g_a)}, "
            f"g_b={repr(self.g_b)}, "
            f"l_s={repr(self.l_s)}, "
            f"c_m={repr(self.c_m)}, "
            f"rho={repr(self.rho)} "
            f")"
        )
        return s

    @staticmethod
    def from_pcn(pcn, match_weights: bool = False, **kwargs) -> "BioPCN":
        """Create BioPCN network matching a Whittington & Bogacz network.
        
        :param pcn: source `PCNetwork`
        :param match_weights: if true, copy over initial weights and biases; otherwise
            only match static parameters, like conductances
        :param **kwargs: additional arguments are passed directly to `BioPCN`,
            overriding any other values
        :return: a `BioPCN` instance with matching parameters
        """
        g_a = 0.5 / pcn.variances[1:]
        g_b = 0.5 / pcn.variances[:-1]

        g_a[-1] *= 2
        g_b[0] *= 2

        kwargs.setdefault("activation", pcn.activation)
        kwargs.setdefault("g_a", g_a)
        kwargs.setdefault("g_b", g_b)
        kwargs.setdefault("c_m", 0)
        kwargs.setdefault("l_s", kwargs["g_b"])
        kwargs.setdefault("bias_a", pcn.bias)
        kwargs.setdefault("bias_b", pcn.bias)
        if pcn.constrained:
            kwargs["rho"] = pcn.rho

        cpcn = BioPCN(pcn.dims, **kwargs)

        if match_weights:
            D = len(cpcn.inter_dims)
            for i in range(D):
                cpcn.W_a[i] = pcn.W[i + 1].detach().clone()
                cpcn.W_b[i] = pcn.W[i].detach().clone()

            if pcn.bias:
                for i in range(D):
                    if cpcn.bias_a:
                        cpcn.h_a[i] = pcn.h[i + 1].detach().clone()
                    if cpcn.bias_b:
                        cpcn.h_b[i] = pcn.h[i].detach().clone()

            if pcn.constrained:
                for i in range(D):
                    cpcn.Q[i] = pcn.Q[i].detach().clone()

        return cpcn
