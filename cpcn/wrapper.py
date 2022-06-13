"""Define wrappers for getting predictions out of predictive-coding networks."""

import torch

from types import SimpleNamespace
from typing import Union, Callable, Optional


class PCWrapper:
    """Wrap a predictive-coding network to obtain predictions by applying an arbitrary
    `Module` to one of the layers.
    
    Attributes
    :param pc_net: underlying predictive-coding network
    :param predictor: predictor module
    :param dim: index of `z` layer in `pc_net` on which `predictor` acts
    :param loss: loss function
    """

    def __init__(
        self,
        pc_net,
        predictor: Union[Callable, str],
        dim: int = -2,
        loss: Optional[Callable] = None,
    ):
        """Initialize the wrapper.
        
        :param pc_net: predictive-coding network to wrap
        :param predictor: predictor network; this can either be a string or an arbitrary
            `torch.nn.Module`; the string options are:
                "linear":           linear network
                "linear-relu":      linear network followed by ReLU
                "linear-softmax":   linear network followed by softmax.
        :param dim: which activation layer from `pc_net` to use as input to predictor
        :param loss: loss function; default: `torch.nn.MSELoss()`
        """
        self.pc_net = pc_net
        self.dim = dim
        self.loss = loss if loss is not None else torch.nn.MSELoss()

        if isinstance(predictor, str):
            in_size = self.pc_net.dims[self.dim]
            out_size = self.pc_net.dims[-1]

            linear_predictor = torch.nn.Linear(in_size, out_size)
            if predictor == "linear-relu":
                predictor = torch.nn.Sequential(linear_predictor, torch.nn.ReLU())
            elif predictor == "linear-softmax":
                predictor = torch.nn.Sequential(linear_predictor, torch.nn.Softmax())
            elif predictor == "linear":
                predictor = linear_predictor
            else:
                raise ValueError(f'unknown predictor "{predictor}"')

        self.predictor = predictor

    def relax(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> SimpleNamespace:
        """Run relax on underlying PC net, then use predictor net to calculate final
        prediction, storing it as `y_pred` in output namespace.
        
        Any additional keyword arguments are passed to `self.pc_net.relax`.
        """
        ns = self.pc_net.relax(x, y, **kwargs)

        pred_input = ns.z[self.dim].detach()
        pred_output = self.predictor(pred_input)

        ns.y_pred = pred_output
        return ns

    def pc_loss(self, z: list) -> torch.Tensor:
        """Return PC loss as calculated by PC net."""
        return self.pc_net.pc_loss(z)

    def calculate_weight_grad(self, fast: SimpleNamespace, **kwargs):
        """Calculate gradients for predictive-coding slow variables and for predictor
        variables.

        Additional arguments are passed to `pc_net.calculate_weight_grad`.

        Note that a `reduction` option given here needs to be matched by an equivalent
        `reduction` option given to the loss in the constructor `self.__init__`!
        """
        self.pc_net.calculate_weight_grad(fast, **kwargs)

        for param in self.predictor.parameters():
            param.grad = None

        loss = self.loss(fast.y_pred, fast.z[-1])
        loss.backward()

    def parameters(self) -> list:
        """Return PC net parameters concatenated with those from predictor net."""
        pc_params = self.pc_net.parameters()
        pred_params = list(self.predictor.parameters())

        return pc_params + pred_params

    def parameter_groups(self) -> dict:
        """Return PC net parameter groups, plus the parameters for the predictor net.
        
        The predictor-net parameters use the name `"predictor"`.
        """
        pc_param_groups = self.pc_net.parameter_groups()
        pred_params = self.predictor.parameters()

        return pc_param_groups + [{"name": "predictor", "params": pred_params}]

    def train(self):
        """Set in training mode."""
        self.pc_net.train()
        self.predictor.train()

    def eval(self):
        """Set in evaluation mode."""
        self.pc_net.eval()
        self.predictor.eval()

    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers."""
        self.pc_net.to(*args, **kwargs)
        self.predictor.to(*args, **kwargs)

        return self
