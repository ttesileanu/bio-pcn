"""Define utilities for training predictive-coding models. """

from types import SimpleNamespace
import torch

from typing import Iterable, Optional


class TrainerBatch:
    """Handle for one training batch."""

    def __init__(
        self, x: torch.Tensor, y: torch.Tensor, idx: int, n: int, training: bool
    ):
        self.x = x
        self.y = y
        self.idx = idx
        self.n = n
        self.training = training

    def feed(self, net, **kwargs) -> SimpleNamespace:
        """Feed the batch to the network's `relax` method and calculate gradients.

        :param net: network to feed the batch to
        :param **kwargs: additional arguments to pass to `net.relax()`
        :return: namespace containing the batch's `x` and `y`, as well as the output
        from `net.relax()`, in `fast`.
        """
        # run fast dynamics
        res = net.relax(self.x, self.y, **kwargs)
        ns = SimpleNamespace(x=self.x, y=self.y, fast=res)

        if self.training:
            # calculate gradients
            net.calculate_weight_grad(ns.fast)

        return ns

    def every(self, step: int) -> bool:
        """Return true every `step` steps."""
        return self.idx % step == 0

    def count(self, total: int) -> bool:
        """Return true a total of `total` times.
        
        Including first and last batch.
        """
        # should be true when batch = floor(k * (n - 1) / (total - 1)) for integer k
        # this implies (batch * (total - 1)) % (n - 1) == 0 or > (n - total).
        if total == 1:
            return self.idx == 0
        else:
            mod = (self.idx * (total - 1)) % (self.n - 1)
            return mod == 0 or mod > self.n - total


class _TrainerIterator:
    """Iterator used by Trainer."""

    def __init__(self, loader: Iterable, n: int, training: bool):
        self.iterable = loader
        self.n = n
        self.training = training
        self.i = 0
        self.it = iter(self.iterable)

    def __next__(self) -> TrainerBatch:
        if self.i < self.n:
            try:
                x, y = next(self.it)
            except StopIteration:
                self.it = iter(self.iterable)
                x, y = next(self.it)

            batch = TrainerBatch(x=x, y=y, idx=self.i, n=self.n, training=self.training)
            self.i += 1

            return batch
        else:
            raise StopIteration


class TrainerIterable:
    """Iterable returned by calling a Trainer."""

    def __init__(
        self,
        trainer: "Trainer",
        n_batches: int,
        loader: Optional[Iterable] = None,
        training: bool = True,
    ):
        self.trainer = trainer
        self.n_batches = n_batches
        self.loader = self.trainer.loader if loader is None else loader
        self.training = training

    def __iter__(self) -> _TrainerIterator:
        return _TrainerIterator(self.loader, self.n_batches, self.training)

    def __len__(self) -> int:
        return self.n_batches


class TrainerEvaluateIterable(TrainerIterable):
    """Iterable for running a validation run."""

    def __init__(self, trainer: "Trainer", loader: Iterable):
        super().__init__(trainer, len(loader), training=False, loader=loader)

    def run(self, net):
        for batch in self:
            batch.feed(net)


class Trainer:
    """Class used to help train predictive-coding networks."""

    def __init__(self, loader: Iterable):
        self.loader = loader

    def __call__(self, n_batches: int) -> TrainerIterable:
        return TrainerIterable(self, n_batches)

    def __len__(self) -> int:
        """Trainer length equals the length of the loader."""
        return len(self.loader)

    def evaluate(self, val_loader: Iterable) -> TrainerEvaluateIterable:
        return TrainerEvaluateIterable(self, val_loader)
