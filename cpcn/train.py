"""Define utilities for training predictive-coding models. """

from types import SimpleNamespace
import torch
import numpy as np

import warnings

from typing import Iterable, Optional

from cpcn.track import Tracker


class DivergenceError(Exception):
    pass


class DivergenceWarning(Warning):
    pass


class _BatchReporter:
    """Object used to implement the `report` framework for `TrainerBatch`."""

    def __init__(self, batch: "TrainerBatch"):
        self._batch = batch
        self._tracker = self._batch.tracker

    def __getattr__(self, name: str):
        return lambda *args, _name=name, **kwargs: self._report(_name, *args, **kwargs)

    def _report(self, name: str, *args, **kwargs):
        idx = self._batch.idx
        sample_idx = self._batch.sample_idx

        if kwargs.pop("check_nan", False):
            nan_action = self._batch._iterator.trainer.nan_action

            if nan_action != "none":
                value = args[1]
                valid = True
                if torch.is_tensor(value):
                    valid = torch.all(torch.isfinite(value))
                elif hasattr(value, "__iter__"):
                    for elem in value:
                        if torch.is_tensor(elem):
                            if not torch.all(torch.isfinite(elem)):
                                valid = False
                                break
                        else:
                            if not np.all(np.isfinite(value)):
                                valid = False
                                break
                else:
                    valid = np.all(np.isfinite(value))

                if not valid:
                    if nan_action in ["stop", "raise", "warn+stop"]:
                        self._batch.terminate(divergence_error=nan_action == "raise")
                    if nan_action in ["warn", "warn+stop"]:
                        msg = f"divergence at batch {idx}, sample {sample_idx}"
                        warnings.warn(msg, DivergenceWarning)

        report = getattr(self._tracker.report, name)
        report(args[0], idx, *args[1:], **kwargs)

        target_dict = getattr(self._tracker.history, name)
        n = len(target_dict["batch"][-1])

        report(
            "sample",
            self._batch.idx,
            sample_idx + torch.arange(n),
            meld=True,
            overwrite=True,
        )


class TrainerBatch:
    """Handle for one training or evaluation batch.
    
    This helps train or evaluate a network and can also be used to monitor tensor values
    during the training or evaluation.

    The monitoring facility is used by employing the `report` construction; e.g.,
        batch.report.weight("W", net.W[0])
    reports the networks first `W` tensor inside the `weight` namespace under the name
    `"W"`. `batch.report` automatically assigns `batch` index and `sample` index to the
    reported value.

    A check for invalid values can be done automatically like this:
        batch.report.score("L", some_loss(net), check_nan=True)
    Note that despite the name, this checks for either `nan`s or infinities in the
    values that are reported. If such an invalid value is found, `batch.terminate()` is
    called so that the iteration ends on the next step. Depending on the trainer
    settings, the termination can either be silent, or it can be accompanied by a
    warning or an exception. See `Trainer`.

    Attributes:
    :param x: input batch
    :param y: output batch
    :param idx: training batch index -- note that this will refer to the associated
        training batch even if `self` is an evaluation batch
    :param sample_idx: training sample index for the start of the batch -- note that
        this will refer to the associated training batch even if `self` is an evaluation
        batch
    :param n: total number of training batches
    :param training: whether this is a training or evaluation batch.
    :param tracker: object used for monitoring values
    :param report: tool used to monitor values
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        idx: int,
        n: int,
        sample_idx: int,
        training: bool,
        tracker: "Tracker",
        iterator: "_TrainerIterator",
    ):
        self.x = x
        self.y = y

        assert len(self.x) == len(self.y)

        self.idx = idx
        self.n = n
        self.sample_idx = sample_idx
        self.training = training
        self.tracker = tracker
        self.report = _BatchReporter(self)

        self._iterator = iterator
        self._trainer = self._iterator.trainer

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

    def terminate(self, divergence_error: bool = False):
        """Terminate the run early.
        
        Note that this does not stop the iteration instantly, but instead ends it the
        first time a new batch is requested. Put differently, the remaining of the `for`
        loop will still be run before it terminates.

        :param divergence_error: if true, raises a `DivergenceError` the next time a new
            batch is requested
        """
        self._iterator.terminating = True
        if divergence_error:
            self._iterator.divergence = True

    def __len__(self) -> int:
        """Number of samples in batch."""
        return len(self.x)


class _TrainerIterator:
    """Iterator used by Trainer."""

    def __init__(self, iterable: "TrainerIterable"):
        self.iterable = iterable
        self.trainer = self.iterable.trainer
        self.loader = self.iterable.loader
        self.n_batches = self.iterable.n_batches
        self.training = self.iterable.training

        self.tracker = self.trainer.tracker

        self.i = 0
        self.sample = 0
        self.it = iter(self.loader)

        self.terminating = False
        self.divergence = False

    def __next__(self) -> TrainerBatch:
        if self.i < self.n_batches and not self.terminating:
            try:
                x, y = next(self.it)
            except StopIteration:
                self.it = iter(self.loader)
                x, y = next(self.it)

            batch = TrainerBatch(
                x=x,
                y=y,
                idx=self.i,
                n=self.n_batches,
                sample_idx=self.sample,
                training=self.training,
                tracker=self.tracker,
                iterator=self,
            )
            self.i += 1
            self.sample += len(batch)

            return batch
        else:
            # ensure tracker coalesces history at the end of the iteration
            if self.training:
                # ...but only if it is the end of the *training* run!
                self.tracker.finalize()

                if self.divergence:
                    raise DivergenceError(
                        f"divergence at batch {self.i}, sample {self.sample}"
                    )
            raise StopIteration


class TrainerIterable:
    """Iterable returned by calling a Trainer.
    
    Iterating through this yields `TrainerBatch`es. At the end of iteration, the
    `Trainer`'s `Tracker`'s `finalize` method is called to prepare the results for easy
    access.
    """

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
        return _TrainerIterator(self)

    def __len__(self) -> int:
        return self.n_batches


class TrainerEvaluateIterable(TrainerIterable):
    """Iterable for running a validation run."""

    def __init__(self, trainer: "Trainer", loader: Iterable):
        super().__init__(trainer, len(loader), training=False, loader=loader)

    def run(self, net):
        """Run a full validation run.
        
        This is shorthand for:
            for batch in self:
                batch.feed(net)
        """
        for batch in self:
            batch.feed(net)


class Trainer:
    """Class used to help train predictive-coding networks.

    Calling a `Trainer` object returns a `TrainerIterable`. Iterating through that
    iterable yields `TrainerBatch` objects, which can be used to train the network and
    report values to a `Tracker`.
    
    Attributes
    :param loader: iterable returning pairs of input and output batches
    :param tracker: `Tracker` object used for keeping track of reported tensors;
        normally this should not be accessed directly; use the `TrainerBatch.report`
        mechanism to report and use `history` to access the results
    :param history: reference to the tracker's history namespace
    """

    def __init__(self, loader: Iterable, nan_action: str = "none"):
        """Initialize trainer.
        
        :param loader: iterable of `(input, output)` tuples
        :param nan_action: action to take in case a check for invalid values fails:
            "none":         do nothing
            "stop":         stop run silently
            "warn":         print a warning and continue
            "warn+stop":    print a warning and stop
            "raise":        raise `DivergenceError`
        """
        self.loader = loader
        self.tracker = Tracker(index_name="batch")
        self.history = self.tracker.history
        self.nan_action = "none"

    def __call__(self, n_batches: int) -> TrainerIterable:
        return TrainerIterable(self, n_batches)

    def __len__(self) -> int:
        """Trainer length equals the length of the loader."""
        return len(self.loader)

    def evaluate(self, val_loader: Iterable) -> TrainerEvaluateIterable:
        return TrainerEvaluateIterable(self, val_loader)
