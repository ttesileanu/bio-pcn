"""Define utilities for training predictive-coding models. """

from types import SimpleNamespace
import torch
import numpy as np

import warnings

from typing import Iterable

from cpcn.track import Tracker


class DivergenceError(Exception):
    pass


class DivergenceWarning(Warning):
    pass


class Batch:
    """Handle for one batch of a dataset.
    
    This helps evaluate a network on a batch.

    Attributes:
    :param x: input batch
    :param y: output batch
    """

    def __init__(
        self, x: torch.Tensor, y: torch.Tensor,
    ):
        self.x = x
        self.y = y

        assert len(self.x) == len(self.y)

    def feed(self, net, **kwargs) -> SimpleNamespace:
        """Feed the batch to the network's `relax` method.

        :param net: network to feed the batch to
        :param **kwargs: additional arguments to pass to `net.relax()`
        :return: namespace containing the batch's `x` and `y`, as well as the output
        from `net.relax()`, in `fast`.
        """
        # run fast dynamics
        res = net.relax(self.x, self.y, **kwargs)
        ns = SimpleNamespace(x=self.x, y=self.y, fast=res)

        return ns

    def __len__(self) -> int:
        """Number of samples in batch."""
        return len(self.x)


class _BatchReporter:
    """Object used to implement the `report` framework for `TrainerBatch`."""

    def __init__(self, batch: "TrainingBatch"):
        self._batch = batch
        self._tracker = self._batch.tracker

    def __getattr__(self, name: str):
        return lambda *args, _name=name, **kwargs: self._report(_name, *args, **kwargs)

    def _report(self, name: str, *args, **kwargs):
        idx = self._batch.idx
        sample_idx = self._batch.sample_idx

        if kwargs.pop("check_invalid", False):
            nan_action = self._batch.trainer.nan_action

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

        report("sample", idx, sample_idx + torch.arange(n), meld=True, overwrite=True)


class TrainingBatch(Batch):
    """Handle for one training batch.
    
    This helps train a network and can also be used to monitor tensor values.

    The monitoring facility is used by employing the `report` construct; e.g.,
        batch.report.weight("W", net.W[0])
    reports the network's first `W` tensor inside the `weight` namespace under the name
    `"W"`. `batch.report` automatically assigns `batch` index and `sample` index to the
    reported value.

    A check for invalid values can be done automatically like this:
        batch.report.score("L", some_loss(net), check_invalid=True)
    Invalid value here means either `nan`s or infinity. If such an invalid value is
    found, `batch.terminate()` is called so that the iteration ends on the next step.
    Depending on the trainer settings, the termination can either be silent, or it can
    be accompanied by a warning or an exception. See `Trainer`.

    Attributes:
    :param x: input batch
    :param y: output batch
    :param idx: batch index
    :param sample_idx: sample index for the start of the batch
    :param n: total number of training batches in the run
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
        iterator: "TrainingIterable",
    ):
        super().__init__(x, y)

        self.idx = idx
        self.n = n
        self.sample_idx = sample_idx

        self.trainer = iterator.trainer
        self.tracker = self.trainer.tracker
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
        ns = super().feed(net, **kwargs)

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

    def evaluate(self, val_loader: Iterable) -> "EvaluationIterable":
        """Generate an iterable through a validation set.
        
        See `EvaluationIterable`.
        """
        return EvaluationIterable(self.trainer, val_loader)


class TrainingIterable:
    """Iterable returned by calling a Trainer, as well as corresponding iterator.
    
    Iterating through this yields `TrainingBatch`es. At the end of iteration, the
    `Trainer`'s `Tracker`'s `finalize` method is called to prepare the results for easy
    access.
    """

    def __init__(self, trainer: "Trainer", n_batches: int):
        self.trainer = trainer
        self.n_batches = n_batches

        self.loader = self.trainer.loader

        self.terminating = False
        self.divergence = False

        self._it = None
        self._i = 0
        self._sample = 0

    def __iter__(self) -> "TrainingIterable":
        self.terminating = False
        self.divergence = False

        self._i = 0
        self._sample = 0
        self._it = iter(self.loader)
        return self

    def __next__(self) -> TrainingBatch:
        if self._i < self.n_batches and not self.terminating:
            try:
                x, y = next(self._it)
            except StopIteration:
                self._it = iter(self.loader)
                x, y = next(self._it)

            batch = TrainingBatch(
                x=x,
                y=y,
                idx=self._i,
                n=self.n_batches,
                sample_idx=self._sample,
                iterator=self,
            )
            self._i += 1
            self._sample += len(batch)

            return batch
        else:
            # ensure tracker coalesces history at the end of the iteration
            self.trainer.tracker.finalize()

            if self.divergence:
                raise DivergenceError(
                    f"divergence at batch {self._i}, sample {self._sample}"
                )
            raise StopIteration

    def __len__(self) -> int:
        return self.n_batches


class EvaluationIterable:
    """Iterable and corresponding iterator used for evaluation runs.
    
    Iterating through this yields `EvaluationBatch`es.
    
    Can be used as:
        losses = []
        for val_batch in eval_iterable:
            ns = val_batch.feed(net)
            losses.append(net.pc_loss(val_batch.ns))

    More commonly this is used as part of a training run,
        trainer = Trainer(train_loader)
        for batch in trainer(n_batches):
            batch.feed()
            for val_batch in batch.evaluate(val_loader):
                ns = val_batch.feed(net)
                batch.report.latent("z", ns.z)
            
            optimizer.step()
    
    The validation run can also be run in a single line:
        eval_iterable.run(val_loader)
    """

    def __init__(self, trainer: "Trainer", loader: Iterable):
        self.trainer = trainer
        self.loader = loader
        self._it = None

    def __iter__(self) -> "EvaluationIterable":
        self._it = iter(self.loader)
        return self

    def __next__(self) -> Batch:
        x, y = next(self._it)
        return Batch(x=x, y=y)

    def run(self, net):
        """Run a full validation run.
        
        This is shorthand for:
            for batch in self:
                batch.feed(net)
        """
        for batch in self:
            batch.feed(net)

    def __len__(self) -> int:
        return len(self.loader)


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
        self.nan_action = nan_action

    def __call__(self, n_batches: int) -> TrainingIterable:
        return TrainingIterable(self, n_batches)

    def __len__(self) -> int:
        """Trainer length equals the length of the loader."""
        return len(self.loader)
