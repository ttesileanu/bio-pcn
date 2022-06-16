"""Define utilities for training predictive-coding models. """

from types import SimpleNamespace
import torch
import numpy as np

import warnings

from typing import Iterable, Union, Callable

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


def _check_valid(
    field: Union[str, dict],
    value: Union[None, int, float, Iterable, torch.Tensor, np.ndarray] = None,
    **kwargs,
) -> bool:
    """Check whether all values in `value` or in the keys of `field` (if `field` is
    `dict`) are finite.
    """
    if not isinstance(field, str):
        assert value is None
        for crt_field, crt_value in field.items():
            if not _check_valid(crt_field, crt_value, **kwargs):
                return False
        return True

    assert value is not None

    valid = True
    if torch.is_tensor(value):
        valid = torch.all(torch.isfinite(value))
    elif isinstance(value, np.ndarray):
        valid = np.all(np.isfinite(value))
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
        valid = np.isfinite(value)

    return valid


class _BatchReporter:
    """Object used to implement the `report` framework for `TrainerBatch`."""

    def __init__(self, batch: "TrainingBatch", name: str):
        self._batch = batch
        self._name = name
        self._tracker = self._batch._tracker

    def report(self, *args, **kwargs):
        if kwargs.pop("check_invalid", False):
            self._report_invalid(*args, **kwargs)

        idx = self._batch.idx
        sample_idx = self._batch.sample_idx
        epoch = self._batch.epoch
        reporter = getattr(self._tracker, self._name)
        reporter.report((idx, sample_idx, epoch), *args, **kwargs)

    def accumulate(self, *args, **kwargs):
        reporter = getattr(self._tracker, self._name)
        reporter.accumulate(*args, **kwargs)

    def report_accumulated(self, *args, **kwargs):
        idx = self._batch.idx
        sample_idx = self._batch.sample_idx
        epoch = self._batch.epoch

        reporter = getattr(self._tracker, self._name)
        reporter.report_accumulated((idx, sample_idx, epoch), *args, **kwargs)

    def report_batch(self, *args, **kwargs):
        if kwargs.pop("check_invalid", False):
            self._report_invalid(*args, **kwargs)

        batch_size = len(self._batch)

        idx = self._batch.idx
        epoch = self._batch.epoch
        sample_idxs = self._batch.sample_idx + np.arange(batch_size)
        reporter = getattr(self._tracker, self._name)
        reporter.report((idx, sample_idxs, epoch), *args, meld=True, **kwargs)

    def _report_invalid(self, *args, **kwargs):
        invalid_action = self._batch._trainer.invalid_action
        if invalid_action != "none":
            valid = _check_valid(*args, **kwargs)
            if not valid:
                if invalid_action in ["stop", "raise", "warn+stop"]:
                    self._batch.terminate(divergence_error=invalid_action == "raise")
                if invalid_action in ["warn", "warn+stop"]:
                    idx = self._batch.idx
                    sample_idx = self._batch.sample_idx
                    msg = f"divergence at batch {idx}, sample {sample_idx}"
                    warnings.warn(msg, DivergenceWarning)


class TrainingBatch(Batch):
    """Handle for one training batch.
    
    This helps train a network and can also be used to monitor tensor values.

    The monitoring facility is used by employing the `report` construct on arbitrarily
    named fields (though see below):
        batch.weight.report("W", net.W[0])
    reports the network's first `W` tensor inside the `weight` dictionary under the name
    `"W"`. `batch.report` automatically assigns `batch` index and `sample` index to the
    reported value.

    Batches of results -- one for each sample -- can be reported using `report_batch`:
        batch.latent.report_batch("z", net.z)
    The number of samples in the reported values must match the numer of samples in the
    value being reported. Note that this is similar to `report` with `meld = True`,
    except the sample indices will be stored appropriately (increasing for each sample),
    whereas `report(..., meld=True)` will assign the same sample index to the entire
    batch.

    A check for invalid values can be done automatically like this:
        batch.score.report("L", some_loss(net), check_invalid=True)
    Invalid value here means either `nan`s or infinity. If such an invalid value is
    found, `batch.terminate()` is called so that the iteration ends on the next step.
    Depending on the trainer settings, the termination can either be silent, or it can
    be accompanied by a warning or an exception. See `Trainer`.

    Values can be accumulated before reporting,
        batch.score.accumulate("L", some_loss(net))
    After any desired amount of accumulation, the values can be summarized and reported:
        batch.score.report_accumulated()

    Some metrics, such as `pc_loss`, are automatically calculated and recorded for every
    training batch, stored in `history.all_train`. They are also accumulated so that an
    entry of the average metric is made in `history.train` every time an evaluation run
    ends (i.e., an evaluation iterator is used until the end).

    Adding another metric (or removing `pc_loss`) can be achieved by directly accessing
    the `metrics` dictionary of the `Trainer` object. See `Trainer` doc.

    Attributes:
    :param x: input batch
    :param y: output batch
    :param idx: batch index
    :param sample_idx: sample index for the start of the batch
    :param n: total number of training batches in the run
    :param epoch: current "epoch" (how many times the dataset was traversed)
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
        epoch: int,
        iterator: "TrainingIterable",
    ):
        super().__init__(x, y)

        self.idx = idx
        self.n = n
        self.sample_idx = sample_idx
        self.epoch = epoch

        self._trainer = iterator.trainer
        self._tracker = self._trainer.tracker
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

        # evaluate and store metrics, such as pc_loss
        metrics = self._trainer.metrics
        for metric_name, metric_fct in metrics.items():
            metric = metric_fct(ns, net)
            self.all_train.report(metric_name, metric)
            self.train.accumulate(metric_name, metric)

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
        return EvaluationIterable(self._trainer, val_loader, self)

    def __getattr__(self, name: str):
        return _BatchReporter(self, name)

    def __repr__(self) -> str:
        s = (
            f"TrainingBatch("
            f"x={self.x}, y={self.y}, "
            f"idx={self.idx}, n={self.n}, sample_idx={self.sample_idx}, "
            f"epoch={self.epoch})"
        )
        return s


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
        self._epoch = 0

    def __iter__(self) -> "TrainingIterable":
        self.terminating = False
        self.divergence = False

        self._i = 0
        self._sample = 0
        self._epoch = 0
        self._it = iter(self.loader)
        return self

    def __next__(self) -> TrainingBatch:
        if self._i < self.n_batches and not self.terminating:
            try:
                x, y = next(self._it)
            except StopIteration:
                self._it = iter(self.loader)
                x, y = next(self._it)
                self._epoch += 1

            batch = TrainingBatch(
                x=x,
                y=y,
                idx=self._i,
                n=self.n_batches,
                sample_idx=self._sample,
                epoch=self._epoch,
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

    def __repr__(self) -> str:
        s = (
            f"TrainingIterable("
            f"trainer={repr(self.trainer)}, "
            f"loader={repr(self.loader)}, "
            f"n_batches={self.n_batches}, "
            f"terminating={self.terminating}, "
            f"divergence={self.divergence}"
            f")"
        )
        return s

    def __str__(self) -> str:
        s = (
            f"TrainingIterable("
            f"trainer={str(self.trainer)}, "
            f"loader={str(self.loader)}, "
            f"n_batches={self.n_batches}"
            f")"
        )
        return s


class EvaluationBatch(Batch):
    """Handle for one evaluation batch.
    
    This contains a reference to the associated `train_batch` in addition to the usual
    `Batch` attributes.

    The metrics from `Trainer.metrics` are automatically calculated, averaged over all
    validation batches, and stored in `history.validation`. The end of the evaluation
    iteration also triggers reporting of averaged training-mode metrics between the last
    and current evaluation rounds (to be stored in `history.train`). See `Trainer`.
    """

    def __init__(
        self, x: torch.Tensor, y: torch.Tensor, train_batch: TrainingBatch,
    ):
        super().__init__(x, y)

        self.train_batch = train_batch

    def feed(self, net, **kwargs) -> SimpleNamespace:
        """Feed the batch to the network's `relax` method and collect validation-test
        metrics.

        :param net: network to feed the batch to
        :param **kwargs: additional arguments to pass to `net.relax()`
        :return: namespace containing the batch's `x` and `y`, as well as the output
        from `net.relax()`, in `fast`.
        """
        ns = super().feed(net, **kwargs)

        # evaluate and store metrics, such as pc_loss
        metrics = self.train_batch._trainer.metrics
        for metric_name, metric_fct in metrics.items():
            metric = metric_fct(ns, net)
            self.train_batch.validation.accumulate(metric_name, metric)

        return ns

    def __repr__(self) -> str:
        s = (
            f"EvaluationBatch("
            f"x={self.x}, y={self.y}, train_batch={repr(self.train_batch)}"
            f")"
        )
        return s

    def __str__(self) -> str:
        s = f"EvaluationBatch(x={self.x}, y={self.y})"
        return s


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
                batch.latent.accumulate("z", ns.z)
            batch.latent.report_accumulated()
            
            optimizer.step()
    
    The validation run can also be run in a single line:
        eval_iterable.run(val_loader)
    """

    def __init__(
        self, trainer: "Trainer", loader: Iterable, train_batch: TrainingBatch
    ):
        self.trainer = trainer
        self.loader = loader
        self.train_batch = train_batch
        self._it = None

    def __iter__(self) -> "EvaluationIterable":
        self._it = iter(self.loader)
        return self

    def __next__(self) -> EvaluationBatch:
        try:
            x, y = next(self._it)
        except StopIteration:
            self.train_batch.validation.report_accumulated()
            self.train_batch.train.report_accumulated()
            raise StopIteration
        return EvaluationBatch(x=x, y=y, train_batch=self.train_batch)

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

    def __repr__(self) -> str:
        s = (
            f"EvaluationIterable("
            f"trainer={repr(self.trainer)}, "
            f"loader={repr(self.loader)}, "
            f"train_batch={repr(self.train_batch)}"
            f")"
        )
        return s

    def __str__(self) -> str:
        s = (
            f"EvaluationIterable("
            f"trainer={str(self.trainer)}, "
            f"loader={str(self.loader)}, "
            f"train_batch={str(self.train_batch)}"
            f")"
        )
        return s


class Trainer:
    """Class used to help train predictive-coding networks.

    Calling a `Trainer` object returns a `TrainerIterable`. Iterating through that
    iterable yields `TrainerBatch` objects, which can be used to train the network and
    report values to a `Tracker`.

    The `Trainer` evaluates certain metrics (see the `metrics` attribute below) for
    every training and evaluation batch, storing the training values in `all_train`, and
    averaging the scores for the evaluation batches into `validation`. The training
    results between consecutive validation runs are averaged and stored in `train`.
    
    Attributes
    :param loader: iterable returning pairs of input and output batches
    :param tracker: `Tracker` object used for keeping track of reported tensors;
        normally this should not be accessed directly; use the `TrainerBatch.foo.report`
        mechanism to report and use `history` to access the results
    :param history: reference to the tracker's history namespace
    :param metrics: dictionary of metrics to be calculated during both training and
        evaluation; the keys are strings (the names of the metrics), the values are
        callables with signature
            metric(ns, net) -> float
        where `ns` is the output from `batch.feed` and `net` is the network that is
        being trained.
    """

    def __init__(self, loader: Iterable, invalid_action: str = "none"):
        """Initialize trainer.
        
        :param loader: iterable of `(input, output)` tuples
        :param invalid_action: action to take in case a check for invalid values fails:
            "none":         do nothing
            "stop":         stop run silently
            "warn":         print a warning and continue
            "warn+stop":    print a warning and stop
            "raise":        raise `DivergenceError`
        """
        self.loader = loader
        self.tracker = Tracker(index_name=("batch", "sample", "epoch"))
        self.history = self.tracker.history
        self.invalid_action = invalid_action
        self.metrics = {"pc_loss": lambda ns, net: net.pc_loss(ns.fast.z).item()}

    def __call__(self, n_batches: int) -> TrainingIterable:
        return TrainingIterable(self, n_batches)

    def __len__(self) -> int:
        """Trainer length equals the length of the loader."""
        return len(self.loader)

    def __repr__(self) -> str:
        s = (
            f"Trainer("
            f"loader={repr(self.loader)}, "
            f"tracker={repr(self.tracker)}, "
            f"metrics={repr(self.metrics)}"
            f")"
        )
        return s

    def __str__(self) -> str:
        s = (
            f"Trainer("
            f"loader={str(self.loader)}, "
            f"metrics={str(self.metrics)}"
            f")"
        )
        return s


def multi_lr(optim: Callable, parameter_groups: list, lr_factors: dict, **kwargs):
    """Instantiate an optimizer with different learning rates for different parameter
    groups.

    If the names in `parameter_groups` are "layered", in the sense that we have the same
    suffix followed by a colon followed by a layer identifier (e.g., "W:0", "W:1", ...),
    then all the layers sharing a suffix can be addressed using the suffix alone (e.g.,
    "W").
    
    :param optim: constructor for a learning rate optimizer, e.g., `torch.optim.SGD`
    :param parameter_groups: list of parameter groups (each a `dict` with keys `"name"`
        and `"params"`)
    :param lr_factors: dictionary of learning-rate factors; any parameter group that is
        not mentioned here is assumed to have a factor of 1
    :**kwargs: additional keyword arguments passed to `optim()`
    """
    # slightly annoying bit: if default learning rate is used, we can't read it from
    # `kwargs`; so we first create an optimizer with all the relevant groups, but the
    # same learning rate for all
    optim_instance = optim(parameter_groups, **kwargs)
    for param in optim_instance.param_groups:
        if param["name"] in lr_factors:
            param["lr"] *= lr_factors[param["name"]]
        else:
            # do we have an across-layer factor?
            for factor_name, factor in lr_factors.items():
                if param["name"].startswith(f"{factor_name}:"):
                    param["lr"] *= factor

    return optim_instance
