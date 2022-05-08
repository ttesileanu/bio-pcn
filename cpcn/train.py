"""Define utilities for training predictive-coding models. """

from types import SimpleNamespace
from typing import Optional, Callable, Iterable, Union, Sequence, Tuple
from collections import OrderedDict

import torch
import numpy as np

from .util import one_hot_accuracy, hierarchical_get, pretty_size


class Trainer:
    """A class for training (constrained or not) predictive-coding networks.
    
    Attributes
    :param net: the network to train
    :param train_loader: data loader for training set
    :param validation_loader: data loader for validation set
    :param history: history from last call to `run()`
    :param optimizer_class: callable to create the optimizer to use; default: Adam
    :param optimizer_kwargs: keyword arguments to pass to the `optimizer()`
    :param accuracy_fct: function used to calculate accuracy; called as
        `accuracy_fct(y, y_pred)`, where `y` is ground truth, `y_pred` is prediction
    :param classifier: this can be a trainable neural network used to predict the output
        samples from the `dim`th layer of `net` after a call to `net.forward()`; it can
        also be a string or `None` -- see `set_classifier()`
    :param classifier_optim_class: callable to create the optimizer for the classifier
    :param classifier_optim_kwargs: keyword arguments to pass to `classifier_optim()`
    :param classifier_criterion: objective fucntion for training classifier; default:
        `MSELoss()`
    :param classifier_dim: which layer of `net` to pass into the classifier
    :param observers: list of tuples `(observer, condition)`; see `add_observer()`
    :param validation_condition: callable identifying batches where a validation run is
        performed
    :param history: namespace of history data for the last call to `run()`; see the
        `peek...` functions
    :param schedulers: list of tuples `(scheduler, condition)` of learning-rate
        scheduler constructors and the conditions under which they should be called
    :param lr_factors: dictionary of learning-rate scaling factors
    """

    def __init__(self, net, train_loader: Iterable, validation_loader: Iterable):
        """Initialize the trainer.
        
        :param net: the network to train
        :param train_loader: data loader for training set
        :param validation_loader: data loader for validation set
        """
        self.net = net
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.optimizer_class = torch.optim.Adam
        self.optimizer_kwargs = {}
        self.lr_factors = {}

        self.accuracy_fct = one_hot_accuracy

        self.classifier = None
        self.classifier_optim_class = torch.optim.Adam
        self.classifier_optim_kwargs = {}
        self.classifier_criterion = None
        self.classifier_dim = -2

        self.observers = []
        self.schedulers = []
        self.validation_condition = None  # see call to `peek_validation` below

        # these will be set by `run`
        self._last_val = None
        self._n_batches = None
        self._optimizer = None

        # set up the history with storage for tracking the loss and accuracy
        # (and learning rate)
        self.history = SimpleNamespace()
        self._setup_history("all_train", ["pc_loss", "accuracy", "lr"], "batch", size=1)
        self._setup_history("train", ["pc_loss", "accuracy"], "batch", size=1)
        self._setup_history("validation", ["pc_loss", "accuracy"], "batch", size=1)

        # by default: run on validation test once per epoch
        self.peek_validation(every=len(self.train_loader))

    def run(
        self,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        progress: Optional[Callable] = None,
    ) -> SimpleNamespace:
        """Run the training.
        
        :param n_epochs: number of training epochs; either this or `n_batches` must be
            given
        :param n_batches: number of training batches; if larger than the number of
            batches in the dataset, a new epoch is started; if `n_batches` is provided,
            it overrides `n_epochs`
        :param progress: progress indicator; should have `tqdm` interface
        :return: `self.history`, which contains outputs from any monitors registered
            with the `self.peek...` functions, as well as the progress of the predictive
            coding loss (`"pc_loss"`) and `"accuracy"` for every training batch (in
            `all_train`) and for every validation checkpoint (in `validation`); the
            average training-set `"pc_loss"` and `"accuracy"` calculated between every
            pair of validation checkpoints is included in `train`; `all_train` also
            keeps track of the learning rate, in `"lr"`
        """
        # initialization work
        optimizer, classifier_optim = self._setup_optimizers()
        self._optimizer = optimizer

        self.net.train()
        if self.classifier:
            self.classifier.train()

        self._reset_history()

        # construct learning-rate schedulers
        schedulers = []
        for constructor, condition in self.schedulers:
            schedulers.append((constructor(optimizer), condition))

        # set up progress bar, if any
        if n_batches is None:
            n_train = len(self.train_loader)
            n_batches = n_epochs * n_train
        if progress is not None:
            pbar = progress(total=n_batches)
        else:
            pbar = None

        # run the training
        it = iter(self.train_loader)
        self._n_batches = n_batches
        self._last_val = 0
        epoch = 0
        for batch in range(n_batches):
            # get next batch, restarting the training-set iteration if needed
            try:
                x, y = next(it)
            except StopIteration:
                # start over
                it = iter(self.train_loader)
                x, y = next(it)
                epoch += 1

            # train main net, asking for loss+latent profile if needed for obsevers
            observers, need_profile = self._get_observers(batch)
            batch_results = self.net.relax(
                x, y, pc_loss_profile=need_profile, latent_profile=need_profile
            )
            with torch.no_grad():
                pc_loss = self.net.pc_loss(batch_results.z).item()
            self.net.calculate_weight_grad(batch_results)
            optimizer.step()

            # use and train classifier, if we have one
            y_pred = self._get_and_train_prediction(x, y, classifier_optim)

            with torch.no_grad():
                # keep track of per-batch indicators
                accuracy = self.accuracy_fct(y_pred, y)

                # call any batch observers; this also stores training-mode diagnostics
                self._report(observers, epoch, batch, pc_loss, accuracy, batch_results)
                if self.validation_condition(batch):
                    # this fills out the `validation` and `train` fields of history
                    self._validation_run(epoch, batch)

                # run learning-rate schedulers
                for scheduler, condition in schedulers:
                    if condition(batch):
                        scheduler.step()

                # update progress bar, if any
                if pbar is not None:
                    pbar.update()
                    progress_info = self._pbar_report(epoch, batch)
                    pbar.set_postfix(progress_info, refresh=False)

            batch += 1

        # convert all history information to tensors
        self._coalesce_history()

        if pbar is not None:
            pbar.close()

        return self.history

    def _setup_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
        """Set up optimizers for the main network and for the classifier, if any."""
        if len(self.lr_factors) == 0:
            optimizer = self.optimizer_class(
                self.net.slow_parameters(), **self.optimizer_kwargs
            )
        else:
            # use per-group learning rate scaling factors
            param_groups = self.net.slow_parameter_groups()

            # slightly annoying bit: if default learning rate is used, we can't read it
            # from `self.optimizer_kwargs`; so we first create an optimizer with all
            # the relevant groups, but the same learning rate for all
            optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)
            for param in optimizer.param_groups:
                if param["name"] in self.lr_factors:
                    param["lr"] *= self.lr_factors[param["name"]]

        if self.classifier is not None:
            if self.classifier_criterion is None:
                self.classifier_criterion = torch.nn.MSELoss()

            classifier_optim = self.classifier_optim_class(
                self.classifier.parameters(), **self.classifier_optim_kwargs
            )
        else:
            classifier_optim = None

        return optimizer, classifier_optim

    def _get_observers(self, batch: int) -> Tuple[list, bool]:
        """Get list of observers that should be called given the batch index. Also find
        whether any of the observers need a loss profile calculation.
        """
        observers = []
        need_profile = False
        for observer, condition, profile in self.observers:
            if condition(batch):
                observers.append(observer)
                if profile:
                    need_profile = True

        return observers, need_profile

    def _get_and_train_prediction(
        self, x: torch.Tensor, y: torch.Tensor, classifier_optim: Callable
    ) -> torch.Tensor:
        """Get prediction, using the classifier (if any), and train the classifier (if
        any).
        """
        z_fwd = self.net.forward(x)
        if self.classifier is not None:
            y_pred = self.classifier(z_fwd[self.classifier_dim])

            classifier_optim.zero_grad()
            classifier_loss = self.classifier_criterion(y_pred, y)
            classifier_loss.backward()
            classifier_optim.step()
        else:
            y_pred = z_fwd[-1]

        return y_pred

    def _report(
        self,
        observers: list,
        epoch: int,
        batch: int,
        pc_loss: float,
        accuracy: float,
        batch_results: SimpleNamespace,
    ):
        """Send per-batch information to a list of observers, and store training-mode
        diagnostics in `self.history`.
        """
        if len(observers) > 0:
            batch_ns = SimpleNamespace(
                epoch=epoch,
                batch=batch,
                net=self.net,
                batch_results=batch_results,
                classifier=self.classifier,
                trainer=self,
                train_loss=pc_loss,
                train_accuracy=accuracy,
            )
            for observer in observers:
                observer(batch_ns)

        # keep track of loss and accuracy
        target_dict = self.history.all_train
        target_dict["epoch"].append(epoch)
        target_dict["batch"].append(batch)
        target_dict["pc_loss"].append(torch.FloatTensor([pc_loss]))
        target_dict["accuracy"].append(torch.FloatTensor([accuracy]))
        lr = self._optimizer.param_groups[0]["lr"]
        target_dict["lr"].append(torch.FloatTensor([lr]))

    def _validation_run(self, epoch: int, batch: int):
        # evaluate performance on validation set
        self.net.eval()
        if self.classifier:
            self.classifier.eval()
        pc_loss, accuracy = evaluate(
            self.net,
            self.validation_loader,
            accuracy_fct=self.accuracy_fct,
            classifier=self.classifier,
            classifier_dim=self.classifier_dim,
        )
        pc_loss = torch.mean(torch.FloatTensor([pc_loss]))
        accuracy = torch.mean(torch.FloatTensor([accuracy]))

        target_dict = self.history.validation
        target_dict["epoch"].append(epoch)
        target_dict["batch"].append(batch)
        target_dict["pc_loss"].append(torch.FloatTensor([pc_loss]))
        target_dict["accuracy"].append(torch.FloatTensor([accuracy]))

        # fill out average performance on training set
        all_train = self.history.all_train
        last_losses = all_train["pc_loss"][self._last_val :]
        last_accuracies = all_train["accuracy"][self._last_val :]
        avg_pc_loss = torch.mean(torch.FloatTensor([last_losses]))
        avg_accuracy = torch.mean(torch.FloatTensor([last_accuracies]))
        target_dict = self.history.train
        target_dict["epoch"].append(epoch)
        target_dict["batch"].append(batch)
        target_dict["pc_loss"].append(torch.FloatTensor([avg_pc_loss]))
        target_dict["accuracy"].append(torch.FloatTensor([avg_accuracy]))
        self._last_val = len(all_train["pc_loss"])

        # switch back to training mode
        self.net.train()
        if self.classifier:
            self.classifier.train()

    def _pbar_report(self, epoch: int, batch: int) -> dict:
        """Generate `dict` of progress information for the progress bar."""
        loss_list = self.history.validation["pc_loss"]
        if len(loss_list) > 0:
            val_loss_str = f"{loss_list[-1].item():.2g}"
        else:
            val_loss_str = "???"

        acc_list = self.history.validation["accuracy"]
        if len(acc_list) > 0:
            acc_list_str = f"{acc_list[-1].item():.2f}"
        else:
            acc_list_str = "???"

        progress_info = {
            "epoch": f"{epoch}",
            "val_loss": val_loss_str,
            "val_acc": acc_list_str,
        }

        # add CUDA memory usage, if CUDA is being used
        param = self.net.slow_parameters()[0]
        if param.is_cuda:
            memory = torch.cuda.memory_allocated(param.device)
            progress_info["cuda_mem"] = pretty_size(memory)

        return progress_info

    def __str__(self) -> str:
        s = f"Trainer(net={str(self.net)}, optimizer_class={str(self.optimizer_class)})"
        return s

    def __repr__(self) -> str:
        s = (
            f"Trainer("
            f"net={repr(self.net)}, "
            f"optimizer_class={repr(self.optimizer_class)}, "
            f"classifier={repr(self.classifier)}, "
            f"classifier_dim={self.classifier_dim}, "
            f"classifier_criterion={repr(self.classifier_criterion)}, "
            f"accuracy_fct={repr(self.accuracy_fct)}, "
            f")"
        )
        return s

    def set_classifier(
        self,
        classifier: Union[Callable, None],
        classifier_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "Trainer":
        """Set a classifier to use for predictions.

        This can be a trainable neural network used to predict the output samples from
        the `classifier_dim`th layer of `net` after a call to `net.forward()`.
        
        It can also be a string:
        * "linear": a linear layer of appropriate input and ouptut dimensions;
        * "linear_softmax": linear layer followed by softmax;
        * "linear_relu": linear layer followed by relu.
        * "mlp": a neural net with one hidden layer, with the same size as the output
            layer; the nonlinearity after the hidden layer is ReLU, and the output
            nonlinearity is softmax
        
        You can also set it to `None`, in which case the output from `net.forward()` is
        used.
        
        :param classifier: the classifier to use
        :param classifier_dim: which layer of `net` to pass into the classifier
        :param device: device to send the classifier to; default: the device to which
            the first output of `net.slow_parameters()` is assigned
        """
        self.classifier = classifier
        if isinstance(self.classifier, str):
            if self.classifier in ["linear", "linear_softmax", "linear_relu"]:
                dims = self.net.dims
                linear = torch.nn.Linear(dims[self.classifier_dim], dims[-1])
                if self.classifier == "linear":
                    layers = OrderedDict([("linear", linear)])
                elif self.classifier == "linear_softmax":
                    layers = OrderedDict(
                        [("linear", linear), ("softmax", torch.nn.Softmax(1))]
                    )
                elif self.classifier == "linear_relu":
                    layers = OrderedDict(
                        [("linear", linear), ("relu", torch.nn.ReLU(1))]
                    )
                self.classifier = torch.nn.Sequential(layers)
            elif self.classifier == "mlp":
                dims = self.net.dims

                n_in = dims[self.classifier_dim]
                n_out = dims[-1]

                layers = OrderedDict(
                    [
                        ("linear", torch.nn.Linear(n_in, n_out)),
                        ("relu", torch.nn.ReLU()),
                        ("full", torch.nn.Linear(n_out, n_out)),
                        ("softmax", torch.nn.Softmax(dim=1)),
                    ]
                )
                self.classifier = torch.nn.Sequential(layers)
            else:
                raise ValueError("Invalid classifier type")

        if classifier_dim is not None:
            self.classifier_dim = classifier_dim

        if device is None:
            device = self.net.slow_parameters()[0].device
        self.classifier = self.classifier.to(device)

        return self

    def set_classifier_dim(self, classifier_dim: int) -> "Trainer":
        """Set the layer on which the classifier should act."""
        self.classifier_dim = classifier_dim
        return self

    def set_optimizer(self, optimizer: torch.optim.Optimizer, **kwargs) -> "Trainer":
        """Set the optimizer to be used for training.
        
        More specifically, this sets a callable which creates the optimizer (typically a
        class, like `torch.optim.Adam`). Additional arguments are stored, to be passed
        to `optimizer()` at `run` time.

        :param optimizer: the callable that generates the optimizer
        :param **kwargs: additional arguments to pass to the `optimizer()` call
        """
        self.optimizer_class = optimizer
        self.optimizer_kwargs = kwargs
        return self

    def add_observer(
        self,
        observer: Callable,
        condition: Optional[Callable] = None,
        every: Optional[int] = None,
        count: Optional[int] = None,
        mask: Optional[Sequence] = None,
        profile: bool = False,
    ) -> "Trainer":
        """Set an observer.
        
        An observer is a function called after every training batch that satisfies a
        certain condition. The observer gets called with a namespace argument,
            observer(ns: SimpleNamespace)
        where the `SimpleNamespace` contains
            batch:          the index of the batch that was just processed
            epoch:          an epoch index
            net:            the network that is being optimized
            batch_results:  namespace returned by `net.relax()`
            trainer:        the `Trainer` object, `self`
            classifier:     network used for classifier output (if any)
            train_loss:     (predictive-coding) loss on training set
            train_accuracy: accuracy on training set

            (if `profile == True`)
            pc_loss_profile:the loss profile from `net.relax()`
            latent_profile: the latent profile from `net.relax()`
        
        Note that setting `profile` to true could potentially slow down learning.
        
        :param observer: the observer callback
        :param condition: condition to be fulfilled for the observer to be called; this
            has signature (batch: int) -> bool; must return true to call observer
        :param every: run observer every `every` batches, starting at 0; overridden by
            `condition` or `mask`
        :param count: run observer `count` times during a run; the batches at which the
            observer is run are given by `floor(linspace(0, n_batches - 1, count))`;
            overriden by `condition` or `mask`
        :param mask: store whenever `mask[batch]` is true
        :param profile: whether to request the loss and latent profile from
            `net.relax()`
        """
        condition = self._get_condition_fct(condition, every, count, mask)
        self.observers.append((observer, condition, profile))
        return self

    def set_accuracy_fct(self, accuracy_fct: Callable) -> "Trainer":
        """Set the function used to calculate accuracy scores.
        
        This is called as `accuracy_fct(y, y_pred)`, where `y` is ground truth, `y_pred`
        is prediction.
        """
        self.accuracy_fct = accuracy_fct
        return self

    def set_lr_factor(self, params: Union[Sequence, str], factor: float) -> "Trainer":
        """Set a scaling factor for the learning rate of a particular parameter group.
        
        :param params: name of a single set of parameters, or iterable of such names;
            this must match the names returned from `net.slow_parameter_groups()`
        :param factor: learning-rate factor
        """
        if isinstance(params, str):
            params = [params]

        for param in params:
            self.lr_factors[param] = factor

        return self

    def set_classifier_optimizer(
        self, optimizer: torch.optim.Optimizer, **kwargs
    ) -> "Trainer":
        """Set the optimizer to be used for training the classifier.
        
        More specifically, this sets a callable which creates the optimizer (typically a
        class, like `torch.optim.Adam`). Additional arguments are stored, to be passed
        to `optimizer()` at `run` time.

        :param optimizer: the callable that generates the optimizer
        :param **kwargs: additional arguments to pass to the `optimizer()` call
        """
        self.classifier_optim_class = optimizer
        self.classifier_optim_kwargs = kwargs
        return self

    def set_classifier_criterion(self, criterion=Callable) -> "Trainer":
        """Set the objective fucntion for training the classifier."""
        self.classifier_criterion = criterion
        return self

    def peek(
        self,
        name: str,
        vars: Sequence,
        condition: Optional[Callable] = None,
        every: Optional[int] = None,
        count: Optional[int] = None,
        mask: Optional[Sequence] = None,
    ) -> "Trainer":
        """Add per-batch monitoring.
        
        This is used to store values of parameters after each batch that obeys a
        condition). The values will be stored in `self.history` during calls to
        `self.run()`.

        :param name: the name to be used in `self.history` for the stored values
        :param vars: variables to track; these should be names of attributes of the
            model under training
        :param condition: condition to be fulfilled for values to be stored; see
            `add_observer`
        :param every: store every `every` batches; see `add_observer`
        :param count: store `count` times during a run; see `add_observer`
        :param mask: store whenever `mask[batch]` is true; see `add_observer`
        """
        if hasattr(self.history, name):
            raise ValueError("monitor name already in use")
        self._setup_history(name, vars, "batch")
        return self.add_observer(
            lambda ns, name=name: self._monitor(name, ns),
            condition=condition,
            every=every,
            count=count,
            mask=mask,
        )

    def peek_sample(
        self,
        name: str,
        vars: Sequence,
        condition: Optional[Callable] = None,
        every: Optional[int] = None,
        count: Optional[int] = None,
        mask: Optional[Sequence] = None,
        sample_mask: Optional[Sequence] = None,
    ) -> "Trainer":
        """Add per-sample monitoring.
        
        This is used to store values of parameters that change with each sample -- these
        are the variables returned by `net.relax()`. The values will be stored in
        `self.history` during calls to `self.run()`.

        :param name: the name to be used in `self.history` for the stored values
        :param vars: variables to track; these should be names of attributes contained
            in the namespace returned by `net.relax()`
        :param condition: condition to be fulfilled for values to be stored; see
            `add_observer`
        :param every: store every `every` batches; see `add_observer`
        :param count: store `count` times during a run; see `add_observer`
        :param mask: store whenever `mask[batch]` is true; see `add_observer`
        :param sample_mask: a mask showing which samples within a batch to use
        """
        if hasattr(self.history, name):
            raise ValueError("monitor name already in use")
        self._setup_history(name, vars, "sample")
        return self.add_observer(
            lambda ns, name=name, sel=sample_mask: self._sample_monitor(name, ns, sel),
            condition=condition,
            every=every,
            count=count,
            mask=mask,
        )

    def peek_fast_dynamics(
        self,
        name: str,
        vars: Sequence,
        condition: Optional[Callable] = None,
        every: Optional[int] = None,
        count: Optional[int] = None,
        mask: Optional[Sequence] = None,
        sample_mask: Optional[Sequence] = None,
    ) -> "Trainer":
        """Add monitoring of fast (`relax`) dynamics.
        
        This is used to store the evolution of the latent variables during the fast
        dynamics generated by `net.relax()`. The results are stored in
        `self.history` during calls to `self.run()`.

        :param name: the name to be used in `self.history` for the stored values
        :param vars: variables to track; these should be a subset of the variables
            returned by `net.relax()` when `latent_profile` is true;
        :param condition: condition to be fulfilled for values to be stored; see
            `add_observer`
        :param every: store every `every` batches; see `add_observer`
        :param count: store `count` times during a run; see `add_observer`
        :param mask: store whenever `mask[batch]` is true; see `add_observer`
        :param sample_mask: a mask showing which samples within a batch to use
        """
        if hasattr(self.history, name):
            raise ValueError("monitor name already in use")
        self._setup_history(name, vars, "sample")
        return self.add_observer(
            lambda ns, name=name, sel=sample_mask: self._sub_sample_monitor(
                name, ns, sel
            ),
            condition=condition,
            every=every,
            count=count,
            mask=mask,
            profile=True,
        )

    def peek_validation(
        self,
        condition: Optional[Callable] = None,
        every: Optional[int] = None,
        count: Optional[int] = None,
        mask: Optional[Sequence] = None,
    ) -> "Trainer":
        """Set how often to evaluate on validation set.
        
        The values are stored in `self.history` during calls to `self.run()`.

        :param condition: condition to be fulfilled for values to be stored; see
            `add_observer`
        :param every: store every `every` batches; see `add_observer`
        :param count: store `count` times during a run; see `add_observer`
        :param mask: store whenever `mask[batch]` is true; see `add_observer`
        """
        self.validation_condition = self._get_condition_fct(
            condition, every, count, mask
        )
        return self

    def add_scheduler(
        self,
        scheduler: Callable,
        condition: Optional[Callable] = None,
        every: Optional[int] = None,
        mask: Optional[Sequence] = None,
    ) -> "Trainer":
        """Add a constructor for a learning-rate scheduler.
        
        Calling `scheduler()` with the optimizer as an argument should create a
        learning-rate scheduler. (This is needed since the optimizer is only generated
        during a call to `run()`.) The scheduler is run after every epoch, by default,
        or according to the given `condition`, `every`, or `count`.        

        You can use an inline function to pass additional arguments to standard
        schedulers, e.g.:
            lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        
        :param scheduler: the constructor for the scheduler
        :param condition: condition to be fulfilled for the scheduler to be called; this
            has signature (batch: int) -> bool; default: after every epoch
        :param every: run scheduler every `every` batches; unlike observers, this is not
            run on the 0th batch, but starting at batch `every - 1`; overridden by
            `condition`
        :param mask: run whenever `mask[batch]` is true
        """
        if condition is None:
            if mask is None:
                if every is None:
                    every = len(self.train_loader)
                condition = lambda batch, every=every: (batch + 1) % every == 0
            else:
                condition = lambda batch, mask=mask: mask[batch]
        self.schedulers.append((scheduler, condition))
        return self

    def _get_condition_fct(
        self,
        condition: Optional[Callable],
        every: Optional[int],
        count: Optional[int],
        mask: Optional[Sequence],
    ):
        """Convert various ways of specifying a condition to a function."""
        if condition is None:
            if mask is not None:
                condition = lambda batch, mask=mask: mask[batch]
            elif every is not None:
                condition = lambda batch, every=every: batch % every == 0
            elif count is not None and count > 0:
                condition = lambda batch, count=count: self._count_condition(
                    batch, count
                )
            else:
                condition = lambda _: True

        return condition

    def _count_condition(self, batch: int, count: int) -> bool:
        """Condition function for running exactly `count` times during a run."""
        # should be true when batch = floor(k * (n - 1) / (count - 1)) for integer k
        # this implies (batch * (count - 1)) % (n - 1) == 0 or > (n - count).
        if count == 1:
            return batch == 0
        else:
            n_batches = self._n_batches
            mod = (batch * (count - 1)) % (n_batches - 1)
            return mod == 0 or mod > n_batches - count

    def _monitor(self, name: str, ns: SimpleNamespace):
        """Observer called to update per-epoch or per-batch monitors."""
        target_dict = getattr(self.history, name)
        target_dict["epoch"].append(ns.epoch)
        if "batch" in target_dict:
            target_dict["batch"].append(ns.batch)
        for var, target in target_dict.items():
            if var in ["epoch", "batch"]:
                continue

            var, *parts = var.split(":")
            if "." not in var:
                value = getattr(ns.net, var)
            else:
                value = hierarchical_get(ns, var)
            if len(parts) > 0:
                # this is part of a multi-layer variable
                k = int(parts[-1])
                value = value[k]

            # send to CPU so this does not take up GPU memory!
            target.append(value.detach().to("cpu").clone().unsqueeze(0))

    def _sample_monitor(self, name: str, ns: SimpleNamespace, sel: Optional[Sequence]):
        """Observer called to update per-sample monitors."""
        target_dict = getattr(self.history, name)

        batch_size = None
        for var, target in target_dict.items():
            if var in ["epoch", "batch", "sample"]:
                continue

            var, *parts = var.split(":")
            value = getattr(ns.batch_results, var)
            if len(parts) > 0:
                # this is part of a multi-layer variable
                value = value[int(parts[-1])]

            # handle batch size of 1
            if value.ndim == 1:
                value = value.unsqueeze(0)
            value = value.detach()
            if batch_size is None:
                batch_size = len(value)
            if sel is not None:
                value = value[sel]
            # send to CPU so this does not take up GPU memory!
            # target.append(value.to("cpu").clone())
            # this should not require cloning since these variables are recreated for
            # every batch
            target.append(value.to("cpu"))

        if sel is None:
            count = batch_size
            target_dict["sample"].extend(list(range(batch_size)))
        else:
            batch_list = [i for i in range(batch_size) if sel[i]]
            count = len(batch_list)
            target_dict["sample"].extend(batch_list)

        target_dict["epoch"].extend(count * [ns.epoch])
        target_dict["batch"].extend(count * [ns.batch])

    def _sub_sample_monitor(
        self, name: str, ns: SimpleNamespace, sel: Optional[Sequence]
    ):
        """Observer called to keep track of fast dynamics."""
        target_dict = getattr(self.history, name)
        batch_size = None
        for var, target in target_dict.items():
            if var in ["epoch", "batch", "sample"]:
                continue

            var, *parts = var.split(":")
            value = getattr(ns.batch_results.profile, var)
            if len(parts) > 0:
                # this is part of a multi-layer variable
                # note: this is always true for the fast dynamics
                value = value[int(parts[-1])]

            # relax should ensure that we always have a batch index
            assert value.ndim > 1
            # send to CPU so this does not take up GPU memory!
            # value = value.detach().to("cpu").clone()
            # this should not require cloning since these variables are recreated for
            # every batch
            value = value.detach().to("cpu")
            # index0 = fast-dynamics iterations; index1 = sample index
            value = value.transpose(0, 1)
            if batch_size is None:
                batch_size = len(value)
            if sel is not None:
                value = value[sel]
            target.append(value)

        if sel is None:
            count = batch_size
            target_dict["sample"].extend(list(range(batch_size)))
        else:
            batch_list = [i for i in range(batch_size) if sel[i]]
            count = len(batch_list)
            target_dict["sample"].extend(batch_list)

        target_dict["epoch"].extend(count * [ns.epoch])
        target_dict["batch"].extend(count * [ns.batch])

    def _setup_history(self, name: str, vars: Sequence, type: str, size: int = 0):
        """Set up storage for a monitor.
        
        Normally whether the variable is multi-layered or not is inferred from the
        network itself. The argument `size` can be used to override this behavior. If
        `size` is provided, `self.net` is never accessed.
        """
        if type == "sample":
            # need a trick to find out the number of layers for the fast variables
            device = self.net.slow_parameters()[0].device
            test_in = torch.zeros(self.net.dims[0], device=device)
            test_out = torch.zeros(self.net.dims[-1], device=device)
            test_object = self.net.relax(test_in, test_out)
        else:
            test_object = self.net

        storage = {}
        for var in vars:
            layered = False
            crt_size = size
            if size == 0:
                if "." not in var:
                    value = getattr(test_object, var)
                else:
                    if not var.startswith("classifier"):
                        raise ValueError(f"can't parse variable name, {var}")
                    value = hierarchical_get(self, var)
                if isinstance(value, (list, tuple)):
                    crt_size = len(value)
                    layered = True
                else:
                    crt_size = 1
            elif size > 1:
                layered = True

            if layered:
                # this is a multi-layer variable
                for k in range(crt_size):
                    storage[var + ":" + str(k)] = None
            else:
                storage[var] = None

        storage["epoch"] = []
        if type in ["batch", "sample"]:
            storage["batch"] = []
            if type == "sample":
                storage["sample"] = []
        setattr(self.history, name, storage)

    def _reset_history(self):
        """Reset history storage (used before a `run()`)."""
        for name in self.history.__dict__:
            history = getattr(self.history, name)
            for var in history:
                history[var] = []

    def _coalesce_history(self):
        """Coalesce history into tensor form."""
        for name in self.history.__dict__:
            history = getattr(self.history, name)
            for var in history:
                if var not in ["epoch", "batch", "sample"]:
                    history[var] = torch.cat(history[var])

            history["epoch"] = torch.IntTensor(history["epoch"])
            if "batch" in history:
                history["batch"] = torch.IntTensor(history["batch"])
            if "sample" in history:
                history["sample"] = torch.IntTensor(history["sample"])


def evaluate(
    net,
    loader,
    accuracy_fct: Callable = one_hot_accuracy,
    classifier: Optional[Callable] = None,
    classifier_dim: int = -2,
) -> tuple:
    """Evaluate PCN or CPCN network on a test / validation set.
    
    :param net: network whose performance to evaluate; should have `pc_loss()` member
    :param loader: data loader for the test / validation set
    :param accuracy_fct: function used to calculate accuracy; called as
        `accuracy_fct(y_pred, y)`, where `y_pred` is prediction and `y` is ground truth
    :param classifier: classifier to use to obtain predictions from the network; this is
        applied to the output from the `classifier_dim`th layer; if it is not provided,
        the output from `net.forward()` is used
    :param classifier_dim: layer of `net` to use with the classifier
    :return: tuple of Numpy arrays (PC_loss, accuracy), where `PC_loss` is the
        predictive-coding loss for each batch, as returned by `net.pc_loss()` after a
        run of `net.relax()`; the accuracy is calculated based on the
        output from either the classifier (if given) or `net.forward()`, using the given
        `accuracy_fct()`
    """
    loss = []
    accuracy = []
    for x, y in loader:
        ns = net.relax(x, y)
        loss.append(net.pc_loss(ns.z).item())

        # figure out model predictions
        z_pred = net.forward(x)
        if classifier is not None:
            y_pred = classifier(z_pred[classifier_dim])
        else:
            y_pred = z_pred[-1]

        accuracy.append(accuracy_fct(y_pred, y))

    return loss, accuracy
