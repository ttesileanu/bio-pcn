"""Define utilities for training predictive-coding models. """

from types import SimpleNamespace
from typing import Optional, Callable, Iterable, Union, Sequence

import torch
import numpy as np

from .util import one_hot_accuracy


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
        also be the string "linear", in which case a linear layer of appropriate input
        and ouptut dimensions is used; if it is not provided (or set to `None`), the
        output from `net.forward()` is used
    :param classifier_optim_class: callable to create the optimizer for the classifier
    :param classifier_optim_kwargs: keyword arguments to pass to `classifier_optim()`
    :param classifier_criterion: objective fucntion for training classifier; default:
        `MSELoss()`
    :param classifier_dim: which layer of `net` to pass into the classifier
    :param epoch_observers: list of tuples `(observer, condition)`; see
        `add_epoch_observer()`
    :param batch_observers: list of tuples `(observer, condition, profile)`; see
        `add_batch_observer()`
    :param history: namespace of history data for the last call to `run()`; see the
        `peek...` functions
    :param schedulers: list of learning-rate scheduler constructors
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

        self.accuracy_fct = one_hot_accuracy

        self.classifier = None
        self.classifier_optim_class = torch.optim.Adam
        self.classifier_optim_kwargs = {}
        self.classifier_criterion = None
        self.classifier_dim = -2

        self.epoch_observers = []
        self.batch_observers = []
        self.schedulers = []

        self.history = SimpleNamespace()

    def run(
        self, n_epochs: int, progress: Optional[Callable] = None
    ) -> SimpleNamespace:
        """Run the training.
        
        :param n_epochs: number of training epochs
        :param progress: progress indicator; should have `tqdm` interface
        """
        res = SimpleNamespace(train=SimpleNamespace(), validation=SimpleNamespace())

        res.train.pc_loss = []
        res.train.accuracy = []

        res.validation.pc_loss = []
        res.validation.accuracy = []

        optimizer = self.optimizer_class(
            self.net.slow_parameters(), **self.optimizer_kwargs
        )

        # set up any learning-rate schedulers
        schedulers = []
        for constructor in self.schedulers:
            schedulers.append(constructor(optimizer))

        if self.classifier is not None:
            if self.classifier_criterion is None:
                self.classifier_criterion = torch.nn.MSELoss()

            classifier_optim = self.classifier_optim_class(
                self.classifier.parameters(), **self.classifier_optim_kwargs
            )

        if progress is not None:
            epoch_range = progress(range(n_epochs))
        else:
            epoch_range = range(n_epochs)

        self._reset_history()
        for epoch in epoch_range:
            # train
            batch_train_loss = []
            batch_train_accuracy = []
            for i, (x, y) in enumerate(self.train_loader):
                # check for any observers -- need to know whether to request profiles
                observers = []
                need_profile = False
                for observer, condition, profile in self.batch_observers:
                    if condition(epoch, i):
                        observers.append(observer)
                        if profile:
                            need_profile = True

                # train main net
                batch_results = self.net.forward_constrained(
                    x, y, pc_loss_profile=need_profile, latent_profile=need_profile
                )
                pc_loss = self.net.pc_loss()

                self.net.calculate_weight_grad()
                optimizer.step()

                # train classifier, if we have one; but don't overwrite latents!
                z_fwd = self.net.forward(x, inplace=False)
                if self.classifier is not None:
                    y_pred = self.classifier(z_fwd[self.classifier_dim])

                    classifier_optim.zero_grad()
                    classifier_loss = self.classifier_criterion(y_pred, y)
                    classifier_loss.backward()
                    classifier_optim.step()
                else:
                    y_pred = z_fwd[-1]

                accuracy = self.accuracy_fct(y_pred, y)
                batch_train_loss.append(pc_loss.item())
                batch_train_accuracy.append(accuracy)

                if len(observers) > 0:
                    batch_ns = SimpleNamespace(
                        epoch=epoch,
                        batch=i,
                        net=self.net,
                        train_loss=pc_loss,
                        train_accuracy=accuracy,
                    )
                    if need_profile:
                        batch_ns.pc_loss_profile = batch_results.pc_loss
                        batch_ns.latent_profile = batch_results.latent
                    for observer in observers:
                        observer(batch_ns)

            # evaluate performance on validation set
            batch_val_loss, batch_val_accuracy = evaluate(
                self.net,
                self.validation_loader,
                accuracy_fct=self.accuracy_fct,
                classifier=self.classifier,
                classifier_dim=self.classifier_dim,
            )

            # store loss and accuracy values
            epoch_train_loss = np.mean(batch_train_loss)
            epoch_train_accuracy = np.mean(batch_train_accuracy)
            epoch_val_loss = np.mean(batch_val_loss)
            epoch_val_accuracy = np.mean(batch_val_accuracy)
            res.train.pc_loss.append(epoch_train_loss)
            res.train.accuracy.append(epoch_train_accuracy)
            res.validation.pc_loss.append(epoch_val_loss)
            res.validation.accuracy.append(epoch_val_accuracy)

            # call any epoch observers
            epoch_ns = SimpleNamespace(
                epoch=epoch,
                net=self.net,
                train_loss=epoch_train_loss,
                train_accuracy=epoch_train_accuracy,
                val_loss=epoch_val_loss,
                val_accuracy=epoch_val_accuracy,
                batch_profile=SimpleNamespace(
                    train_loss=batch_train_loss,
                    train_accuracy=batch_train_accuracy,
                    val_loss=batch_val_loss,
                    val_accuracy=batch_val_accuracy,
                ),
            )
            for observer, condition in self.epoch_observers:
                if condition(epoch):
                    observer(epoch_ns)

            # run learning-rate schedulers
            for scheduler in schedulers:
                scheduler.step()

            # update progress bar, if any
            if progress is not None:
                epoch_range.set_postfix(
                    {
                        "val_loss": f"{epoch_val_loss:.2g}",
                        "val_acc": f"{epoch_val_accuracy:.2f}",
                    }
                )

        self._coalesce_history()

        # convert to Numpy
        for ns in [res.train, res.validation]:
            ns.pc_loss = np.asarray(ns.pc_loss)
            ns.accuracy = np.asarray(ns.accuracy)

        return res

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
        self, classifier: Union[Callable, None], classifier_dim: Optional[int] = None
    ) -> "Trainer":
        """Set a classifier to use for predictions.

        This can be a trainable neural network used to predict the output samples from
        the `classifier_dim`th layer of `net` after a call to `net.forward()`.
        
        It can also be the string "linear", in which case a linear layer of appropriate
        input and ouptut dimensions is used.
        
        You can also set it to `None`, in which case the output from `net.forward()` is
        used.
        
        :param classifier: the classifier to use
        :param classifier_dim: which layer of `net` to pass into the classifier
        """
        self.classifier = classifier
        if isinstance(self.classifier, str):
            if self.classifier == "linear":
                if hasattr(self.net, "dims"):
                    dims = self.net.dims
                else:
                    dims = self.net.pyr_dims
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(dims[self.classifier_dim], dims[-1])
                )
            else:
                raise ValueError("Invalid classifier type")

        if classifier_dim is not None:
            self.classifier_dim = classifier_dim

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

    def add_epoch_observer(
        self, observer: Callable, condition: Optional[Callable] = None
    ) -> "Trainer":
        """Set an epoch-dependent observer.
        
        An observer is a function called after every training epoch, or after every
        epoch satisfying a certain condition. The observer gets called with a namespace
        argument,
            observer(ns: SimpleNamespace)
        where the `SimpleNamespace` contains
            epoch:          the index of the epoch that just ended
            net:            the network that is being optimized
            train_loss:     (predictive-coding) loss on training set
            val_loss:       (predictive-coding) loss on validation set
            train_accuracy: accuracy on training set
            val_accuracy:   accuracy on validation set
        
        :param observer: the observer callback
        :param condition: condition to be fulfilled for the observer to be called; this
            has signature (epoch: int) -> bool; must return true to call observer
        """
        if condition is None:
            condition = lambda _: True
        self.epoch_observers.append((observer, condition))
        return self

    def add_batch_observer(
        self,
        observer: Callable,
        condition: Optional[Callable] = None,
        profile: bool = False,
    ) -> "Trainer":
        """Set a batch-dependent observer.
        
        An observer is a function called after every batch of every training epoch -- or
        after those satisfying a certain condition. The observer gets called with a
        namespace argument,
            observer(ns: SimpleNamespace)
        where the `SimpleNamespace` contains
            epoch:          the index of the epoch that just ended
            batch:          the index of the batch that was just processed
            net:            the network that is being optimized
            train_loss:     (predictive-coding) loss on training set
            train_accuracy: accuracy on training set

            (if `profile == True`)
            pc_loss_profile:the loss profile from `net.forward_constrained()`
            latent_profile: the latent profile from `net.forward_constrained()`
        
        Note that setting `profile` to true could potentially slow down learning.
        
        :param observer: the observer callback
        :param condition: condition to be fulfilled for the observer to be called; this
            has signature (epoch: int, batch: int) -> bool; must return true to call
            observer
        :param profile: whether to request the loss and latent profile from
            `net.forward_constrained()`
        """
        if condition is None:
            condition = lambda epoch, batch: True
        self.batch_observers.append((observer, condition, profile))
        return self

    def set_accuracy_fct(self, accuracy_fct: Callable) -> "Trainer":
        """Set the function used to calculate accuracy scores.
        
        This is called as `accuracy_fct(y, y_pred)`, where `y` is ground truth, `y_pred`
        is prediction.
        """
        self.accuracy_fct = accuracy_fct
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

    def peek_epoch(
        self, name: str, vars: Sequence, condition: Optional[Callable] = None
    ) -> "Trainer":
        """Add per-epoch monitoring.
        
        This is used to store values of parameters after each epoch (or after those
        epochs obeying a condition). The values will be stored in `self.history` during
        calls to `self.run()`.

        :param name: the name to be used in `self.history` for the stored values
        :param vars: variables to track; these should be names of attributes of the
            model under training
        :param condition: condition to be fulfilled for the observer to be called; this
            has signature (epoch: int) -> bool
        """
        if hasattr(self.history, name):
            raise ValueError("monitor name already in use")
        self._setup_history(name, vars, "epoch")
        return self.add_epoch_observer(
            lambda ns, name=name: self._monitor(name, ns), condition
        )

    def peek_batch(
        self, name: str, vars: Sequence, condition: Optional[Callable] = None
    ) -> "Trainer":
        """Add per-batch monitoring.
        
        This is used to store values of parameters after each batch (or after those
        batches obeying a condition). The values will be stored in `self.history` during
        calls to `self.run()`.

        :param name: the name to be used in `self.history` for the stored values
        :param vars: variables to track; these should be names of attributes of the
            model under training
        :param condition: condition to be fulfilled for the observer to be called; this
            has signature (epoch: int, batch: int) -> bool
        """
        if hasattr(self.history, name):
            raise ValueError("monitor name already in use")
        self._setup_history(name, vars, "batch")
        return self.add_batch_observer(
            lambda ns, name=name: self._monitor(name, ns), condition
        )

    def peek_sample(
        self, name: str, vars: Sequence, condition: Optional[Callable] = None
    ) -> "Trainer":
        """Add per-sample monitoring.
        
        This is used to store values of parameters that change with each sample. The
        values will be stored in `self.history` during calls to `self.run()`.

        :param name: the name to be used in `self.history` for the stored values
        :param vars: variables to track; these should be names of attributes of the
            model under training
        :param condition: condition to be fulfilled for the observer to be called; this
            has signature (epoch: int, batch: int) -> bool; either all or none of the
            samples in a batch are stored
        """
        if hasattr(self.history, name):
            raise ValueError("monitor name already in use")
        self._setup_history(name, vars, "sample")
        return self.add_batch_observer(
            lambda ns, name=name: self._sample_monitor(name, ns), condition
        )

    def peek_fast_dynamics(
        self, name: str, vars: Sequence, condition: Optional[Callable] = None
    ) -> "Trainer":
        """Add monitoring of fast (`forward_constrained`) dynamics.
        
        This is used to store the evolution of the latent variables during the fast
        dynamics generated by `net.forward_constrained()`. The results are stored in
        `self.history` during calls to `self.run()`.

        :param name: the name to be used in `self.history` for the stored values
        :param vars: variables to track; these should be a subset of the variables
            returned by `net.forward_constrained()` when `latent_profile` is true;
        :param condition: condition to be fulfilled for the observer to be called; this
            has signature (epoch: int, batch: int) -> bool; either all or none of the
            samples in a batch are stored
        """
        if hasattr(self.history, name):
            raise ValueError("monitor name already in use")
        self._setup_history(name, vars, "sample")
        return self.add_batch_observer(
            lambda ns, name=name: self._sub_sample_monitor(name, ns),
            condition,
            profile=True,
        )

    def add_scheduler(self, scheduler: Callable) -> "Trainer":
        """Add a constructor for a learning-rate scheduler.
        
        Calling `scheduler()` with the optimizer as an argument should create a
        learning-rate scheduler. (This is needed since the optimizer is only generated
        during a call to `run()`.)

        You can use an inline function to pass additional arguments to standard
        schedulers, e.g.:
            lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        """
        self.schedulers.append(scheduler)
        return self

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
            value = getattr(ns.net, var)
            if len(parts) > 0:
                # this is part of a multi-layer variable
                k = int(parts[-1])
                target.append(value[k].detach().clone().unsqueeze(0))
            else:
                target.append(value.detach().clone().unsqueeze(0))

    def _sample_monitor(self, name: str, ns: SimpleNamespace):
        """Observer called to update per-sample monitors."""
        target_dict = getattr(self.history, name)
        for var, target in target_dict.items():
            if var in ["epoch", "batch", "sample"]:
                continue

            var, *parts = var.split(":")
            value = getattr(ns.net, var)
            if len(parts) > 0:
                # this is part of a multi-layer variable
                value = value[int(parts[-1])]

            # handle batch size of 1
            if value.ndim == 1:
                value = value.unsqueeze(0)
            value = value.detach().clone()
            target.append(value)

        batch_size = len(value)
        target_dict["epoch"].extend(batch_size * [ns.epoch])
        target_dict["batch"].extend(batch_size * [ns.batch])
        target_dict["sample"].extend(list(range(batch_size)))

    def _sub_sample_monitor(self, name: str, ns: SimpleNamespace):
        """Observer called to keep track of fast dynamics."""
        target_dict = getattr(self.history, name)
        for var, target in target_dict.items():
            if var in ["epoch", "batch", "sample"]:
                continue

            var, *parts = var.split(":")
            value = getattr(ns.latent_profile, var)
            if len(parts) > 0:
                # this is part of a multi-layer variable
                # note: this is always true for the fast dynamics
                value = value[int(parts[-1])]

            # forward_constrained should ensure that we always have a batch index
            assert value.ndim > 1
            # index0 = fast-dynamics iterations; index1 = sample index
            value = value.detach().clone().transpose_(0, 1)
            target.append(value)

        batch_size = len(value)
        target_dict["epoch"].extend(batch_size * [ns.epoch])
        target_dict["batch"].extend(batch_size * [ns.batch])
        target_dict["sample"].extend(list(range(batch_size)))

    def _setup_history(self, name: str, vars: Sequence, type: str):
        """Set up storage for a monitor."""
        storage = {}
        for var in vars:
            value = getattr(self.net, var)
            if isinstance(value, (list, tuple)):
                # this is a multi-layer variable
                for k in range(len(value)):
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
        run of `net.forward_constrained()`; the accuracy is calculated based on the
        output from either the classifier (if given) or `net.forward()`, using the given
        `accuracy_fct()`
    """
    loss = []
    accuracy = []
    for x, y in loader:
        net.forward_constrained(x, y)
        loss.append(net.pc_loss().item())

        # figure out model predictions
        z_pred = net.forward(x, inplace=False)
        if classifier is not None:
            y_pred = classifier(z_pred[classifier_dim])
        else:
            y_pred = z_pred[-1]

        accuracy.append(accuracy_fct(y_pred, y))

    return loss, accuracy
