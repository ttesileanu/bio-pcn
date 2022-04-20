"""Define some utilities."""

import torch
import torchvision
import numpy as np

from types import SimpleNamespace
from typing import Optional, Callable, Iterable, Union


def make_onehot(y) -> torch.Tensor:
    y_oh = torch.FloatTensor(y.shape[0], y.max().item() + 1)
    y_oh.zero_()
    y_oh.scatter_(1, y.reshape(-1, 1), 1)

    return y_oh


def one_hot_accuracy(y_pred, y) -> float:
    """Calculate accuracy of a batch of predictions, assuming one-hot true labels.
    
    :param y_pred: prediction
    :param y: ground truth; this is assumed to be one-hot; first index = batch
    :return: fraction of correctly identified samples in the batch
    """
    n = len(y)
    idx_pred = y_pred.argmax(dim=1)
    n_correct = y[range(n), idx_pred].sum()

    return n_correct / n


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
        y_pred = net.forward(x)
        if classifier is not None:
            y_pred = classifier(net.z[classifier_dim])

        accuracy.append(accuracy_fct(y_pred, y))

    return loss, accuracy


class Trainer:
    """A class for training (constrained or not) predictive-coding networks.
    
    Attributes
    :param net: the network to train
    :param train_loader: data loader for training set
    :param validation_loader: data loader for validation set
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

                # train classifier, if we have one
                y_pred = self.net.forward(x)
                if self.classifier is not None:
                    classifier_optim.zero_grad()

                    y_pred = self.classifier(self.net.z[self.classifier_dim])
                    classifier_loss = self.classifier_criterion(y_pred, y)
                    classifier_loss.backward()
                    classifier_optim.step()

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

            # update progress bar, if any
            if progress is not None:
                epoch_range.set_postfix(
                    {
                        "val_loss": f"{epoch_val_loss:.2g}",
                        "val_acc": f"{epoch_val_accuracy:.2f}",
                    }
                )

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


def load_mnist(
    n_train: Optional[int] = None,
    n_validation: int = 0,
    n_test: Optional[int] = None,
    center: bool = True,
    normalize: bool = True,
    one_hot: bool = True,
    cache_path: str = "data/",
    device: Optional[torch.device] = None,
    batch_size: int = 128,
    return_loaders: bool = True,
) -> dict:
    """Load (parts of) the MNIST dataset and split out a validation set.
    
    :param n_train: number of training sample to keep; default: all, except what is used
        for validation
    :param n_validation: number of validation samples; default: no validation set
    :param n_test: number of test samples; default: all of the MNIST test set
    :param center: whether to center the samples such that the mean is 0
    :param normalize: whether to normalize the samples such that the stdev is 1
    :param one_hot: whether to convert the labels to a one-hot encoding
    :param cache_path: cache from where to load / where to store the datasets
    :param device: device to send the data to
    :param batch_size: if `return_loaders` is true, this sets the batch size used
    :param return_loaders: if true, data loaders are returned instead of the data sets;
        only the training loader uses shuffling
    :return: a dictionary with keys `"train"`, `"validation"`, `"test"`, each of which
        maps to either a data loader (if `return_loaders` is true), or a tuple of two
        tensors, one for input, one for labels
    """
    trainset = torchvision.datasets.MNIST(cache_path, train=True, download=True)
    testset = torchvision.datasets.MNIST(cache_path, train=False, download=True)

    traindata = trainset.data.float()
    testdata = testset.data.float()

    # figure out normalization
    mu = torch.mean(traindata) if center else 0.0
    scale = 1.0 / torch.std(traindata) if normalize else 1.0

    # handle defaults
    if n_train is None:
        if n_validation is None:
            n_train = len(traindata)
            n_validation = 0
        else:
            n_train = len(traindata) - n_validation

    if n_test is None:
        n_test = len(testdata)

    # select the requested number of samples
    dataset = {
        "train": (traindata[:n_train], trainset.targets[:n_train]),
        "validation": (traindata[-n_validation:], trainset.targets[-n_validation:]),
        "test": (testdata[:n_test], testset.targets[:n_test]),
    }

    # pre-process inputs and labels, as requested
    for key, (input, labels) in dataset.items():
        input = (scale * (input - mu)).reshape(len(input), -1)
        if one_hot:
            labels = make_onehot(labels)
        if device is not None:
            input = input.to(device)
            labels = labels.to(device)

        if return_loaders:
            tensor_dataset = torch.utils.data.TensorDataset(input, labels)
            dataset[key] = torch.utils.data.DataLoader(
                tensor_dataset, batch_size=batch_size, shuffle=(key == "train")
            )
        else:
            dataset[key] = (input, labels)

    return dataset
