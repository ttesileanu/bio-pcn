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
    n = 0
    n_correct = 0
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


def train(
    net,
    n_epochs: int,
    train_loader: Iterable,
    validation_loader: Iterable,
    optimizer: Callable = torch.optim.Adam,
    optimizer_kwargs: Optional[dict] = None,
    accuracy_fct: Callable = one_hot_accuracy,
    classifier: Union[None, str, Callable] = None,
    classifier_optim: Callable = torch.optim.Adam,
    classifier_optim_kwargs: Optional[dict] = None,
    classifier_criterion: Optional[Callable] = None,
    classifier_dim: int = -2,
    per_batch: bool = False,
    progress: Optional[Callable] = None,
) -> SimpleNamespace:
    """Train a (constrained or not) predictive-coding network.
    
    This uses `net.forward_constrained()` to set the latent variables given input and
    output samples, then uses `net.calculate_weight_grad()` to calculate gradients for
    the optimizer.

    :param net: the network to train
    :param n_epochs: number of tarining epochs
    :param train_loader: data loader for training set
    :param validation_loader: data loader for validation set
    :param optimizer: callable to create the optimizer to use; default: Adam
    :param optimizer_kwargs: keyword arguments to pass to the `optimizer()`
    :param accuracy_fct: function used to calculate accuracy; called as
        `accuracy_fct(y, y_pred)`, where `y` is ground truth, `y_pred` is prediction
    :param classifier: this can be a trainable neural network used to predict the output
        samples from the `dim`th layer of `net` after a call to `net.forward()`; it can
        also be the string "linear", in which case a linear layer of appropriate input
        and ouptut dimensions is used; if it is not provided (or set to `None`), the
        output from `net.forward()` is used
    :param classifier_optim: callable to create the optimizer for the classifier
    :param classifier_optim_kwargs: keyword arguments to pass to `classifier_optim()`
    :param classifier_criterion: objective fucntion for training classifier; default:
        `MSELoss()`
    :param classifier_dim: which layer of `net` to pass into the classifier
    :param per_batch: if true, losses and accuracies are returned for each batch; if
        false, they are averaged over the batches to return one value for each epoch
    :param progress: progress indicator; should have `tqdm` interface
    :return: a namespace with the members `train` and `validation`, each containing
        two Numpy arrays, `pc_losses` and `accuracies`.
    """
    # handle some defaults
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if classifier_optim_kwargs is None:
        classifier_optim_kwargs = {}

    res = SimpleNamespace(train=SimpleNamespace(), validation=SimpleNamespace())

    res.train.pc_loss = []
    res.train.accuracy = []

    res.validation.pc_loss = []
    res.validation.accuracy = []

    optimizer = optimizer(net.slow_parameters(), **optimizer_kwargs)

    if classifier is not None:
        if isinstance(classifier, str):
            if classifier == "linear":
                if hasattr(net, "dims"):
                    dims = net.dims
                else:
                    dims = net.pyr_dims
                classifier = torch.nn.Sequential(
                    torch.nn.Linear(dims[classifier_dim], dims[-1])
                )
            else:
                raise ValueError("Invalid classifier type")

        if classifier_criterion is None:
            classifier_criterion = torch.nn.MSELoss()

        classifier_optim = classifier_optim(
            classifier.parameters(), **classifier_optim_kwargs
        )

    if progress is not None:
        epoch_range = progress(range(n_epochs))
    else:
        epoch_range = range(n_epochs)

    for epoch in epoch_range:
        # train
        batch_train_loss = []
        batch_train_accuracy = []
        for i, (x, y) in enumerate(train_loader):
            # train main net
            net.forward_constrained(x, y)
            pc_loss = net.pc_loss()

            net.calculate_weight_grad()
            optimizer.step()

            # train classifier, if we have one
            y_pred = net.forward(x)
            if classifier is not None:
                classifier_optim.zero_grad()

                y_pred = classifier(net.z[classifier_dim])
                classifier_loss = classifier_criterion(y_pred, y)
                classifier_loss.backward()
                classifier_optim.step()

            batch_train_loss.append(pc_loss.item())
            batch_train_accuracy.append(accuracy_fct(y_pred, y))

        # evaluate performance on validation set
        batch_val_loss, batch_val_accuracy = evaluate(
            net,
            validation_loader,
            accuracy_fct=accuracy_fct,
            classifier=classifier,
            classifier_dim=classifier_dim,
        )

        # store loss and accuracy values
        epoch_val_loss = np.mean(batch_val_loss)
        epoch_val_accuracy = np.mean(batch_val_accuracy)
        if per_batch:
            res.train.pc_loss.extend(batch_train_loss)
            res.train.accuracy.extend(batch_train_accuracy)
            res.validation.pc_loss.extend(batch_val_loss)
            res.validation.accuracy.extend(batch_val_accuracy)
        else:
            res.train.pc_loss.append(np.mean(batch_train_loss))
            res.train.accuracy.append(np.mean(batch_train_accuracy))
            res.validation.pc_loss.append(epoch_val_loss)
            res.validation.accuracy.append(epoch_val_accuracy)

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


def load_mnist(
    n_train: Optional[int] = None,
    n_validation: int = 0,
    n_test: Optional[int] = None,
    center: bool = True,
    normalize: bool = True,
    one_hot: bool = True,
    cache_path: str = "data/",
    device: Optional[torch.device] = None,
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
    :return: a dictionary with keys `"train"`, `"validation"`, `"test"`, each of which
        maps to a tuple of two tensors, one for input, one for labels
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

        dataset[key] = (input, labels)

    return dataset
