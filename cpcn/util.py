"""Define some utilities."""

from syslog import LOG_DAEMON
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
    n_correct = y[range(n), idx_pred].sum().item()

    return n_correct / n


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


def hierarchical_get(obj, attr: str):
    """Get an attribute in an object hierarchy.
    
    This follows several levels of indirection, as indicated by `"."` symbols in the
    attribute name.

    :param obj: object to access
    :param attr: attribute name
    :return: attribute value
    """
    while True:
        parts = attr.split(".", 1)
        obj = getattr(obj, parts[0])
        if len(parts) == 1:
            return obj
        else:
            attr = parts[1]
