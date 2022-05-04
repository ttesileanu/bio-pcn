"""Define some utilities."""

import torch
import torchvision
import numpy as np

from types import SimpleNamespace
from typing import Optional, Callable, Iterable, Union, Sequence


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
    batch_size_val: int = 1000,
    batch_size_test: int = 1000,
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
    :param batch_size_val: if `return_loaders` is true, this sets the batch size for the
        validation set
    :param batch_size_test: if `return_loaders` is true, this sets the batch size for the
        test set
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
            batch_size_dict = {
                "train": batch_size,
                "validation": batch_size_val,
                "test": batch_size_test,
            }
            tensor_dataset = torch.utils.data.TensorDataset(input, labels)
            dataset[key] = torch.utils.data.DataLoader(
                tensor_dataset,
                batch_size=batch_size_dict[key],
                shuffle=(key == "train"),
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


def get_constraint_diagnostics(
    latent: dict, every: int = 10, rho: Union[Sequence, float] = 1.0, var: str = "z"
) -> dict:
    """Calculate diagnostics for the inequality constraint on the `z` covariance
    matrices.

    The inequality constraint is
        <z z.T> <= rho * identity ,
    where `<.>` is the average over samples and `rho` is a scale parameter.

    :param latent: dictionary containing the evolution of the latent variables; should
        contain members `"batch"` and `"z:?"`, where `?` is an integer ranging from 0 to
        the number of layers; the name of the variable can be changed from `"z"` by
        using the `var` argument (see below)
    :param every: how many batches between covariance estimates; all the recorded
        batches between estimates are used to calculate the covariance `<z z.T>`
    :param rho: scale parameter used in the constraint (see above)
    :param var: name of the latent variable to use instead of `"z"`
    :return: dictionary with keys `"batch"` (the middle of the range where the
        covariance matrix is estimated); `"cov:?"` (the covariance matrix for each
        layer); `"trace:?"` (the trace of the constraint `<z z.T> - rho * identity` for
        each layer); `"evals:?"` (eigenvalues of the covariance matrix, as obtained by
        `eigh`, for each layer); `"max_eval:?"` (maximum eigenvalue for each layer)
    """
    n_batch = torch.max(latent["batch"]).item()
    n_cov = n_batch // every

    res = {"batch": torch.arange(n_cov) * every}
    for key, value in latent.items():
        parts = key.split(":")
        if len(parts) < 1 or parts[0] != var:
            continue

        layer = parts[1]

        size = value.shape[-1]
        crt_cov = torch.zeros((n_cov, size, size))
        for i in range(n_cov):
            crt_start = i * every
            crt_end = (i + 1) * every
            crt_sel = (crt_start <= latent["batch"]) & (latent["batch"] < crt_end)
            crt_z = value[crt_sel]

            crt_cov[i] = crt_z.T @ crt_z / len(crt_z)

        cov_name = "cov:" + layer
        res[cov_name] = crt_cov

        if hasattr(rho, "__getitem__"):
            crt_rho = rho[int(layer)]
        else:
            crt_rho = rho
        res["trace:" + layer] = torch.FloatTensor(
            [torch.trace(crt_cov[i] - crt_rho * torch.eye(size)) for i in range(n_cov)]
        )

        crt_all_evals = torch.stack([torch.linalg.eigh(_)[0] for _ in crt_cov])
        crt_max_eval = crt_all_evals.max(dim=-1)[0]

        res["evals:" + layer] = crt_all_evals
        res["max_eval:" + layer] = crt_max_eval

    return res
