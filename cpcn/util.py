"""Define some utilities."""

import os
import pickle

import torch
import torchvision
import torchvision.transforms as T

import numpy as np
import pandas as pd

from typing import Optional, Union, Sequence, Any

from cpcn.lfw_wrapper import LFWSanitized


def make_onehot(y) -> torch.Tensor:
    if not torch.is_tensor(y):
        y = torch.LongTensor(y)
    y_oh = torch.FloatTensor(y.shape[0], y.max().item() + 1)
    y_oh.zero_()
    y_oh.scatter_(1, y.reshape(-1, 1), 1)

    return y_oh


def one_hot_accuracy(ns, net) -> float:
    """Calculate accuracy of a batch of predictions, assuming one-hot true labels.
    
    :param ns: namespace with output from `TrainingBatch.feed()`; should have fields
        `ns.y` (the ground truth) and `ns.y_pred` (the prediction)
    :param net: network making the prediction; this is unused here
    :return: fraction of correctly identified samples in the batch
    """
    y = ns.y
    y_pred = ns.y_pred

    n = len(y)
    idx_pred = y_pred.argmax(dim=1)
    n_correct = y[range(n), idx_pred].sum().item()

    return n_correct / n


def dot_accuracy(ns, net) -> float:
    """Calculate accuracy in the sense of normalized dot product mapped from 0 to 1.
    
    For two vectors `x` and `y`, we consider the following measure of normalized dot
    product:
        0.5 * (1 + dot(x, y) / (norm(x) * norm(y))) .

    :param ns: namespace with output from `TrainingBatch.feed()`; should have fields
        `ns.y` (the ground truth) and `ns.y_pred` (the prediction)
    :param net: network making the prediction; this is unused here
    :return: an indication of the normalized dot product (uncentered correlation)
        averaged over samples; this will range from 0 to 1
    """
    y = ns.y
    y_pred = ns.y_pred

    norm_pred = torch.linalg.norm(y_pred, dim=1)
    norm = torch.linalg.norm(y, dim=1)

    dot = torch.sum(y_pred * y, dim=1)

    accuracy = 0.5 * (1 + dot / (norm_pred * norm))
    return torch.mean(accuracy).item()


def load_supervised(
    trainset,
    testset,
    n_train: Optional[int] = None,
    n_validation: int = 0,
    n_test: Optional[int] = None,
    center: bool = True,
    normalize: bool = True,
    one_hot: bool = True,
    batch_size: int = 128,
    batch_size_val: int = 1000,
    batch_size_test: int = 1000,
    return_loaders: bool = True,
) -> dict:
    """Load (parts of) a torchvision supervised dataset and split out a validation set.
    
    :param trainset: training set
    :param testset: test set
    :param n_train: number of training sample to keep; default: all, except what is used
        for validation
    :param n_validation: number of validation samples; default: no validation set
    :param n_test: number of test samples; default: all of the MNIST test set
    :param center: whether to center the samples such that the mean is 0
    :param normalize: whether to normalize the samples such that the stdev is 1
    :param one_hot: whether to convert the labels to a one-hot encoding
    :param cache_path: cache from where to load / where to store the datasets
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
    if torch.is_tensor(trainset.data):
        traindata = trainset.data.float()
    else:
        traindata = torch.FloatTensor(trainset.data)
    if torch.is_tensor(testset.data):
        testdata = testset.data.float()
    else:
        testdata = torch.FloatTensor(testset.data)

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


def load_torchvision(
    name: str, cache_path: str = "data/", dataset_kws: Optional[dict] = None, **kwargs
) -> dict:
    """Load a torchvision dataset.
    
    :param name: name of the dataset to load
    :param cache_path: cache from where to load / where to store the dataset
    :param dataset_kws: additional keyword arguments to pass to dataset constructor
    :param **kwargs: additional eyword arguments are passed to `load_supervised()`
    """
    dataset_kws = {} if dataset_kws is None else dataset_kws

    constructor = getattr(torchvision.datasets, name)
    trainset = constructor(cache_path, train=True, download=True, **dataset_kws)
    testset = constructor(cache_path, train=False, download=True, **dataset_kws)

    return load_supervised(trainset, testset, **kwargs)


def load_mnist(cache_path: str = "data/", **kwargs) -> dict:
    """Load (parts of) the MNIST dataset and split out a validation set.
    
    :param cache_path: cache from where to load / where to store the dataset
    :param **kwargs: additional eyword arguments are passed to `load_supervised()`
    """
    return load_torchvision("MNIST", cache_path, **kwargs)


def load_lfw(
    cache_path: str = "data/",
    n_train: Optional[int] = None,
    n_validation: int = 0,
    n_test: Optional[int] = None,
    normalize: bool = True,
    transform: Optional[Any] = None,
    batch_size: int = 128,
    batch_size_val: int = 1000,
    batch_size_test: int = 1000,
    test_as_validation: bool = True,
) -> dict:
    """Load (parts of) the LFWPairs dataset.
    
    :param cache_path: cache from where to load / where to store the datasets
    :param n_train: number of training samples to keep; default: all, except what is
        used for validation
    :param n_validation: number of validation samples; default: no validation set
    :param n_test: number of test samples; default: all of the test split
    :param normalize: whether to center and normalize the samples by stdev
    :param transform: transform to apply to the dataset; `ToTensor()`, flattening, and
        normalization (if `normalize` is true) are applied automatically
    :param batch_size: if `return_loaders` is true, this sets the batch size used
    :param batch_size_val: if `return_loaders` is true, this sets the batch size for the
        validation set
    :param batch_size_test: if `return_loaders` is true, this sets the batch size for
        the test set
    :param test_as_validation: if true, use the test set as validation set; in this case
        `n_validation` is assumed to be zero
    :return: a dictionary with keys `"train"`, `"validation"`, `"test"`, each of which
        maps to a data loader
    """
    transform_list = [T.ToTensor()]
    if transform is not None:
        transform_list = [transform] + transform_list
    transform = T.Compose(transform_list)

    trainset = LFWSanitized(
        cache_path,
        download=True,
        split="Train",
        normalize=normalize,
        flatten=True,
        transform=transform,
    )
    testset = LFWSanitized(
        cache_path,
        download=True,
        split="Test",
        normalize=normalize,
        flatten=True,
        transform=transform,
    )

    if test_as_validation:
        n_validation = 0

    if n_train is None:
        n_train = len(trainset) - n_validation
    if n_test is None:
        n_test = len(testset)

    final_trainset = torch.utils.data.Subset(trainset, range(n_train))
    final_valset = torch.utils.data.Subset(
        trainset, range(n_train, n_train + n_validation)
    )
    final_testset = torch.utils.data.Subset(testset, range(n_test))

    dataset = {}
    dataset["train"] = torch.utils.data.DataLoader(
        final_trainset, batch_size=batch_size, shuffle=True
    )
    dataset["test"] = torch.utils.data.DataLoader(
        final_testset, batch_size=batch_size_test, shuffle=False
    )

    if test_as_validation:
        dataset["validation"] = dataset["test"]
    else:
        dataset["validation"] = torch.utils.data.DataLoader(
            final_valset, batch_size=batch_size_val, shuffle=False
        )

    return dataset


def load_csv(
    file_in: str,
    file_out: str,
    n_train: Optional[int] = None,
    n_validation: int = 0,
    center: bool = True,
    normalize: bool = True,
    batch_size: int = 128,
    batch_size_val: int = 1000,
    return_loaders: bool = True,
    tidy: Union[bool, str] = "auto",
    read_csv_kws: Optional[dict] = None,
) -> dict:
    """Load a dataset contained in two CSV files and split out a validation set.

    :param file_in: file name for input samples
    :param file_out: file name for output samples    
    :param n_train: number of training sample to keep; default: all, except what is used
        for validation
    :param n_validation: number of validation samples; default: no validation set
    :param center: whether to center the samples such that the mean is 0
    :param normalize: whether to normalize the samples such that the stdev is 1
    :param batch_size: if `return_loaders` is true, this sets the batch size used
    :param batch_size_val: if `return_loaders` is true, this sets the batch size for the
        validation set
    :param batch_size_test: if `return_loaders` is true, this sets the batch size for the
        test set
    :param return_loaders: if true, data loaders are returned instead of the data sets;
        only the training loader uses shuffling
    :param tidy: a boolean, or the string `"auto"`; if true, the data is assumed to be
        in a "tidy" (long) format -- each row is a sample; if false, the data is assumed
        to be in the "Wide" format -- each column is a sample; if `"auto"`, the sample
        index is assumed to be the one that is shared across both input and output, and
        if both directions are equal, the longer one; (if the length is also the same,
        tidy format is assumed)
    :param read_csv_kws: optional arguments to be passed to `pd.read_csv`
    :return: a dictionary with keys `"train"`, `"validation"`, each of which maps to
        either a data loader (if `return_loaders` is true), or a tuple of two tensors,
        one for input, one for labels
    """
    # read the data and convert to tensor
    if read_csv_kws is None:
        read_csv_kws = {}
    read_csv_kws.setdefault("header", None)

    df_in = pd.read_csv(file_in, **read_csv_kws)
    df_out = pd.read_csv(file_out, **read_csv_kws)

    if tidy == "auto":
        if df_in.shape == df_out.shape:
            tidy = df_in.shape[0] >= df_in.shape[1]
        else:
            if df_in.shape[0] == df_out.shape[0]:
                tidy = True
            elif df_in.shape[1] == df_out.shape[1]:
                tidy = False
            else:
                raise ValueError(
                    "the input and output datasets have incompatible shapes"
                )

    data_in = torch.FloatTensor(df_in.values)
    data_out = torch.FloatTensor(df_out.values)

    if not tidy:
        data_in.t_()
        data_out.t_()

    # figure out normalization
    mu_in = torch.mean(data_in) if center else 0.0
    scale_in = 1.0 / torch.std(data_in) if normalize else 1.0

    mu_out = torch.mean(data_out) if center else 0.0
    scale_out = 1.0 / torch.std(data_out) if normalize else 1.0

    # handle defaults
    if n_train is None:
        if n_validation is None:
            n_train = len(data_in)
            n_validation = 0
        else:
            n_train = len(data_in) - n_validation

    # select the requested number of samples
    dataset = {
        "train": (data_in[:n_train], data_out[:n_train]),
        "validation": (data_in[-n_validation:], data_out[-n_validation:]),
    }

    # pre-process inputs and outputs, as requested
    for key, (input, output) in dataset.items():
        input = scale_in * (input - mu_in)
        output = scale_out * (output - mu_out)

        if return_loaders:
            batch_size_dict = {
                "train": batch_size,
                "validation": batch_size_val,
            }
            tensor_dataset = torch.utils.data.TensorDataset(input, output)
            dataset[key] = torch.utils.data.DataLoader(
                tensor_dataset,
                batch_size=batch_size_dict[key],
                shuffle=(key == "train"),
            )
        else:
            dataset[key] = (input, output)

    return dataset


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
    n_batch = np.max(latent["batch"]) + 1
    n_cov = n_batch // every

    res = {"batch": np.arange(n_cov) * every}
    for key, value in latent.items():
        parts = key.split(":")
        if len(parts) < 1 or parts[0] != var:
            continue

        layer = parts[1]

        size = value.shape[-1]
        crt_cov = np.zeros((n_cov, size, size))
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
        res["trace:" + layer] = np.array(
            [np.trace(crt_cov[i] - crt_rho * np.eye(size)) for i in range(n_cov)]
        )

        crt_all_evals = np.stack([np.linalg.eigh(_)[0] for _ in crt_cov])
        crt_max_eval = crt_all_evals.max(axis=-1)

        res["evals:" + layer] = crt_all_evals
        res["max_eval:" + layer] = crt_max_eval

    return res


def pretty_size(num: float, suffix: str = "B", multiplier: float = 1024.0) -> str:
    """Pretty-format a size.

    This does things like write `1024` as `1KB`. Largest level is terra (`"T"`).
    
    :param num: the number to format
    :param suffix: suffix to use
    :param multiplier: multiplier to use
    """
    # adapted from https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ["", "K", "M", "G"]:
        if abs(num) < multiplier:
            return f"{num:3.1f}{unit}{suffix}"
        num /= multiplier
    return f"{num:.1f}T{suffix}"


def read_best_hyperparams(
    path: str,
    lr_scale: float,
    prefix: str = "hyper_",
    extension: str = ".pkl",
    return_best_value: bool = False,
) -> dict:
    """Find best hyperparameters from a folder of optimization results.
    
    This finds all the files in `path` that start with `"hyper_"` and have extension
    `.pkl`, loads them (assuming they are pickeld Optuna studies), and identifies the
    set of parameters with the lowest `best_value`. (The prefix and extension can be
    configured; see below).

    :param path: path where to look for hyperparameter optimization files
    :param lr_scale: factor by which to multiply the learning rate; this helps as a
        safety margin to avoid divergences
    :param prefix: prefix to use for the file names instead of `"hyper_"`
    :param extension: extension to use instead of `.pkl`
    :param return_best_value: if true, a `"best_value"` key is included in the return
        dictionary indicating the best value reached during optimization
    :return: dictionary of parameters yielding best performance
    """
    filenames = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.startswith(prefix)
        and f.endswith(extension)
        and os.path.isfile(os.path.join(path, f))
    ]

    best_value = np.inf
    best_params = None
    for name in filenames:
        with open(name, "rb") as f:
            study = pickle.load(f)
            if study.best_value < best_value:
                best_value = study.best_value
                best_params = study.best_params

    if best_params is not None:
        if return_best_value:
            best_params["best_value"] = best_value

        if "lr" in best_params:
            best_params["lr"] *= lr_scale

    return best_params
