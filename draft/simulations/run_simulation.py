#! /usr/bin/env python
"""Optimize hyperparameters for a PCN or BioPCN."""

import os
import os.path as osp

import argparse
import time

import torch
import torchvision.transforms as T
import numpy as np

from cpcn import (
    PCNetwork,
    LinearBioPCN,
    BioPCN,
    load_torchvision,
    load_csv,
    load_lfw,
    Trainer,
    tqdmw,
)
from cpcn import (
    dot_accuracy,
    one_hot_accuracy,
    multi_lr,
    get_constraint_diagnostics,
    read_best_hyperparams,
)

import pickle

from typing import Tuple


def create_net(algo: str, dims: list, rho: list, best_params: dict):
    kwargs = {"z_lr": best_params["z_lr"], "z_it": 50, "rho": rho}
    if algo == "pcn":
        net = PCNetwork(
            dims,
            activation="none",
            variances=1,
            bias=False,
            constrained=True,
            **kwargs,
        )
    elif algo == "wb":  # Whittington&Bogacz
        net = PCNetwork(
            dims,
            activation="none",
            variances=1,
            bias=False,
            constrained=False,
            **kwargs,
        )
    elif algo == "biopcn":
        # set parameters to match a simple PCN network
        g_a = 0.5 * torch.ones(len(dims) - 2)
        g_a[-1] *= 2

        g_b = 0.5 * torch.ones(len(dims) - 2)
        g_b[0] *= 2

        net = LinearBioPCN(
            dims, g_a=g_a, g_b=g_b, c_m=0, l_s=g_b, bias_a=False, bias_b=False, **kwargs
        )
    elif algo == "biopcn-nl":
        # set parameters to match a simple PCN network
        g_a = 0.5 * torch.ones(len(dims) - 2)
        g_a[-1] *= 2

        g_b = 0.5 * torch.ones(len(dims) - 2)
        g_b[0] *= 2

        net = BioPCN(
            dims,
            activation="tanh",
            g_a=g_a,
            g_b=g_b,
            c_m=0,
            l_s=g_b,
            bias_a=False,
            bias_b=False,
            **kwargs,
        )
    else:
        raise ValueError(f"unknown algorithm, {algo}")

    return net


def run_simulation(
    net, n_batches: int, dataset: dict, best_params: dict
) -> Tuple[Trainer, list]:
    lr = best_params["lr"]
    lr_decay = best_params["lr_decay"]

    lr_factors = {}
    if hasattr(net, "Q"):
        lr_factors["Q"] = best_params["Q_lrf"]
    if "Wa_lrf" in best_params and hasattr(net, "W_a") and hasattr(net, "W_b"):
        lr_factors["W_a"] = best_params["Wa_lrf"]

    optimizer = multi_lr(
        torch.optim.SGD, net.parameter_groups(), lr_factors=lr_factors, lr=lr
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda batch: 1 / (1 + lr_decay * batch)
    )

    trainer = Trainer(dataset["train"], invalid_action="warn+stop")
    if accuracy_fct is not None:
        trainer.metrics["accuracy"] = accuracy_fct

    # figure out which parameters to monitor
    if hasattr(net, "W_a"):
        param_list = ["W_a", "W_b"]
    else:
        param_list = ["W"]

    if hasattr(net, "Q"):
        param_list.append("Q")

    checkpoints = []
    for batch in tqdmw(trainer(n_batches)):
        if batch.every(10):
            batch.evaluate(dataset["validation"]).run(net)

        if batch.every(5):
            checkpoints.append(net.clone().to("cpu"))

        if batch.every(20):
            for param in param_list:
                batch.weight.report(param, getattr(net, param))

        ns = batch.feed(net)

        batch.latent.report_batch("z", ns.z)

        optimizer.step()
        scheduler.step()

    return trainer, checkpoints


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(
        description="Run simulation using hyperparameters obtained from optimization"
    )

    parser.add_argument(
        "outdir",
        help="base output folder; saving results under "
        "${outdir}/${dataset}_${algo}_${arch}[_${rho}]/",
    )
    parser.add_argument(
        "hyperdir",
        help="base hyperparam optimization folder; "
        "looking for hyper_*.pkl files in ${hyperdir}/"
        "${dataset}_${algo}_${arch}[_${rho}]/",
    )
    parser.add_argument(
        "dataset", help="dataset: mnist, mmill, fashionmnist, cifar10, cifar100, lfw"
    )
    parser.add_argument("algo", help="algorithm: pcn, biopcn, biopcn-nl, wb")
    parser.add_argument("arch", help="architecture: small or many_n1[_n2[...]]")
    parser.add_argument("seed", type=int, help="starting random number seed")

    parser.add_argument("--n-batches", type=int, default=3000, help="number of batches")
    parser.add_argument(
        "--n-validation", type=int, default=2000, help="number of validation samples"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=0.80,
        help="scale factor for learning rate -- safety margin against divergence",
    )
    parser.add_argument(
        "--rho", type=float, nargs="+", default=1.0, help="constraint magnitudes"
    )
    parser.add_argument(
        "--arch-alias", default="", help="alias for architecture to use in folder names"
    )

    args = parser.parse_args()

    print(
        f"{args.algo} on {args.dataset}, seed {args.seed}, "
        f"n_batches: {args.n_batches}, n_validation: {args.n_validation}, "
        f"lr-scale: {args.lr_scale}."
    )

    # let SLURM run several simulations on one node instead of multi-threading sims
    torch.set_num_threads(1)

    # harder to find cluster nodes for GPU, so using CPU for all models
    device = torch.device("cpu")
    print(f"device: {device}")

    t0 = time.time()

    # load dataset
    tv_mapping = {
        "mnist": "MNIST",
        "fashionmnist": "FashionMNIST",
        "cifar10": "CIFAR10",
        "cifar100": "CIFAR100",
    }
    if args.dataset in tv_mapping.keys():
        tv_dataset = tv_mapping[args.dataset]
        dataset = load_torchvision(
            tv_dataset,
            n_validation=args.n_validation,
            batch_size=args.batch_size,
            device=device,
        )
        accuracy_fct = one_hot_accuracy
    elif args.dataset == "mmill":
        data_path = os.path.join("data", "mediamill")
        dataset = load_csv(
            os.path.join(data_path, "view1.csv"),
            os.path.join(data_path, "view2.csv"),
            n_validation=args.n_validation,
            batch_size=args.batch_size,
            device=device,
        )
        accuracy_fct = dot_accuracy
    elif args.dataset == "lfw":
        transform = T.Compose([T.Grayscale(), T.Resize(size=22), T.CenterCrop(15)])
        # for LFW validation set is set to test set by default
        dataset = load_lfw(transform=transform, n_test=500, batch_size=100)
        accuracy_fct = None
    else:
        raise ValueError(f"unknown dataset, {args.dataset}")

    # choose net params
    if args.arch == "small":
        hidden_dims = [5]
    elif args.arch.startswith("many_"):
        hidden_dims_list = args.arch.split("_")[1:]
        hidden_dims = [int(_) for _ in hidden_dims_list]
    else:
        raise ValueError(f"unknown architecture, {args.arch}")
    one_sample = next(iter(dataset["train"]))
    dims = [one_sample[0].shape[-1]] + hidden_dims + [one_sample[1].shape[-1]]
    print(f"network dims: {dims}")
    print(f"rho: {args.rho}")

    original_rho = args.rho
    if hasattr(args.rho, "__len__"):
        if len(args.rho) == 1:
            args.rho = args.rho * len(hidden_dims)
        else:
            assert len(args.rho) == len(hidden_dims)
    else:
        original_rho = [original_rho]
        args.rho = len(hidden_dims) * [args.rho]

    # create network using parameters from hyperparam optimization
    if len(args.arch_alias) == 0:
        args.arch_alias = args.arch
    if args.algo == "wb":
        rho_part = ""
    else:
        # unfortunately the naming convention for the hyperparam optimization wasn't
        # very well chosen...
        convert = lambda x: f"{x:.1f}" if np.abs(x - np.round(x)) < 1e-8 else f"{x:g}"
        rho_part = "_rho" + "_".join(convert(_) for _ in original_rho)
    subfolder = f"{args.dataset}_{args.algo}_{args.arch_alias}{rho_part}"
    folder = osp.join(args.hyperdir, subfolder)
    print(f"looking for hyperparameters in {folder}...")
    hyperparams = read_best_hyperparams(folder, args.lr_scale)
    print(hyperparams)

    torch.manual_seed(args.seed)
    net = create_net(args.algo, dims, args.rho, hyperparams).to(device)

    # run the simulation
    trainer, checkpoints = run_simulation(
        net, args.n_batches, dataset, best_params=hyperparams
    )

    # process z samples into cov and evals
    cons_diag = get_constraint_diagnostics(
        trainer.history.latent, rho=[1.0] + list(args.rho) + [1.0], every=20
    )

    # no need to keep track of input and output layer covariances
    cons_diag.pop("cov:0")
    cons_diag.pop(f"cov:{len(dims) - 1}")

    # replace storage of latent z's with constraint diagnostics
    del trainer.history.latent
    trainer.history.constraint = cons_diag

    # save to file
    outhistory = osp.join(args.outdir, subfolder, f"history_{args.seed}.pkl")
    print(f"Writing history to {outhistory}.")
    with open(outhistory, "wb") as f:
        pickle.dump(trainer.history, f)

    outcheckpoints = osp.join(args.outdir, subfolder, f"checkpoints_{args.seed}.pkl")
    print(f"Writing checkpoints to {outcheckpoints}.")
    with open(outcheckpoints, "wb") as f:
        pickle.dump(checkpoints, f)

    # done!
    t1 = time.time()

    print(f"total time: {t1 - t0:.1f} seconds.")
