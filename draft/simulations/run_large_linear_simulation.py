#! /usr/bin/env python
"""Run a simulation using a linear PCN or BioPCN network."""

import os
import os.path as osp
import argparse
import time
from functools import partial

from tqdm import tqdm

import torch
import numpy as np
from cpcn import PCNetwork, LinearBioPCN, Trainer
from cpcn import load_mnist, get_constraint_diagnostics

import pickle


def read_best_hyperparams(path: str, lr_scale: float) -> dict:
    """Find best hyperparameters from a folder of optimization results."""
    filenames = [
        osp.join(path, f)
        for f in os.listdir(path)
        if f.startswith("hyper_")
        and f.endswith(".pkl")
        and osp.isfile(osp.join(path, f))
    ]

    best_value = np.inf
    best_params = None
    for name in filenames:
        with open(name, "rb") as f:
            study = pickle.load(f)
            if study.best_value < best_value:
                best_value = study.best_value
                best_params = study.best_params

    best_params["lr"] *= lr_scale
    return best_params


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
    else:
        raise ValueError(f"unknwon algorithm, {algo}")

    return net


def run_simulation(net, n_batches: int, dataset: dict, best_params: dict) -> Trainer:
    # create trainer and set parameters
    trainer = Trainer(net, dataset["train"], dataset["validation"])

    optimizer_class = torch.optim.SGD
    trainer.set_optimizer(optimizer_class, lr=best_params["lr"]).add_nan_guard(count=10)
    if hasattr(net, "Q"):
        trainer.set_lr_factor("Q", best_params["Q_lrf"])
    if hasattr(net, "W_a"):
        trainer.set_lr_factor("W_a", best_params["Wa_lrf"])
    trainer.add_scheduler(
        partial(
            torch.optim.lr_scheduler.LambdaLR,
            lr_lambda=lambda batch: 1 / (1 + best_params["lr_rate"] * batch),
        ),
        every=1,
    )

    # trainer.set_classifier("linear")

    # set monitoring details
    trainer.peek_validation(every=10)
    trainer.peek_model(count=5)

    if hasattr(net, "W_a"):
        param_list = ["W_a", "W_b"]
    else:
        param_list = ["W"]

    if hasattr(net, "Q"):
        param_list.append("Q")

    trainer.peek("weight", param_list, count=20)
    trainer.peek_sample("latent", ["z"])

    # run
    trainer.run(n_batches=n_batches, progress=tqdm)
    return trainer


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(
        description="Run linear simulation using hyperparameters obtained "
        "from optimization"
    )

    parser.add_argument(
        "outdir",
        help="base output folder; saving results under"
        "${outdir}/${dataset}_${algo}_${arch}/",
    )
    parser.add_argument(
        "hyperdir",
        help="base hyperparam optimization folder; "
        "looking for hyper_*.pkl files in ${hyperdir}/${dataset}_${algo}_${arch}/",
    )
    parser.add_argument("algo", help="algorithm: wb, pcn, or biopcn")
    parser.add_argument("seed", type=int, help="starting random number seed")

    parser.add_argument("--n-batches", type=int, default=5000, help="number of batches")
    parser.add_argument(
        "--n-validation", type=int, default=1000, help="number of validation samples"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=0.80,
        help="scale factor for learning rate -- safety margin against divergence",
    )

    args = parser.parse_args()

    torch.set_num_threads(1)

    # using the GPU hinders more than helps for the smaller models
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(device)

    t0 = time.time()
    dataset = load_mnist(
        n_validation=args.n_validation, batch_size=args.batch_size, device=device
    )

    hidden_dims = [50, 5]
    one_sample = next(iter(dataset["train"]))
    dims = [one_sample[0].shape[-1]] + hidden_dims + [one_sample[1].shape[-1]]

    # create network using parameters from hyperparam optimization
    subfolder = f"mnist_{args.algo}_large_small"
    folder = osp.join(args.hyperdir, subfolder)
    hyperparams = read_best_hyperparams(folder, args.lr_scale)
    print(hyperparams)
    torch.manual_seed(args.seed)
    net = create_net(args.algo, dims, [1.0, 0.1], hyperparams).to(device)

    # run the simulation
    trainer = run_simulation(net, args.n_batches, dataset, best_params=hyperparams)

    # process z samples into cov and evals
    cons_diag = get_constraint_diagnostics(
        trainer.history.latent, rho=net.rho, every=20
    )

    # no need to keep track of input and output layer covariances
    cons_diag.pop("cov:0")
    cons_diag.pop(f"cov:{len(dims) - 1}")

    # replace storage of latent z's with constraint diagnostics
    del trainer.history.latent
    trainer.history.constraint = cons_diag

    # save to file
    outhistory = osp.join(args.outdir, subfolder, f"history_{args.seed}.pkl")
    with open(outhistory, "wb") as f:
        pickle.dump(trainer.history, f)

    outcheckpoints = osp.join(args.outdir, subfolder, f"checkpoints_{args.seed}.pkl")
    with open(outcheckpoints, "wb") as f:
        pickle.dump(trainer.checkpoint, f)

    # done!
    t1 = time.time()

    print(f"total time: {t1 - t0:.1f} seconds.")
