#! /usr/bin/env python
"""Optimize hyperparameters for a PCN or BioPCN."""

import os
import argparse
import time
from functools import partial

import torch
from cpcn import PCNetwork, LinearBioPCN, BioPCN, load_mnist, load_csv, Trainer
from cpcn import dot_accuracy, one_hot_accuracy, multi_lr

import optuna
from optuna.trial import TrialState

import pickle

from typing import Callable, Optional


def create_net(algo: str, dims: list, rho: list, trial, z_lr_range: tuple):
    z_lr = trial.suggest_float("z_lr", z_lr_range[0], z_lr_range[1], log=True)

    kwargs = {"z_lr": z_lr, "z_it": 50, "rho": rho}
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


def objective(
    trial: optuna.trial.Trial,
    n_batches: int,
    dataset: dict,
    device: torch.device,
    seed: int,
    n_rep: int,
    algo: str,
    dims: list,
    rho: list,
    accuracy_fct: Callable,
    constraint: bool,
    z_lr_range: tuple,
    lr_range: tuple,
    lr_decay_range: tuple,
    Q_lrf_range: tuple,
    Wa_lrf_range: Optional[tuple],
) -> float:
    torch.manual_seed(seed)
    seed0 = torch.randint(0, 2_000_000_000, (1,)).item()

    scores = torch.zeros(n_rep)
    for i in range(n_rep):
        torch.manual_seed(seed0 + i)
        net = create_net(algo, dims, rho, trial, z_lr_range).to(device)

        lr = trial.suggest_float("lr", *lr_range, log=True)
        lr_decay = trial.suggest_float("lr_decay", *lr_decay_range, log=True)

        lr_factors = {}
        if constraint:
            lr_factors["Q"] = trial.suggest_float("Q_lrf", *Q_lrf_range, log=True)
        if Wa_lrf_range is not None and hasattr(net, "W_a") and hasattr(net, "W_b"):
            lr_factors["W_a"] = trial.suggest_float("Wa_lrf", *Wa_lrf_range, log=True)

        optimizer = multi_lr(
            torch.optim.SGD, net.parameter_groups(), lr_factors=lr_factors, lr=lr
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda batch: 1 / (1 + lr_decay * batch)
        )

        trainer = Trainer(dataset["train"], invalid_action="warn+stop")
        trainer.metrics["accuracy"] = accuracy_fct
        for batch in trainer(n_batches):
            if batch.every(10):
                batch.evaluate(dataset["validation"]).run(net)

            batch.feed(net)
            optimizer.step()
            scheduler.step()

        scores[i] = trainer.history.validation["pc_loss"][-1]

    score = torch.quantile(scores, 0.90).item()
    return score


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization")

    parser.add_argument("out", help="output file")
    parser.add_argument("dataset", help="dataset: mnist or mmill")
    parser.add_argument("algo", help="algorithm: pcn, biopcn, biopcn-nl, wb")
    parser.add_argument("arch", help="architecture: small or many_n1[_n2[...]]")
    parser.add_argument("trials", type=int, help="number of trials")
    parser.add_argument("seed", type=int, help="starting random number seed")

    parser.add_argument(
        "--rho", type=float, nargs="+", default=1.0, help="constraint magnitudes"
    )
    parser.add_argument("--n-batches", type=int, default=500, help="number of batches")
    parser.add_argument("--n-rep", type=int, default=10, help="number of repetitions")

    parser.add_argument(
        "--z-lr", type=float, nargs=2, default=[0.01, 0.2], help="range for z_lr"
    )
    parser.add_argument(
        "--lr", type=float, nargs=2, default=[5e-4, 0.05], help="range for lr"
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        nargs=2,
        default=[1e-5, 0.2],
        help="range for lr_decay",
    )
    parser.add_argument(
        "--Q-lrf", type=float, nargs=2, default=[0.1, 20.0], help="range for Q_lrf"
    )
    parser.add_argument(
        "--Wa-lrf", type=float, nargs=2, default=None, help="range for Wa_lrf"
    )

    args = parser.parse_args()

    print(f"{args.algo} on {args.dataset} for {args.trials} trials; seed {args.seed}")
    print(f"n_batches: {args.n_batches}, n_rep: {args.n_rep}")

    torch.set_num_threads(1)

    # harder to find cluster nodes for GPU, so using CPU for all models
    device = torch.device("cpu")
    print(f"device: {device}")

    t0 = time.time()
    if args.dataset == "mnist":
        dataset = load_mnist(n_validation=500, batch_size=100, device=device)
        accuracy_fct = one_hot_accuracy
    elif args.dataset == "mmill":
        data_path = os.path.join("data", "mediamill")
        dataset = load_csv(
            os.path.join(data_path, "view1.csv"),
            os.path.join(data_path, "view2.csv"),
            n_validation=500,
            batch_size=100,
            device=device,
        )
        accuracy_fct = dot_accuracy
    else:
        raise ValueError(f"unknown dataset, {args.dataset}")

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

    if hasattr(args.rho, "__len__"):
        assert len(args.rho) == len(hidden_dims)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            n_batches=args.n_batches,
            dataset=dataset,
            algo=args.algo,
            dims=dims,
            rho=args.rho,
            seed=args.seed,
            n_rep=args.n_rep,
            accuracy_fct=accuracy_fct,
            constraint=args.algo != "wb",
            z_lr_range=args.z_lr,
            lr_range=args.lr,
            lr_decay_range=args.lr_decay,
            Q_lrf_range=args.Q_lrf,
            Wa_lrf_range=args.Wa_lrf,
            device=device,
        ),
        n_trials=args.trials,
        timeout=86400,
        show_progress_bar=True,
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    t1 = time.time()

    print(
        f"{len(study.trials)} trials in {t1 - t0:.1f} seconds: "
        f"{len(complete_trials)} complete, {len(pruned_trials)} pruned."
    )

    trial = study.best_trial
    print(f"best pc_loss: {trial.value}, for params:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(os.path.join(args.out), "wb") as f:
        pickle.dump(study, f)
