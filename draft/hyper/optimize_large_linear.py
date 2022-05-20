#! /usr/bin/env python
"""Optimize hyperparameters for a PCN or BioPCN with a two hidden layer, one large."""

import os
import argparse
import time
from functools import partial

from tqdm import tqdm

import torch
from cpcn import PCNetwork, LinearBioPCN, BioPCN, load_mnist, load_csv, Trainer
from cpcn import dot_accuracy, one_hot_accuracy

import optuna
from optuna.trial import TrialState

import pickle

from typing import Callable


def create_net(algo: str, dims: list, rho: list, trial):
    z_lr = trial.suggest_float("z_lr", 0.01, 0.2, log=True)

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
) -> float:
    torch.manual_seed(seed)
    seed0 = torch.randint(0, 2_000_000_000, (1,)).item()

    scores = torch.zeros(n_rep)
    for i in range(n_rep):
        torch.manual_seed(seed0 + i)
        net = create_net(algo, dims, rho, trial).to(device)

        optimizer_class = torch.optim.SGD
        lr = trial.suggest_float("lr", 1e-3, 0.05, log=True)

        trainer = Trainer(net, dataset["train"], dataset["validation"])
        trainer.set_accuracy_fct(accuracy_fct)
        trainer.set_optimizer(optimizer_class, lr=lr).add_nan_guard(count=10)

        lr_power = 1.0
        lr_rate = trial.suggest_float("lr_rate", 1e-5, 0.01, log=True)
        trainer.add_scheduler(
            partial(
                torch.optim.lr_scheduler.LambdaLR,
                lr_lambda=lambda batch: 1 / (1 + lr_rate * batch ** lr_power),
            ),
            every=1,
        )

        if constraint:
            Q_lrf = trial.suggest_float("Q_lrf", 0.1, 20, log=True)
            trainer.set_lr_factor("Q", Q_lrf)

        if hasattr(net, "W_a") and hasattr(net, "W_b"):
            Wa_lrf = trial.suggest_float("Wa_lrf", 0.3, 4, log=True)
            trainer.set_lr_factor("W_a", Wa_lrf)

        trainer.peek_validation(count=10)

        results = trainer.run(n_batches=n_batches)
        scores[i] = results.validation["pc_loss"][-1]

    score = torch.quantile(scores, 0.90).item()
    return score


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization")

    parser.add_argument("out", help="output file")
    parser.add_argument("algo", help="algorithm: pcn, biopcn, wb")
    parser.add_argument("trials", type=int, help="number of trials")
    parser.add_argument("seed", type=int, help="starting random number seed")

    parser.add_argument("--n-batches", type=int, default=500, help="number of batches")
    parser.add_argument("--n-rep", type=int, default=4, help="number of repetitions")

    args = parser.parse_args()

    torch.set_num_threads(1)

    # using the GPU hinders more than helps for the smaller models
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(device)

    t0 = time.time()
    dataset = load_mnist(n_validation=500, batch_size=100, device=device)
    hidden_dims = [50, 5]

    one_sample = next(iter(dataset["train"]))
    dims = [one_sample[0].shape[-1]] + hidden_dims + [one_sample[1].shape[-1]]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            n_batches=args.n_batches,
            dataset=dataset,
            algo=args.algo,
            dims=dims,
            rho=[1.0, 0.1],
            seed=args.seed,
            n_rep=args.n_rep,
            accuracy_fct=one_hot_accuracy,
            constraint=args.algo != "wb",
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
