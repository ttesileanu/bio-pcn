# %% [markdown]
# # A simple test of using `optuna` for hyperparameter optimization

import os
import time
from types import SimpleNamespace

import pydove as dv
from tqdm.notebook import tqdm

import numpy as np
import torch
from cpcn import PCNetwork, load_mnist, Trainer, tqdmw, multi_lr

import optuna
from optuna.trial import TrialState

# %% [markdown]
# ## Defining the optimization


def create_pcn(trial):
    dims = [28 * 28, 5, 10]

    z_lr = trial.suggest_float("z_lr", 0.01, 0.2, log=True)
    rho = 0.015

    net = PCNetwork(
        dims,
        activation="none",
        z_lr=z_lr,
        z_it=50,
        variances=1,
        bias=False,
        constrained=True,
        rho=rho,
    )

    return net


def objective(
    trial: optuna.trial.Trial,
    n_batches: int,
    dataset: dict,
    device: torch.device,
    seed: int,
    n_rep: int,
) -> float:
    scores = np.zeros(n_rep)
    for i in tqdm(range(n_rep)):
        torch.manual_seed(seed + i)
        net = create_pcn(trial).to(device)

        lr = trial.suggest_float("lr", 5e-4, 0.05, log=True)
        lr_rate = trial.suggest_float("lr_rate", 1e-5, 0.2, log=True)
        Q_lrf = trial.suggest_float("Q_lrf", 0.1, 20, log=True)

        optimizer = multi_lr(
            torch.optim.SGD, net.parameter_groups(), lr_factors={"Q": Q_lrf}, lr=lr
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda batch: 1 / (1 + lr_rate * batch)
        )

        trainer = Trainer(dataset["train"], invalid_action="warn+stop")
        for batch in trainer(n_batches):
            if batch.every(10):
                batch.evaluate(dataset["validation"]).run(net)

            batch.feed(net)
            optimizer.step()
            scheduler.step()

        scores[i] = trainer.history.validation["pc_loss"][-1]

    score = np.quantile(scores, 0.90)
    return score


# %%
# minimizing PC loss
t0 = time.time()

device = torch.device("cpu")

n_batches = 500
seed = 1927
n_rep = 5

dataset = load_mnist(n_validation=500, batch_size=100)

sampler = optuna.samplers.TPESampler(seed=seed)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(
    lambda trial: objective(trial, n_batches, dataset, device, seed, n_rep),
    n_trials=25,
    timeout=15000,
    show_progress_bar=True,
)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

t1 = time.time()

# %%
print(
    f"{len(study.trials)} trials in {t1 - t0:.1f} seconds: "
    f"{len(complete_trials)} complete, {len(pruned_trials)} pruned."
)

trial = study.best_trial
print(f"best pc_loss: {trial.value}, for params:")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%

optuna.visualization.matplotlib.plot_param_importances(study)
