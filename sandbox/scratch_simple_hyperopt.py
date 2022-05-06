# %% [markdown]
# # A simple test of using `optuna` for hyperparameter optimization

import os
import time
from types import SimpleNamespace
from functools import partial

import pydove as dv
from tqdm.notebook import tqdm

import torch
from cpcn import LinearBioPCN, load_mnist, Trainer

import optuna
from optuna.trial import TrialState

import pickle

# %% [markdown]
# ## Defining the optimization


def optuna_reporter(trial: optuna.trial.Trial, ns: SimpleNamespace):
    trial.report(ns.val_loss, ns.epoch)

    # early pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


def create_biopcn(trial):
    dims = [28 * 28, 5, 10]

    rho = 0.015

    z_lr = trial.suggest_float("z_lr", 0.01, 0.2, log=True)
    # z_lr = 0.1

    # set parameters to match a simple PCN network
    g_a = 0.5 * torch.ones(len(dims) - 2)
    g_a[-1] *= 2

    g_b = 0.5 * torch.ones(len(dims) - 2)
    g_b[0] *= 2

    net = LinearBioPCN(
        dims,
        z_lr=z_lr,
        z_it=50,
        g_a=g_a,
        g_b=g_b,
        c_m=0,
        l_s=g_b,
        rho=rho,
        bias_a=False,
        bias_b=False,
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
    scores = torch.zeros(n_rep)
    for i in tqdm(range(n_rep)):
        torch.manual_seed(seed + i)
        net = create_biopcn(trial).to(device)

        optimizer_class = torch.optim.SGD
        lr = trial.suggest_float("lr", 5e-4, 0.05, log=True)

        trainer = Trainer(net, dataset["train"], dataset["validation"])
        trainer.set_optimizer(optimizer_class, lr=lr)

        lr_power = 1.0
        lr_rate = trial.suggest_float("lr_rate", 1e-5, 0.2, log=True)
        trainer.add_scheduler(
            partial(
                torch.optim.lr_scheduler.LambdaLR,
                lr_lambda=lambda batch: 1 / (1 + lr_rate * batch ** lr_power),
            ),
            every=1,
        )

        Q_lrf = trial.suggest_float("Q_lrf", 0.1, 20, log=True)
        trainer.set_lr_factor("Q", Q_lrf)

        trainer.peek_validation(count=10)

        # trainer.add_epoch_observer(lambda ns: optuna_reporter(trial, ns))
        results = trainer.run(n_batches=n_batches)
        scores[i] = results.validation["pc_loss"][-1]

    score = torch.quantile(scores, 0.90).item()
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

# %%

with open(os.path.join("save", "hyperopt_biopcn.pkl"), "wb") as f:
    pickle.dump(study, f)

# %%
