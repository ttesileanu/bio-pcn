# %% [markdown]
# # A simple test of using `optuna` for hyperparameter optimization

import os
import time
from types import SimpleNamespace
from functools import partial

import pydove as dv
from tqdm.notebook import tqdm

import torch
from cpcn import PCNetwork, load_mnist, Trainer

import optuna
from optuna.trial import TrialState

import pickle

# %% [markdown]
# ## Defining the optimization


def create_pcn(trial):
    dims = [28 * 28, 5, 10]

    z_lr = trial.suggest_float("z_lr", 0.01, 0.2, log=True)
    rho = 0.015
    net = PCNetwork(
        dims,
        activation=lambda _: _,
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
    scores = torch.zeros(n_rep)
    for i in tqdm(range(n_rep)):
        torch.manual_seed(seed + i)
        net = create_pcn(trial).to(device)

        optimizer_class = torch.optim.SGD
        lr = trial.suggest_float("lr", 5e-4, 0.05, log=True)

        trainer = Trainer(net, dataset["train"], dataset["validation"])
        trainer.set_optimizer(optimizer_class, lr=lr).add_nan_guard()

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

        results = trainer.run(n_batches=n_batches)
        scores[i] = results.validation["pc_loss"][-1]

    score = torch.quantile(scores, 0.90).item()
    return score


# %%
# minimizing PC loss
t0 = time.time()

device = torch.device("cpu")

n_batches = 400
seed = 1927
n_rep = 5

dataset = load_mnist(n_validation=500, batch_size=100)

sampler = optuna.samplers.TPESampler(seed=seed)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(
    lambda trial: objective(trial, n_batches, dataset, device, seed, n_rep),
    n_trials=15,
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

with open(os.path.join("save", "hyperopt_pcn.pkl"), "wb") as f:
    pickle.dump(study, f)
