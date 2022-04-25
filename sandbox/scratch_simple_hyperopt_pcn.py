# %% [markdown]
# # A simple test of using `optuna` for hyperparameter optimization

import time
from types import SimpleNamespace

import pydove as dv
from tqdm.notebook import tqdm

import torch
from cpcn import PCNetwork, load_mnist, Trainer

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


def create_pcn(trial):
    dims = [28 * 28, 5, 10]

    z_lr = trial.suggest_float("z_lr", 1e-5, 0.2, log=True)
    net = PCNetwork(dims, lr_inference=z_lr, it_inference=50, variances=1, bias=False)

    return net


def objective(
    trial: optuna.trial.Trial,
    n_epochs: int,
    dataset: dict,
    device: torch.device,
    seed: int,
    n_rep: int,
) -> float:
    scores = torch.zeros(n_rep)
    for i in range(n_rep):
        torch.manual_seed(seed + i)

        net = create_pcn(trial).to(device)

        # optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        # optimizer_class = getattr(torch.optim, optimizer_type)
        optimizer_class = torch.optim.Adam
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        rep_gamma = trial.suggest_float("rep_gamma", 1e-7, 0.2, log=True)

        trainer = Trainer(net, dataset["train"], dataset["validation"])
        trainer.set_optimizer(optimizer_class, lr=lr)
        trainer.add_scheduler(
            lambda optim: torch.optim.lr_scheduler.ExponentialLR(
                optim, gamma=1 - rep_gamma
            )
        )

        # trainer.add_epoch_observer(lambda ns: optuna_reporter(trial, ns))
        results = trainer.run(n_epochs)
        scores[i] = results.validation.pc_loss[-1]

    score = torch.quantile(scores, 0.90)
    return score


# %%
# minimizing PC loss
t0 = time.time()

device = torch.device("cpu")

n_epochs = 30
seed = 1927
n_rep = 8

dataset = load_mnist(n_train=2000, n_validation=500, batch_size=100)

sampler = optuna.samplers.TPESampler(seed=seed)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(
    lambda trial: objective(trial, n_epochs, dataset, device, seed, n_rep),
    n_trials=50,
    timeout=8000,
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

with open("hyperopt_pcn.pkl", "wb") as f:
    pickle.dump(study, f)
