# %% [markdown]
# # A simple test of using `optuna` for hyperparameter optimization

import time
from types import SimpleNamespace

import pydove as dv
from tqdm.notebook import tqdm

import torch
from cpcn import LinearCPCNetwork, load_mnist, Trainer

import optuna
from optuna.trial import TrialState

# %% [markdown]
# ## Defining the optimization


def optuna_reporter(trial: optuna.trial.Trial, ns: SimpleNamespace):
    trial.report(ns.val_loss, ns.epoch)

    # early pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


def create_cpcn(trial):
    # n_hidden = trial.suggest_int("n_hidden", 1, 3)
    n_hidden = 1
    # dims = [28 * 28]
    # for i in range(n_hidden):
    #     n_units = trial.suggest_int(f"n_units_l{i}", 3, 64)
    #     dims.append(n_units)
    # dims.append(10)
    dims = [28 * 28, 5, 10]

    z_lr = trial.suggest_float("z_lr", 1e-5, 0.2, log=True)

    # set parameters to match a simple PCN network
    g_a = 0.5 * torch.ones(len(dims) - 2)
    g_a[-1] *= 2

    g_b = 0.5 * torch.ones(len(dims) - 2)
    g_b[0] *= 2

    net = LinearCPCNetwork(
        dims,
        z_lr=z_lr,
        z_it=50,
        g_a=g_a,
        g_b=g_b,
        c_m=0,
        l_s=g_b,
        bias_a=False,
        bias_b=False,
    )

    return net


def objective(
    trial: optuna.trial.Trial, n_epochs: int, dataset: dict, device: torch.device
) -> float:
    net = create_cpcn(trial).to(device)

    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer_class = getattr(torch.optim, optimizer_type)
    # optimizer_class = torch.optim.Adam
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    rep_gamma = trial.suggest_float("rep_gamma", 1e-7, 0.2, log=True)

    trainer = Trainer(net, dataset["train"], dataset["validation"])
    trainer.set_optimizer(optimizer_class, lr=lr)
    trainer.add_scheduler(
        lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, gamma=1 - rep_gamma)
    )

    trainer.add_epoch_observer(lambda ns: optuna_reporter(trial, ns))
    results = trainer.run(n_epochs)

    return results.validation.pc_loss[-1]


# %%
# minimizing PC loss
t0 = time.time()

device = torch.device("cpu")

n_epochs = 50
dataset = load_mnist(n_train=2000, n_validation=1000, batch_size=100)

study = optuna.create_study(direction="minimize")
study.optimize(
    lambda trial: objective(trial, n_epochs, dataset, device),
    n_trials=200,
    timeout=1800,
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
