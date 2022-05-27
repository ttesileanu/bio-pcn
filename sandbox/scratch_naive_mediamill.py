# %% [markdown]
# # Quick comparison of PCN and BioPCN on Mediamill

# %%

import os.path as osp

import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import numpy as np
import torch

from tqdm.notebook import tqdm
from functools import partial

from cpcn import (
    LinearBioPCN,
    PCNetwork,
    load_csv,
    Trainer,
    dot_accuracy,
    get_constraint_diagnostics,
)
from cpcn.graph import (
    show_learning_curves,
    show_constraint_diagnostics,
    show_latent_convergence,
)

# %% [markdown]
# ## Setup

# %%
device = torch.device("cpu")

# for reproducibility
torch.manual_seed(123)

# get train and validation loaders for Mediamill
path = osp.join("data", "mediamill")
batch_size = 100
dataset = load_csv(
    osp.join(path, "view1.csv"),
    osp.join(path, "view2.csv"),
    n_validation=1000,
    batch_size=batch_size,
    device=device,
)

# %% [markdown]
# ## Train PCN

# %%
one_batch = next(iter(dataset["train"]))
n_in = one_batch[0].shape[1]
n_out = one_batch[1].shape[1]
dims = [n_in, 5, n_out]
n_batches = 2000
z_it = 50
z_lr = 0.062

torch.manual_seed(123)

net = PCNetwork(
    dims,
    activation="none",
    z_lr=z_lr,
    z_it=z_it,
    variances=1.0,
    constrained=False,
    bias=False,
)
net = net.to(device)

trainer = Trainer(net, dataset["train"], dataset["validation"])
trainer.set_accuracy_fct(dot_accuracy)
trainer.peek_validation(every=10).add_nan_guard()
trainer.set_classifier("linear")

trainer.set_optimizer(torch.optim.SGD, lr=0.01)
# trainer.set_lr_factor("Q", 0.11)
lr_power = 1.0
lr_rate = 4e-4
trainer.add_scheduler(
    partial(
        torch.optim.lr_scheduler.LambdaLR,
        lr_lambda=lambda batch: 1 / (1 + lr_rate * batch ** lr_power),
    ),
    every=1,
)

trainer.peek_model(count=4)
trainer.peek_sample("latent", ["z"])
trainer.peek_fast_dynamics("fast", ["z"], count=4)
trainer.peek("weight", ["W"], every=10)

results = trainer.run(n_batches=n_batches, progress=tqdm)

# %% [markdown]
# ### Show PCN learning curves

# %%

with dv.FigureManager() as (_, ax):
    show_latent_convergence(results.fast, ax=ax)

# %%

_ = show_learning_curves(results)
_ = show_learning_curves(results, var_names=("pc_loss", "prediction_error"))

# %% [markdown]
# ### Show covariance diagnostics

rho = 0.05
# rho = 0.5
# rho = 0.015
# rho = 0.0012
cons_diag = get_constraint_diagnostics(results.latent, rho=rho)
_ = show_constraint_diagnostics(cons_diag, rho=rho)

# %% [markdown]
# ## Train BioPCN

# %%
z_it = 50
z_lr = 0.13

torch.manual_seed(123)

# match the PCN network
g_a = 0.5 * np.ones(len(dims) - 2)
g_a[-1] *= 2

g_b = 0.5 * np.ones(len(dims) - 2)
g_b[0] *= 2

biopcn_net = LinearBioPCN(
    dims,
    z_lr=z_lr,
    z_it=z_it,
    g_a=g_a,
    g_b=g_b,
    c_m=0,
    l_s=g_b,
    rho=rho,
    bias_a=False,
    bias_b=False,
    q0_scale=np.sqrt(1 + dims[2] / dims[1]),
)
biopcn_net = biopcn_net.to(device)

biopcn_trainer = Trainer(biopcn_net, dataset["train"], dataset["validation"])
biopcn_trainer.set_accuracy_fct(dot_accuracy)
biopcn_trainer.peek_validation(every=10)
biopcn_trainer.set_classifier("linear")

biopcn_trainer.set_optimizer(torch.optim.SGD, lr=0.003)
biopcn_trainer.set_lr_factor("Q", 3.5)
lr_power = 1.0
lr_rate = 2e-4
biopcn_trainer.add_scheduler(
    partial(
        torch.optim.lr_scheduler.LambdaLR,
        lr_lambda=lambda batch: 1 / (1 + lr_rate * batch ** lr_power),
    ),
    every=1,
)

biopcn_trainer.peek_model(count=4)
biopcn_trainer.peek_sample("latent", ["z"])
biopcn_trainer.peek_fast_dynamics("fast", ["z"], count=4)
biopcn_trainer.peek("weight", ["W_a", "W_b", "Q"], every=10)

biopcn_results = biopcn_trainer.run(n_batches=n_batches, progress=tqdm)

# %% [markdown]
# ### Show BioPCN learning curves

# %%

with dv.FigureManager() as (_, ax):
    show_latent_convergence(biopcn_results.fast, ax=ax)

# %%

with dv.FigureManager(1, 2) as (_, axs):
    show_learning_curves(
        results,
        show_train=False,
        labels=("", "Whittington&Bogacz"),
        colors=("C0", "gray"),
        axs=axs,
    )
    show_learning_curves(
        biopcn_results,
        show_train=False,
        labels=("", "BioPCN"),
        colors=("C0", "red"),
        axs=axs,
    )

with dv.FigureManager(1, 2) as (_, axs):
    show_learning_curves(
        results,
        show_train=False,
        labels=("", "Whittington&Bogacz"),
        colors=("C0", "gray"),
        var_names=("pc_loss", "prediction_error"),
        axs=axs,
    )
    show_learning_curves(
        biopcn_results,
        show_train=False,
        labels=("", "BioPCN"),
        colors=("C0", "red"),
        var_names=("pc_loss", "prediction_error"),
        axs=axs,
    )

# %% [markdown]
# ### Show covariance diagnostics

biopcn_cons_diag = get_constraint_diagnostics(biopcn_results.latent, rho=rho)
_ = show_constraint_diagnostics(biopcn_cons_diag, rho=rho)

# %%
