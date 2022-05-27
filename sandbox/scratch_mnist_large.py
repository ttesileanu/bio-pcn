# %% [markdown]
# # Comparison of PCN and BioPCN on MNIST with two large hidden (linear) layers

# %%
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
    load_mnist,
    Trainer,
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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# for reproducibility
torch.manual_seed(123)

# get train, validation, and test loaders for MNIST
dataset = load_mnist(n_validation=500, device=device)

# %% [markdown]
# ## Train PCN

# %%
n_batches = 2000
dims = [784, 50, 5, 10]
z_it = 50
z_lr = 0.07

torch.manual_seed(123)

net = PCNetwork(
    dims,
    activation=lambda _: _,
    z_lr=z_lr,
    z_it=z_it,
    variances=1.0,
    # constrained=True,
    constrained=False,
    bias=False,
)
net = net.to(device)

trainer = Trainer(net, dataset["train"], dataset["validation"])
trainer.peek_validation(every=10)
# trainer.set_classifier("linear")

trainer.set_optimizer(torch.optim.SGD, lr=0.008)
# trainer.set_optimizer(torch.optim.Adam, lr=0.002)
# trainer.add_scheduler(partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.99))

trainer.peek_sample("latent", ["z"])
trainer.peek_fast_dynamics("fast", ["z"], count=4)
trainer.peek("weight", ["W"], every=10)

results = trainer.run(n_batches=n_batches, progress=tqdm)

# %% [markdown]
# ### Show PCN learning curves

# %%

_ = show_learning_curves(results)
_ = show_learning_curves(results, var_names=("pc_loss", "prediction_error"))

# %% [markdown]
# ### Show covariance diagnostics

cons_diag = get_constraint_diagnostics(results.latent, rho=[0.2, 0.02])
_ = show_constraint_diagnostics(cons_diag, layer=1, rho=0.2)
_ = show_constraint_diagnostics(cons_diag, layer=2, rho=0.02)

# %% [markdown]
# ## Train BioPCN

# %%
z_it = 50
z_lr = 0.02
Q_lr_factor = 6
# rho = [2.0, 0.3]
# rho = [1.0, 0.1]
# rho = [5.0, 0.5]
rho = [0.2, 0.02]
# rho = [0.05, 0.005]
# rho = 0.1
# rho = 0.0012

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
    # l_s=g_b,
    l_s=g_a + g_b,
    rho=rho,
    bias_a=False,
    bias_b=False,
    q0_scale=np.sqrt(1 + dims[2] / dims[1]),
)
biopcn_net = biopcn_net.to(device)

biopcn_trainer = Trainer(biopcn_net, dataset["train"], dataset["validation"])
biopcn_trainer.peek_validation(every=10).add_nan_guard(every=10)
# biopcn_trainer.set_classifier("linear")

# biopcn_trainer.set_optimizer(torch.optim.Adam, lr=0.001)
biopcn_trainer.set_optimizer(torch.optim.SGD, lr=0.012)
# biopcn_trainer.set_optimizer(torch.optim.SGD, lr=0.001)
biopcn_trainer.set_lr_factor("Q", Q_lr_factor)
biopcn_trainer.set_lr_factor("W_a:0", 2)
biopcn_trainer.set_lr_factor("W_a:1", 2)
# biopcn_trainer.add_scheduler(partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.997))

lr_power = 1.0
# lr_rate = -0.4e-3
lr_rate = 0
biopcn_trainer.add_scheduler(
    partial(
        torch.optim.lr_scheduler.LambdaLR,
        lr_lambda=lambda batch: 1 / (1 + lr_rate * batch ** lr_power),
    ),
    every=1,
)

biopcn_trainer.peek_sample("latent", ["z"])
biopcn_trainer.peek_fast_dynamics("fast", ["z"], count=4)
biopcn_trainer.peek("weight", ["W_a", "W_b", "Q"], every=10)

biopcn_results = biopcn_trainer.run(n_batches=n_batches, progress=tqdm)

# %% [markdown]
# ### Show BioPCN learning curves

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

# %%

with dv.FigureManager(1, 2) as (_, (ax1, ax2)):
    show_latent_convergence(biopcn_results.fast, layer=1, ax=ax1)
    show_latent_convergence(biopcn_results.fast, layer=2, ax=ax2)

# %% [markdown]
# ### Show covariance diagnostics

biopcn_cons_diag = get_constraint_diagnostics(biopcn_results.latent, rho=biopcn_net.rho)
_ = show_constraint_diagnostics(biopcn_cons_diag, layer=1, rho=biopcn_net.rho[0])
_ = show_constraint_diagnostics(biopcn_cons_diag, layer=2, rho=biopcn_net.rho[1])

# %%
