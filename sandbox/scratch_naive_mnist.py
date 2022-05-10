# %% [markdown]
# # Quick comparison of PCN and BioPCN on MNIST

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import numpy as np
import torch

from tqdm.notebook import tqdm
from functools import partial

from cpcn import LinearBioPCN, PCNetwork, load_mnist, Trainer
from cpcn.graph import show_learning_curves

# %% [markdown]
# ## Setup

# %%
device = torch.device("cpu")

# for reproducibility
torch.manual_seed(123)

# get train, validation, and test loaders for MNIST
dataset = load_mnist(n_validation=1000, device=device)

# %% [markdown]
# ## Train PCN

# %%
n_batches = 3000
dims = [784, 5, 10]
z_it = 50
z_lr = 0.062
rho = 0.015
# rho = 0.0012

torch.manual_seed(123)

net = PCNetwork(
    dims,
    activation="none",
    z_lr=z_lr,
    z_it=z_it,
    variances=1.0,
    constrained=True,
    rho=rho,
    bias=False,
)
net = net.to(device)

trainer = Trainer(net, dataset["train"], dataset["validation"])
trainer.peek_validation(every=10).add_nan_guard()
trainer.set_classifier("linear")

trainer.set_optimizer(torch.optim.SGD, lr=0.01)
trainer.set_lr_factor("Q", 0.11)
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

results = trainer.run(n_batches=n_batches, progress=tqdm)

# %% [markdown]
# ### Show PCN learning curves

# %%

_ = show_learning_curves(results)

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

# %%
