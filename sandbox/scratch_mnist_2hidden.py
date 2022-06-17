# %% [markdown]
# # Comparison of PCN and BioPCN on MNIST with two hidden (linear) layers

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import numpy as np
import torch
import time

from cpcn import *
from cpcn.graph import *

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
dims = [784, 10, 5, 10]
z_it = 50
z_lr = 0.07
rho = 0.02
# rho = 0.0012

t0 = time.time()
torch.manual_seed(123)

net0 = PCNetwork(
    dims,
    activation=lambda _: _,
    z_lr=z_lr,
    z_it=z_it,
    variances=1.0,
    constrained=True,
    rho=rho,
    bias=False,
)

net = PCWrapper(net0, "linear").to(device)
optimizer = torch.optim.SGD(net.pc_net.parameters(), lr=0.008)
predictor_optimizer = torch.optim.Adam(net.predictor.parameters())
trainer = Trainer(dataset["train"])
trainer.metrics["accuracy"] = one_hot_accuracy
for batch in tqdmw(trainer(n_batches)):
    if batch.every(10):
        batch.evaluate(dataset["validation"]).run(net)

    ns = batch.feed(net)
    optimizer.step()
    predictor_optimizer.step()

results = trainer.history
print(f"Training PCN took {time.time() - t0:.1f} seconds.")

# %% [markdown]
# ### Show PCN learning curves

# %%

_ = show_learning_curves(results)

# %% [markdown]
# ## Train BioPCN

# %%
z_it = 50
z_lr = 0.02
Q_lr_factor = 2.0

t0 = time.time()
torch.manual_seed(123)

# match the PCN network
g_a = 0.5 * np.ones(len(dims) - 2)
g_a[-1] *= 2

g_b = 0.5 * np.ones(len(dims) - 2)
g_b[0] *= 2

biopcn_net0 = LinearBioPCN(
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

biopcn_net = PCWrapper(biopcn_net0, "linear").to(device)
biopcn_optimizer = multi_lr(
    torch.optim.SGD,
    biopcn_net.pc_net.parameter_groups(),
    lr_factors={"Q": Q_lr_factor},
    lr=0.008,
)
biopcn_predictor_optimizer = torch.optim.Adam(biopcn_net.predictor.parameters())
biopcn_trainer = Trainer(dataset["train"])
biopcn_trainer.metrics = trainer.metrics
for batch in tqdmw(biopcn_trainer(n_batches)):
    if batch.every(10):
        batch.evaluate(dataset["validation"]).run(biopcn_net)

    ns = batch.feed(biopcn_net)
    biopcn_optimizer.step()
    biopcn_predictor_optimizer.step()

biopcn_results = biopcn_trainer.history
print(f"Training BioPCN took {time.time() - t0:.1f} seconds.")

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
