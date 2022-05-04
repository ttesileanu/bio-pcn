# %% [markdown]
# # Diagnosing PCN with constraint

from types import SimpleNamespace
from functools import partial
import pydove as dv
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch

from tqdm.notebook import tqdm

from cpcn import *

# %% [markdown]
# ## Setup

# %%

device = torch.device("cpu")

# for reproducibility
torch.manual_seed(123)

# get train, validation, and test loaders for MNIST
batch_size = 100
dataset = load_mnist(n_validation=1000, batch_size=batch_size, device=device)

# %% [markdown]
# ## Train PCN

# %%

n_batches = 3000
dims = [784, 5, 10]

z_it = 50
z_lr = 0.07
# rho = 0.015
# rho = 0.001875
rho = 0.0012

torch.manual_seed(123)

net = PCNetwork(
    dims,
    lr_inference=z_lr,
    it_inference=z_it,
    constrained=True,
    rho=rho,
    fast_optimizer=torch.optim.Adam,
    bias=False,
)
net = net.to(device)

trainer = Trainer(net, dataset["train"], dataset["validation"])
trainer.peek_validation(every=10)
trainer.set_classifier("linear")

trainer.set_optimizer(torch.optim.Adam, lr=0.003)
# trainer.add_scheduler(partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9))

if net.constrained:
    trainer.peek("weight", ["W", "Q"], every=10)
else:
    trainer.peek("weight", ["W"], every=10)
trainer.peek_sample("latent", ["z"])

trainer.peek_fast_dynamics("fast", ["z"], count=4)

results = trainer.run(n_batches=n_batches, progress=tqdm)

# %% [markdown]
# ## Check convergence of latent variables

# %%

with dv.FigureManager() as (_, ax):
    show_latent_convergence(results.fast)

# %% [markdown]
# ## Check whitening constraint in hidden layer

# %%

cons_diag = get_constraint_diagnostics(results.latent, rho=rho)
_ = show_constraint_diagnostics(cons_diag, rho=rho)

# %% [markdown]
# ## Show loss and accuracy evolution

# %%

_ = show_learning_curves(results)

# %% [markdown]
# ## Check weight evolution

# %%

D = len(net.dims) - 1
with dv.FigureManager(1, D) as (_, axs):
    for k, ax in enumerate(axs):
        show_weight_evolution(results.weight["batch"], results.weight[f"W:{k}"], ax=ax)

# %%
