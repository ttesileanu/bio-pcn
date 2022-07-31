# %% [markdown]
# # Diagnosing PCN with constraint

from types import SimpleNamespace
from functools import partial
import pydove as dv
import matplotlib as mpl
import matplotlib.pyplot as plt

import time
import torch

from tqdm.notebook import tqdm

from cpcn import *
from cpcn.graph import *

# %% [markdown]
# ## Setup

# %%

device = torch.device("cpu")

# for reproducibility
torch.manual_seed(123)

# get train, validation, and test loaders for MNIST
batch_size = 100
dataset = load_mnist(n_validation=1000, batch_size=batch_size)

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

t0 = time.time()
torch.manual_seed(123)

net0 = PCNetwork(
    dims,
    z_lr=z_lr,
    z_it=z_it,
    constrained=True,
    rho=rho,
    fast_optimizer=torch.optim.Adam,
    bias=False,
)

net = PCWrapper(net0, "linear").to(device)
optimizer = multi_lr(
    torch.optim.SGD, net.pc_net.parameter_groups(), lr_factors={"Q": 10.0}, lr=0.02,
)
predictor_optimizer = torch.optim.Adam(net.predictor.parameters())
trainer = Trainer(dataset["train"], invalid_action="warn+stop")
trainer.metrics["accuracy"] = one_hot_accuracy
for batch in tqdmw(trainer(n_batches)):
    if batch.every(10):
        batch.evaluate(dataset["validation"]).run(net)
        batch.weight.report(
            {"W": net.pc_net.W, "Q": net.pc_net.Q,}
        )

    ns = batch.feed(net, latent_profile=True)
    batch.latent.report_batch("z", ns.z)
    if batch.count(4):
        batch.fast.report_batch("z", [_.transpose(0, 1) for _ in ns.profile.z])

    optimizer.step()
    predictor_optimizer.step()

results = trainer.history
print(f"Training BioPCN took {time.time() - t0:.1f} seconds.")

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

D = len(net.pc_net.dims) - 1
with dv.FigureManager(1, D) as (_, axs):
    for k, ax in enumerate(axs):
        show_weight_evolution(results.weight["batch"], results.weight[f"W:{k}"], ax=ax)
        ax.set_title(f"W:{k}")

with dv.FigureManager() as (_, ax):
    show_weight_evolution(results.weight["batch"], results.weight["Q:0"], ax=ax)
    ax.set_title("Q")

# %%
