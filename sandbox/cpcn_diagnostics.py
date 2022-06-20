# %% [markdown]
# # Diagnosing BioPCN

from types import SimpleNamespace
from functools import partial
import pydove as dv
import matplotlib as mpl

import torch
import time

from tqdm.notebook import tqdm

from cpcn import *
from cpcn.graph import *

# %% [markdown]
# ## Setup

# %%

device = torch.device("cpu")

# for reproducibility
torch.manual_seed(123)

# this creates the loaders
batch_size = 100
dataset = load_mnist(n_validation=1000, batch_size=batch_size, device=device)

# %% [markdown]
# ## Train BioPCN

# %%

n_batches = 6000
dims = [784, 5, 10]

z_it = 80
z_lr = 0.1
# rho = 0.015
rho = 0.0012

t0 = time.time()
torch.manual_seed(123)

# match the PCN network
g_a = 0.5 * torch.ones(len(dims) - 2)
g_a[-1] *= 2

g_b = 0.5 * torch.ones(len(dims) - 2)
g_b[0] *= 2

net0 = LinearBioPCN(
    dims,
    z_lr=z_lr,
    z_it=z_it,
    g_a=g_a,
    g_b=g_b,
    c_m=0,
    l_s=g_b,
    rho=rho,
    fast_optimizer=torch.optim.Adam,
    bias_a=False,
    bias_b=False,
)

net = PCWrapper(net0, "linear").to(device)
optimizer = multi_lr(
    torch.optim.SGD, net.pc_net.parameter_groups(), lr_factors={"Q": 20.0}, lr=0.008,
)
predictor_optimizer = torch.optim.Adam(net.predictor.parameters())
trainer = Trainer(dataset["train"], invalid_action="warn+stop")
trainer.metrics["accuracy"] = one_hot_accuracy
for batch in tqdmw(trainer(n_batches)):
    if batch.every(10):
        batch.evaluate(dataset["validation"]).run(net)
        batch.weight.report(
            {"W_a": net.pc_net.W_a, "W_b": net.pc_net.W_b, "Q": net.pc_net.Q,}
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

D = len(net.pc_net.inter_dims)
with dv.FigureManager(3, D, squeeze=False) as (_, axs):
    for ax_row, w_choice in zip(axs, ["W_a", "W_b", "Q"]):
        for k, ax in enumerate(ax_row):
            show_weight_evolution(
                results.weight["batch"], results.weight[f"{w_choice}:{k}"], ax=ax
            )
            ax.set_title(f"${w_choice}[{k}]$")

# %%
