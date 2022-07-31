# %% [markdown]
# # Look at the rank of intermediate layers

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import time
import numpy as np
import torch

from tqdm.notebook import tqdm
from functools import partial

from cpcn import *
from cpcn.graph import *

# %% [markdown]
# ## Setup

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for reproducibility
seed = 123
torch.manual_seed(seed)

# get train, validation, and test loaders for MNIST
dataset = load_mnist(n_validation=1000)

# %% [markdown]
# ## Try PCN without constraint

# %%
n_batches = 3000
dims = [784, 50, 50, 10]
z_it = 70
z_lr = 0.1

# %%

t0 = time.time()
torch.manual_seed(seed)

net0 = PCNetwork(
    dims, activation=lambda _: _, z_lr=z_lr, z_it=z_it, variances=1.0, bias=False
)

net = PCWrapper(net0, "linear").to(device)
optimizer = torch.optim.SGD(net.pc_net.parameters(), lr=0.008)
predictor_optimizer = torch.optim.Adam(net.predictor.parameters())
trainer = Trainer(dataset["train"], invalid_action="warn+stop")
trainer.metrics["accuracy"] = one_hot_accuracy
for batch in tqdmw(trainer(n_batches)):
    if batch.every(10):
        batch.evaluate(dataset["validation"]).run(net)
        batch.weight.report({"W": net.pc_net.W})

    ns = batch.feed(net, latent_profile=True)
    batch.latent.report_batch("z", ns.z)
    if batch.count(4):
        batch.fast.report_batch("z", [_.transpose(0, 1) for _ in ns.profile.z])

    optimizer.step()
    predictor_optimizer.step()

results = trainer.history
print(f"Training PCN took {time.time() - t0:.1f} seconds.")

# %% [markdown]
# ## Check convergence of PCN without constraint

# %%

with dv.FigureManager() as (fig, ax):
    show_latent_convergence(results.fast)
    fig.suptitle("PCN no constraint")

fig = show_learning_curves(results)
fig.suptitle("PCN no constraint")

# %% [markdown]
# ## Check whitening of PCN without constraint

# %%

cons_diag = get_constraint_diagnostics(results.latent, rho=1.0)
fig = show_constraint_diagnostics(cons_diag, rho=1.0)
fig.suptitle("PCN no constraint")

crt_log_evals = np.mean(np.log10(cons_diag["evals:1"][-50:]), 0)
crt_log_dist = np.diff(crt_log_evals)
crt_idx = crt_log_dist.argmax()
crt_log_thresh = 0.5 * (crt_log_evals[crt_idx] + crt_log_evals[crt_idx + 1])
crt_thresh = 10 ** crt_log_thresh

with dv.FigureManager() as (fig, ax):
    ax.axhline(crt_thresh, c="k", ls="--", lw=1.0)
    ax.semilogy(cons_diag["batch"][-100:], cons_diag["evals:1"][-100:])
    ax.set_xlabel("batch")
    ax.set_ylabel("evals of $z z^\\top$")
    fig.suptitle("PCN no constraint")

print(f"approximate rank: {np.sum(cons_diag['evals:1'][-1] > crt_thresh)}")

# %% [markdown]
# ## What happens when we add a constraint?

# %%

t0 = time.time()
torch.manual_seed(seed)
rho = 0.1
net_cons0 = PCNetwork(
    dims,
    activation=lambda _: _,
    z_lr=z_lr,
    z_it=z_it,
    variances=1.0,
    constrained=True,
    rho=rho,
    bias=False,
)

net_cons = PCWrapper(net_cons0, "linear").to(device)
optimizer_cons = multi_lr(
    torch.optim.SGD, net_cons.pc_net.parameter_groups(), lr_factors={"Q": 2}, lr=0.008,
)
predictor_optimizer_cons = torch.optim.Adam(net_cons.predictor.parameters())
trainer_cons = Trainer(dataset["train"], invalid_action="warn+stop")
trainer_cons.metrics = trainer.metrics
for batch in tqdmw(trainer_cons(n_batches)):
    if batch.every(10):
        batch.evaluate(dataset["validation"]).run(net_cons)
        batch.weight.report({"W": net_cons.pc_net.W, "Q": net_cons.pc_net.Q})

    ns = batch.feed(net_cons, latent_profile=True)
    batch.latent.report_batch("z", ns.z)
    if batch.count(4):
        batch.fast.report_batch("z", [_.transpose(0, 1) for _ in ns.profile.z])

    optimizer_cons.step()
    predictor_optimizer_cons.step()

results_cons = trainer_cons.history
print(f"Training PCN with constraint took {time.time() - t0:.1f} seconds.")

# %% [markdown]
# ## Check convergence of PCN with constraint

# %%

with dv.FigureManager() as (fig, ax):
    show_latent_convergence(results_cons.fast)
    fig.suptitle("PCN with constraint")

fig = show_learning_curves(results_cons)
fig.suptitle("PCN with constraint")

# %% [markdown]
# ## Check whitening of PCN with constraint

# %%

cons_diag_cons = get_constraint_diagnostics(results_cons.latent, rho=rho)
fig = show_constraint_diagnostics(cons_diag_cons, rho=rho)
fig.suptitle("PCN with constraint")

with dv.FigureManager() as (fig, ax):
    ax.semilogy(cons_diag_cons["batch"][-100:], cons_diag_cons["evals:1"][-100:])
    ax.set_xlabel("batch")
    ax.set_ylabel("evals of $z z^\\top$")
    fig.suptitle("PCN with constraint")

# %% [markdown]
# ## Try CPCN

# %%

t0 = time.time()
torch.manual_seed(seed)

cpcn0 = LinearBioPCN.from_pcn(net_cons0).to(device)

cpcn = PCWrapper(cpcn0, "linear").to(device)
optimizer_cpcn = multi_lr(
    torch.optim.SGD, cpcn0.parameter_groups(), lr_factors={"Q": 0.5}, lr=0.008,
)
predictor_optimizer_cpcn = torch.optim.Adam(cpcn.predictor.parameters())
trainer_cpcn = Trainer(dataset["train"], invalid_action="warn+stop")
trainer_cpcn.metrics = trainer.metrics
for batch in tqdmw(trainer_cpcn(n_batches)):
    if batch.every(10):
        batch.evaluate(dataset["validation"]).run(cpcn)
        batch.weight.report({"W_a": cpcn0.W_a, "W_b": cpcn0.W_b, "Q": cpcn0.Q})

    ns = batch.feed(cpcn, latent_profile=True)
    batch.latent.report_batch("z", ns.z)
    if batch.count(4):
        batch.fast.report_batch("z", [_.transpose(0, 1) for _ in ns.profile.z])

    optimizer_cpcn.step()
    predictor_optimizer_cpcn.step()

results_cpcn = trainer_cpcn.history
print(f"Training BioPCN took {time.time() - t0:.1f} seconds.")

# %% [markdown]
# ## Check convergence of BioPCN

# %%

with dv.FigureManager() as (fig, ax):
    show_latent_convergence(results_cpcn.fast)
    fig.suptitle("BioPCN")

fig = show_learning_curves(results_cpcn)
fig.suptitle("BioPCN")

# %% [markdown]
# ## Check whitening of BioPCN

# %%

cons_diag_cpcn = get_constraint_diagnostics(results_cpcn.latent, rho=1.0)
fig = show_constraint_diagnostics(cons_diag_cpcn, rho=1.0)
fig.suptitle("BioPCN")

crt_log_evals = np.mean(np.log10(cons_diag_cpcn["evals:1"][-50:]), 0)
crt_log_dist = np.diff(crt_log_evals)
crt_idx = crt_log_dist.argmax()
crt_log_thresh = 0.5 * (crt_log_evals[crt_idx] + crt_log_evals[crt_idx + 1])
crt_thresh = 10 ** crt_log_thresh

with dv.FigureManager() as (fig, ax):
    ax.axhline(crt_thresh, c="k", ls="--", lw=1.0)
    ax.semilogy(cons_diag_cpcn["batch"][-100:], cons_diag_cpcn["evals:1"][-100:])
    ax.set_xlabel("batch")
    ax.set_ylabel("evals of $z z^\\top$")
    fig.suptitle("BioPCN")

print(f"approximate rank: {np.sum(cons_diag_cpcn['evals:1'][-1] > crt_thresh)}")

# %%

fig = show_constraint_diagnostics(cons_diag_cpcn, rho=1.0, layer=2)
fig.suptitle("BioPCN")

crt_log_evals = np.mean(np.log10(cons_diag_cpcn["evals:2"][-50:]), 0)
crt_log_dist = np.diff(crt_log_evals)
crt_idx = crt_log_dist.argmax()
crt_log_thresh = 0.5 * (crt_log_evals[crt_idx] + crt_log_evals[crt_idx + 1])
crt_thresh = 10 ** crt_log_thresh

with dv.FigureManager() as (fig, ax):
    ax.axhline(crt_thresh, c="k", ls="--", lw=1.0)
    ax.semilogy(cons_diag_cpcn["batch"][-100:], cons_diag_cpcn["evals:2"][-100:])
    ax.set_xlabel("batch")
    ax.set_ylabel("evals of $z z^\\top$")
    fig.suptitle("BioPCN")

print(f"approximate rank: {np.sum(cons_diag_cpcn['evals:2'][-1] > crt_thresh)}")

# %%
