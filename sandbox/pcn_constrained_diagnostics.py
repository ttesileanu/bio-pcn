# %% [markdown]
# # Diagnosing PCN with constraint

from types import SimpleNamespace
from functools import partial
import pydove as dv
import matplotlib as mpl

import torch

from tqdm.notebook import tqdm

from cpcn import PCNetwork, load_mnist, Trainer

# %% [markdown]
# ## Setup

# %%

device = torch.device("cpu")

# for reproducibility
torch.manual_seed(123)

# this creates the loaders
batch_size = 100
dataset = load_mnist(
    n_train=2000, n_validation=500, batch_size=batch_size, device=device
)

# %% [markdown]
# ## Train PCN

# %%

n_epochs = 300
dims = [784, 5, 10]

z_it = 50
z_lr = 0.1
rho = 0.015

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
trainer.set_classifier("linear")

trainer.set_optimizer(torch.optim.Adam, lr=0.001)
# trainer.add_scheduler(partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9))

trainer.peek_epoch("weight", ["W", "Q"])
trainer.peek_sample("latent", ["z"])

fast_epochs = torch.linspace(0, n_epochs - 1, 4).int()
batch_max = len(dataset["train"])
trainer.peek_fast_dynamics(
    "fast",
    ["z"],
    condition=lambda epoch, batch: epoch in fast_epochs and batch == batch_max - 1,
)

results = trainer.run(n_epochs, progress=tqdm)

# %% [markdown]
# ## Check convergence of latent variables

# %%

with dv.FigureManager() as (_, ax):
    cmap = mpl.cm.winter
    crt_sel = trainer.history.fast["sample"] == batch_size - 1

    crt_epoch = trainer.history.fast["epoch"][crt_sel]
    crt_z = trainer.history.fast["z:1"][crt_sel]

    n = len(crt_epoch)
    for i in range(n):
        color = cmap(int(cmap.N * (0.2 + 0.8 * i / n)))
        crt_diff = crt_z[i, :, :] - crt_z[i, 0, :]
        crt_diff = crt_diff / torch.max(torch.abs(crt_diff))
        ax.plot(crt_diff, c=color, lw=0.5)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.pyplot.Normalize(vmin=0, vmax=n))
    sm.ax = ax
    cbar = dv.colorbar(sm)
    cbar.set_label("epoch")

    ax.set_xlabel("fast dynamics iteration")
    ax.set_ylabel("latent $(z - z_0) / \\rm{max}| z - z_0|$")

# %% [markdown]
# ## Check whitening constraint in hidden layer

# %%

hidden_size = net.dims[1]
z_cov = torch.zeros((n_epochs, hidden_size, hidden_size))
for epoch in range(n_epochs):
    crt_sel = trainer.history.latent["epoch"] == epoch
    crt_z = trainer.history.latent["z:1"][crt_sel]

    z_cov[epoch] = crt_z.T @ crt_z / len(crt_z)

tr_cons = [
    torch.trace(z_cov[i] - net.rho[0] * torch.eye(hidden_size)) for i in range(n_epochs)
]
sel_epochs = torch.linspace(0, n_epochs - 1, 5).int()
with dv.FigureManager(
    1, len(sel_epochs), do_despine=False, figsize=(2.5 * len(sel_epochs), 2)
) as (_, axs):
    for ax, epoch in zip(axs, sel_epochs):
        ax.imshow(z_cov[epoch])
        ax.set_title(f"epoch {epoch}")

max_eval = [torch.max(torch.linalg.eigh(_)[0]) for _ in z_cov]
with dv.FigureManager(1, 2) as (_, (ax1, ax2)):
    ax1.axhline(0, c="k", ls="--")
    ax1.plot(tr_cons)
    ax1.annotate(
        f"{tr_cons[-1]:.2g}",
        (len(tr_cons), tr_cons[-1]),
        xytext=(3, 0),
        textcoords="offset points",
        va="center",
        c="C0",
        fontweight="bold",
    )
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("trace of constraint")

    ax2.axhline(rho, c="k", ls="--")
    ax2.plot(max_eval)
    ax2.annotate(
        f"{max_eval[-1]:.2g}",
        (len(max_eval), max_eval[-1]),
        xytext=(3, 0),
        textcoords="offset points",
        va="center",
        c="C0",
        fontweight="bold",
    )
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("maximum $z$-cov eval")
    ax2.set_ylim(0, 5 * rho)

# %% [markdown]
# ## Show loss and accuracy evolution

# %%

with dv.FigureManager(1, 2) as (_, (ax1, ax2)):
    ax1.plot(results.train["pc_loss"], label="train")
    ax1.plot(results.validation["pc_loss"], label="val")
    ax1.legend(frameon=False)
    ax1.annotate(
        f"{results.validation['pc_loss'][-1]:.2g}",
        (n_epochs, results.validation["pc_loss"][-1]),
        xytext=(3, 0),
        textcoords="offset points",
        c="C1",
        va="center",
        fontweight="bold",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("predictive-coding loss")

    train_error_rate = 100 * (1.0 - results.train["accuracy"])
    val_error_rate = 100 * (1.0 - results.validation["accuracy"])
    ax2.plot(train_error_rate, label="train")
    ax2.plot(val_error_rate, label="val")
    ax2.legend(frameon=False)
    ax2.annotate(
        f"{val_error_rate[-1]:.1f}%",
        (n_epochs, val_error_rate[-1]),
        xytext=(3, 0),
        textcoords="offset points",
        c="C1",
        va="center",
        fontweight="bold",
    )
    ax2.set_ylim(0, None)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("error rate (%)")

# %% [markdown]
# ## Check weight evolution

# %%

D = len(net.dims) - 1
with dv.FigureManager(1, D) as (_, axs):
    for k, ax in enumerate(axs):
        crt_data = trainer.history.weight[f"W:{k}"].reshape(n_epochs, -1)
        n_lines = crt_data.shape[1]
        alpha = max(min(50 / n_lines, 0.5), 0.01)
        ax.plot(crt_data, c="k", lw=0.5, alpha=alpha)

        ax.set_xlabel("epoch")
        ax.set_ylabel("weight")

# %%
