# %% [markdown]
# # Make figures showing weight (a)symmetry

import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import numpy as np

from tqdm.notebook import tqdm

import pickle

# %%

paper_style = [
    "seaborn-paper",
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.labelpad": 1,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "xtick.minor.pad": 1,
        "ytick.minor.pad": 1,
        "xtick.major.size": 1,
        "ytick.major.size": 1,
        "xtick.minor.size": 0.75,
        "ytick.minor.size": 0.75,
        "legend.labelspacing": 0,
        "legend.handlelength": 1.5,
        "axes.linewidth": 0.5,
    },
]

label_map = {"wb": "PC", "biopcn": "BioCCPC", "pcn": "PC + cons"}
color_map = {"wb": "C3", "pcn": "C5", "biopcn": "C0"}
style_map = {"wb": {"ls": "--"}, "pcn": {}, "biopcn": {}}

# mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
#     color=[
#         "#555CA9",  # liberty
#         "#9B4688",  # violet crayola
#         "#CC8B8B",  # old rose
#         "#353531",  # black olive
#         "#000000",  # black
#     ]
# )

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#F15B40",  # red
        "#6DAADD",  # blue
        "#C5AC61",  # green-brown
        "#8C8E90",  # gray
        "#F9AA8F",  # light red
        "#B2CDEC",  # light blue
    ]
)

fig_path = "figs"

# %%

context = "mnist_biopcn_large_rho1.0_0.1"
name = osp.join("simulations", context, "checkpoints_700.pkl")
with open(name, "rb") as f:
    checkpoints = pickle.load(f)

model = checkpoints[-1]

# %%

with plt.style.context(paper_style):
    with dv.FigureManager(
        figsize=(2.5, 1.75), despine_kws={"offset": 2}, constrained_layout=True
    ) as (fig, ax):
        ax.scatter(
            model.W_a[0].detach().numpy().ravel(),
            model.W_b[1].detach().numpy().ravel(),
            alpha=0.2,
        )

        ax.set_xlabel("$W_a$")
        ax.set_ylabel("$W_b$")

fig.savefig(osp.join(fig_path, "mnist_weight_asymmetry.pdf"))

# %%

# context = "mmill_biopcn_large_rho1.0"
# context = "mmill_biopcn_large_rho1.0_long"
context = "mmill_biopcn_large_rho1.0_0.1"
name = osp.join("simulations", context, "checkpoints_700.pkl")
with open(name, "rb") as f:
    checkpoints = pickle.load(f)

model = checkpoints[-1]

# %%

with plt.style.context(paper_style):
    with dv.FigureManager(
        figsize=(2.5, 1.75), despine_kws={"offset": 2}, constrained_layout=True
    ) as (fig, ax):
        ax.scatter(
            model.W_a[0].detach().numpy().ravel(),
            model.W_b[1].detach().numpy().ravel(),
            alpha=0.2,
        )

        ax.set_xlabel("$W_a$")
        ax.set_ylabel("$W_b$")

fig.savefig(osp.join(fig_path, "mmill_weight_asymmetry.pdf"))

# %%

all_arch = ["many_10_5", "many_25_5"]
all_datasets = ["mnist", "mmill"]

rho_values = "0.5_0.05"
models = {}
for dataset in all_datasets:
    for arch in all_arch:
        context = f"{dataset}_biopcn_{arch}_rho{rho_values}"

        name = osp.join("simulations", context, "checkpoints_100.pkl")
        with open(name, "rb") as f:
            checkpoints = pickle.load(f)

        models[dataset, arch] = checkpoints[-1]

# %%

arch_list = all_arch
with plt.style.context(paper_style):
    with dv.FigureManager(
        1, 2, figsize=(3.5, 1.5), despine_kws={"offset": 2}, constrained_layout=True
    ) as (fig, axs):
        for i, ax in enumerate(axs):
            model = models["mmill", arch_list[i]]

            ax.scatter(
                model.W_a[0].detach().numpy().ravel(),
                model.W_b[1].detach().numpy().ravel(),
                alpha=0.2,
            )

            ax.set_xlabel("$W_a$")
            ax.set_ylabel("$W_b$")


# %%

arch_list = all_arch
with plt.style.context(paper_style):
    with dv.FigureManager(
        2, 2, figsize=(5.5, 4.0), despine_kws={"offset": 2}, constrained_layout=True
    ) as (fig, axs):
        for k, ax_row in enumerate(axs):
            for i, ax in enumerate(ax_row):
                model = models[all_datasets[k], arch_list[i]]

                ax.scatter(
                    model.W_a[0].detach().numpy().ravel(),
                    model.W_b[1].detach().numpy().ravel(),
                    alpha=0.2,
                )

                old_xlim = ax.get_xlim()
                old_ylim = ax.get_ylim()
                crt_lim = max(
                    np.max(np.abs(ax.get_xlim())), np.max(np.abs(ax.get_ylim()))
                )
                ax.plot(
                    [-crt_lim, crt_lim],
                    [-crt_lim, crt_lim],
                    ls="--",
                    c="C3",
                    lw=1,
                    zorder=-1,
                )

                ax.set_xlim(old_xlim)
                ax.set_ylim(old_ylim)

                ax.set_xlabel("$W_a$")
                ax.set_ylabel("$W_b$")

                ax.set_title(f"{all_datasets[k]}, {arch_list[i]}")

fig.savefig(osp.join(fig_path, "combined_weight_asymmetry.pdf"))

# %%

arch_list = all_arch
with plt.style.context(paper_style):
    with dv.FigureManager(
        1, 2, figsize=(5.5, 2.0), despine_kws={"offset": 2}, constrained_layout=True
    ) as (fig, axs):
        for i, ax in enumerate(axs):
            model = models["mnist", arch_list[i]]

            ax.scatter(
                model.W_a[0].detach().numpy().ravel(),
                model.W_b[1].detach().numpy().ravel(),
                alpha=0.2,
            )

            old_xlim = ax.get_xlim()
            old_ylim = ax.get_ylim()
            crt_lim = max(np.max(np.abs(ax.get_xlim())), np.max(np.abs(ax.get_ylim())))
            ax.plot(
                [-crt_lim, crt_lim],
                [-crt_lim, crt_lim],
                ls="--",
                c="C3",
                lw=1,
                zorder=-1,
            )

            ax.set_xlim(old_xlim)
            ax.set_ylim(old_ylim)

            ax.set_xlabel("$W_a$")
            ax.set_ylabel("$W_b$")

            ax.set_title(f"{arch_list[i]}")

fig.savefig(osp.join(fig_path, "mnist_weight_asymmetry_multi.pdf"))

# %%
