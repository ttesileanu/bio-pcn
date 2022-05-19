# %% [markdown]
# # Make figures for the linear case

import os
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import torch
import numpy as np

from tqdm.notebook import tqdm

from typing import Tuple

import pickle

from cpcn.graph import plot_with_error_interval, show_constraint_diagnostics

# %%

paper_style = [
    "seaborn-paper",
    {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6},
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

# %% [markdown]
# ### Learning curves

# %%


def make_plot(
    histories: dict,
    max_batch: int,
    batch_size: int,
    figsize: tuple = (2.75, 2),
    y_var: str = "pc_loss",
) -> Tuple[plt.Figure, plt.Axes]:
    with plt.style.context(paper_style):
        with dv.FigureManager(figsize=figsize, despine_kws={"offset": 5}) as (fig, ax):
            for net_type in histories:
                crt_data = [_.validation for _ in histories[net_type]]
                # XXX this changes it in place!
                for _ in crt_data:
                    _["sample"] = batch_size * _["batch"]
                plot_with_error_interval(
                    crt_data,
                    mask=crt_data[0]["batch"] < max_batch,
                    c=color_map[net_type],
                    lw=1,
                    x_var="sample",
                    y_var=y_var,
                    fill_kwargs={"alpha": 0.2},
                    **style_map[net_type],
                )

            for net_type in histories:
                ax.plot(
                    [],
                    [],
                    c=color_map[net_type],
                    label=label_map[net_type],
                    **style_map[net_type],
                )

            ax.set_xlabel("iteration")
            if y_var == "pc_loss":
                ax.set_ylabel("predictive-coding loss")
            else:
                ax.set_ylabel(y_var)
            ax.legend(frameon=False)

            ax.set_yscale("log")
            ax.set_xscale("log")

            # use more human-readable tick labels
            # ax.xaxis.set_major_formatter(
            #     mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}")
            # )
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}")
            )

    return fig, ax


# %% [markdown]
# ## Load data

# %%

# all_arch = ["one", "two", "large-two", "large_small"]
all_arch = ["one", "two", "large_small"]
all_algo = ["wb", "pcn", "biopcn"]
all_histories = {}

for arch in tqdm(all_arch, desc="arch"):
    contexts = {_: f"mnist_{_}_{arch}" for _ in all_algo}
    histories = {_: [] for _ in contexts}
    for net_type, context in contexts.items():
        path = osp.join("simulations", context)
        filenames = [
            osp.join(path, f)
            for f in os.listdir(path)
            if f.startswith("history_")
            and f.endswith(".pkl")
            and osp.isfile(os.path.join(path, f))
        ]
        for name in tqdm(filenames, desc=f"{net_type}_{arch}"):
            with open(name, "rb") as f:
                crt_history = pickle.load(f)
                del crt_history.weight
                del crt_history.constraint
                histories[net_type].append(crt_history)

    all_histories[arch] = histories

# %% [markdown]
# ## Make all the plots

# %%

batch_size = 100
max_batches = {"one": 1000, "two": 1000, "large-two": 3000, "large_small": 5000}
for arch, histories in all_histories.items():
    fig, ax = make_plot(histories, batch_size=batch_size, max_batch=max_batches[arch])
    # fig.savefig(osp.join(fig_path, f"linear_{arch}_pc_loss.png"), dpi=300)
    fig.savefig(osp.join(fig_path, f"linear_{arch}_pc_loss.svg"))

# %%

batch_size = 100
max_batches = {"one": 1000, "two": 1000, "large-two": 3000, "large_small": 5000}
for arch, histories in all_histories.items():
    if "prediction_error" not in histories["wb"][0].validation:
        continue
    fig, ax = make_plot(
        histories,
        batch_size=batch_size,
        max_batch=max_batches[arch],
        y_var="prediction_error",
    )
    # fig.savefig(osp.join(fig_path, f"linear_{arch}_pred_err.png"), dpi=300)
    fig.savefig(osp.join(fig_path, f"linear_{arch}_pred_err.pdf"))

# %%

name = osp.join("simulations", f"mnist_biopcn_large_small", "history_0.pkl")
with open(name, "rb") as f:
    crt_history = pickle.load(f)
    crt_cons_diag = crt_history.constraint

crt_rho = [1.0, 0.1]
_ = show_constraint_diagnostics(crt_cons_diag, layer=1, rho=crt_rho[0])
_ = show_constraint_diagnostics(crt_cons_diag, layer=2, rho=crt_rho[1])

# %%

for layer in range(len(crt_rho)):
    with plt.style.context(paper_style):
        with dv.FigureManager(
            figsize=(2.75, 2), despine_kws={"offset": 5}, tight_layout=False
        ) as (
            fig,
            ax,
        ):
            crt_x = (crt_cons_diag["batch"] * batch_size).numpy()

            l = layer + 1
            crt_y = crt_cons_diag[f"evals:{l}"].numpy()
            ax.axhline(crt_rho[layer], c="k", ls="--", zorder=-1)
            ax.plot(crt_x, crt_y, c=color_map["biopcn"], lw=0.5, alpha=0.7, zorder=1)
            ax.set_xlabel("iteration")

            cov_str = f"${{\\bb E}}\\left[z^{{({l})}} z^{{({l})\\top}}\\right]$"
            ax.set_ylabel(f"evals of {cov_str}")

            ax.set_xscale("log")

            iax = fig.add_axes([0.6, 0.57, 0.22, 0.30])
            crt_cov = crt_cons_diag[f"cov:{l}"][-1].numpy()
            crt_lim = np.max(np.abs(crt_cov))
            h = iax.imshow(crt_cov, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu_r")
            iax.set_xticks([])
            iax.set_yticks([-0.5, len(crt_cov) - 0.5])
            iax.set_yticklabels(["0", str(len(crt_cov))])
            iax.tick_params(axis="both", which="both", length=0)
            iax.set_xlabel(cov_str, fontsize=6, labelpad=1)

            cb = dv.colorbar(h)
            cb.ax.tick_params(length=0, pad=1)
            crt_quant_lim = int(10 * crt_lim) / 10
            cb.ax.set_yticks([-crt_quant_lim, 0, crt_quant_lim])

        fig.savefig(osp.join(fig_path, f"linear_large_small_constraint_{l}.pdf"))

# %%
