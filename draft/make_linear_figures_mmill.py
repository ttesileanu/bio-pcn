# %% [markdown]
# # Make figures for the linear case, Mediamill dataset

import os
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import numpy as np

from tqdm.notebook import tqdm

from typing import Tuple

import pickle

from cpcn.graph import plot_with_error_interval, show_constraint_diagnostics

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

# %% [markdown]
# ### Learning curves

# %%


def make_plot(
    histories: dict,
    max_batch: int,
    min_batch: int = 1,
    figsize: tuple = (1.75, 1.25),
    y_var: str = "pc_loss",
) -> Tuple[plt.Figure, plt.Axes]:
    with plt.style.context(paper_style):
        with dv.FigureManager(
            figsize=figsize, despine_kws={"offset": 2}, constrained_layout=True
        ) as (fig, ax):
            for net_type in histories:
                crt_data = [_.validation for _ in histories[net_type]]
                crt_min_mask = crt_data[0]["batch"] >= min_batch
                crt_max_mask = crt_data[0]["batch"] < max_batch
                crt_mask = crt_min_mask & crt_max_mask
                plot_with_error_interval(
                    crt_data,
                    mask=crt_mask,
                    c=color_map[net_type],
                    lw=1,
                    x_var="sample",
                    y_var=y_var,
                    fill_kwargs={"alpha": 0.2},
                    **style_map[net_type],
                )

            legend_elements = []
            for net_type in histories:
                legend_elements.append(
                    plt.Line2D(
                        [1],
                        [1],
                        color=color_map[net_type],
                        label=label_map[net_type],
                        **style_map[net_type],
                    )
                )
                # ax.plot(
                #     [],
                #     [],
                #     c=color_map[net_type],
                #     label=label_map[net_type],
                #     **style_map[net_type],
                # )
            ax.legend(handles=legend_elements, frameon=False)

            ax.set_xlabel("iteration")
            if y_var == "pc_loss":
                ax.set_ylabel("PC loss")
            else:
                ax.set_ylabel(y_var)

            ax.set_yscale("log")
            ax.set_xscale("log")

            # use more human-readable tick labels
            # ax.xaxis.set_major_formatter(
            #     mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}")
            # )
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y}"))

    return fig, ax


# %% [markdown]
# ## Load data

# %%

all_arch = ["small", "large"]
all_algo = ["wb", "pcn", "biopcn"]
all_histories = {}

# rho = 1.0
rho_mapping = {"small": 0.2, "large": 1.0}

# unfortunately the naming convention for the hyperparam optimization wasn't
# very well chosen...
convert = lambda x: f"{x:.1f}" if np.abs(x - np.round(x)) < 1e-8 else f"{x:g}"
for arch in tqdm(all_arch, desc="arch"):
    rho = rho_mapping[arch]
    contexts = {}
    for algo in all_algo:
        value = f"mmill_{algo}_{arch}"
        if algo != "wb":
            value += f"_rho{convert(rho)}"
            if arch == "large":
                value += f"_{convert(rho / 10)}"
        contexts[algo] = value
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

max_batches = {"small": 3000, "large": 3000}
for arch, histories in all_histories.items():
    fig, ax = make_plot(histories, max_batch=max_batches[arch])
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    fig.savefig(osp.join(fig_path, f"mmill_linear_{arch}_pc_loss.pdf"))

# %%

for arch, histories in all_histories.items():
    if "prediction_error" not in histories["wb"][0].validation:
        continue
    fig, ax = make_plot(
        histories, max_batch=max_batches[arch], y_var="prediction_error"
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.set_ylabel("MSE")
    fig.savefig(osp.join(fig_path, f"mmill_linear_{arch}_pred_err.pdf"))

# %%

rho = rho_mapping["large"]
name = osp.join(
    "simulations",
    f"mmill_biopcn_large_rho{convert(rho)}_{convert(rho / 10)}",
    "history_700.pkl",
)
with open(name, "rb") as f:
    crt_history = pickle.load(f)
    crt_cons_diag = crt_history.constraint

crt_rho = [rho, rho / 10]
_ = show_constraint_diagnostics(crt_cons_diag, layer=1, rho=crt_rho[0])
_ = show_constraint_diagnostics(crt_cons_diag, layer=2, rho=crt_rho[1])

# %%


def rolling_average(y: np.ndarray, n: int) -> np.ndarray:
    if y.ndim < 2:
        y = y[:, None]
        was_unsqueezed = True
    else:
        was_unsqueezed = False

    pad_front = np.zeros_like(y[0])
    pad_back = np.tile(y[-1], (n - 1, 1))
    y_padded = np.vstack((pad_front, y, pad_back))

    y_cumul = np.cumsum(y_padded, axis=0)
    y_roll = (y_cumul[n:] - y_cumul[:-n]) / n

    if was_unsqueezed:
        y_roll = y_roll[:, 0]

    return y_roll


# %%

batch_size = 100
roll_n = 5
for layer in range(len(crt_rho)):
    with plt.style.context(paper_style):
        with dv.FigureManager(
            figsize=(1.75, 1.25), despine_kws={"offset": 2}, constrained_layout=True
        ) as (
            fig,
            ax,
        ):
            crt_x = crt_cons_diag["batch"] * batch_size

            l = layer + 1
            crt_y = rolling_average(crt_cons_diag[f"evals:{l}"], roll_n)
            ax.axhline(crt_rho[layer], c="k", ls="--", lw=1.0, zorder=-1)
            ax.plot(crt_x, crt_y, c=color_map["biopcn"], lw=0.5, alpha=0.7, zorder=1)
            ax.set_xlabel("iteration")

            cov_str = f"${{\\bb E}}\\left[z^{{({l})}} z^{{({l})\\top}}\\right]$"
            # ax.set_ylabel(f"evals of {cov_str}")
            ax.set_ylabel("covariance evals")

            ax.set_xscale("log")

            # ax.set_ylim(max(0, ax.get_ylim()[0]), None)
            ax.set_ylim(0, 3 * crt_rho[layer])

            iax = fig.add_axes([0.65, 0.60, 0.25, 0.30])
            crt_cov = crt_cons_diag[f"cov:{l}"][-1]
            crt_lim = np.max(np.abs(crt_cov))
            h = iax.imshow(crt_cov, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu_r")
            iax.set_xticks([])
            iax.set_yticks([-0.5, len(crt_cov) - 0.5])
            iax.set_yticklabels(["0", str(len(crt_cov))])
            iax.tick_params(axis="both", which="both", length=0)
            iax.set_xlabel(cov_str, fontsize=6, labelpad=2)
            iax.xaxis.set_label_position("top")

            cb = dv.colorbar(h)
            cb.ax.tick_params(length=0, pad=1)
            crt_quant_lim = int(10 * crt_lim) / 10
            cb.set_ticks([-crt_quant_lim, 0, crt_quant_lim])

        fig.savefig(osp.join(fig_path, f"mmill_linear_large_constraint_{l}.pdf"))

# %%

