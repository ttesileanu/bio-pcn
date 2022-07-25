# %% [markdown]
# # Make figures for the dependence on constraint scale

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
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.labelpad": 1,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
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

# %% [markdown]
# ## Load data

# %%

# all_arch = ["one", "two", "large-two", "large_small"]
all_algo = ["wb", "pcn", "biopcn"]
all_rho = {
    "pcn": [0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0],
    "biopcn": [0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0],
    "wb": [1.0],
}
all_histories = {}

arch = "large"
# unfortunately the naming convention for the hyperparam optimization wasn't
# very well chosen...
convert = lambda x: f"{x:.1f}" if np.abs(x - np.round(x)) < 1e-8 else f"{x:g}"
for algo in tqdm(all_algo, desc="algo"):
    contexts = {}
    base = f"mnist_{algo}_{arch}"
    for crt_rho in all_rho[algo]:
        value = base
        if algo != "wb":
            value += f"_rho{convert(crt_rho)}"
            if arch == "large":
                value += f"_{convert(crt_rho / 10)}"
        contexts[crt_rho] = value
    histories = {_: [] for _ in contexts}
    for crt_rho, context in contexts.items():
        path = osp.join("simulations", context)
        filenames = [
            osp.join(path, f)
            for f in os.listdir(path)
            if f.startswith("history_")
            and f.endswith(".pkl")
            and osp.isfile(os.path.join(path, f))
        ]
        for name in tqdm(filenames, desc=f"{algo}_rho{crt_rho:.1f}"):
            with open(name, "rb") as f:
                crt_history = pickle.load(f)
                del crt_history.weight
                del crt_history.constraint

                histories[crt_rho].append(crt_history)

    all_histories[algo] = histories

# %%

context = "mnist_wb_large"
name = osp.join("simulations", context, "history_700.pkl")
with open(name, "rb") as f:
    long_wb_run = pickle.load(f)
    del long_wb_run.weight
    del long_wb_run.constraint

# %% [markdown]
# ## Make the plot


def lighten(color, amount: float) -> tuple:
    color = mpl.colors.to_rgba(color)
    light_color = tuple((1 - amount) * _ + amount for _ in color[:3]) + (color[-1],)
    return light_color


min_batch = 1
max_batch = 2000
with dv.FigureManager() as (_, ax):
    crt_data = [_.validation for _ in all_histories["wb"][1.0]]
    crt_min_mask = crt_data[0]["batch"] >= min_batch
    crt_max_mask = crt_data[0]["batch"] < max_batch
    crt_mask = crt_min_mask & crt_max_mask
    plot_with_error_interval(
        crt_data,
        mask=crt_mask,
        c=color_map["wb"],
        lw=1,
        x_var="sample",
        y_var="pc_loss",
        fill_kwargs={"alpha": 0.2},
        **style_map["wb"],
    )

    for i, rho in enumerate(all_rho["biopcn"]):
        crt_data = [_.validation for _ in all_histories["biopcn"][rho]]
        crt_min_mask = crt_data[0]["batch"] >= min_batch
        crt_max_mask = crt_data[0]["batch"] < max_batch
        crt_mask = crt_min_mask & crt_max_mask
        # plot_with_error_interval(
        #     crt_data,
        #     mask=crt_mask,
        #     c=lighten(color_map["biopcn"], 0.7 * i / len(all_rho["biopcn"])),
        #     lw=1,
        #     x_var="sample",
        #     y_var="pc_loss",
        #     fill_kwargs={"alpha": 0.2},
        #     **style_map["biopcn"],
        # )
        crt_median = np.median(
            np.asarray([_["pc_loss"][crt_mask] for _ in crt_data]), axis=0
        )
        ax.plot(
            crt_data[0]["sample"][crt_mask],
            crt_median,
            lw=1,
            c=lighten(color_map["biopcn"], 1.0 * i / len(all_rho["biopcn"])),
            **style_map["biopcn"],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim(0.25, 5)

# %%

ci_level = 0.95
with plt.style.context(paper_style):
    with dv.FigureManager(
        figsize=(2.5, 1.75), despine_kws={"offset": 2}, constrained_layout=True
    ) as (fig, ax):
        biopcn_low = []
        biopcn_mid = []
        biopcn_high = []

        for crt_rho in all_rho["biopcn"]:
            crt_history = all_histories["biopcn"][crt_rho]
            crt_finals = [_.validation["pc_loss"][-1] for _ in crt_history]
            crt_summaries = np.quantile(
                crt_finals, [(1 - ci_level) / 2, 0.5, (1 + ci_level) / 2]
            )
            biopcn_low.append(crt_summaries[0])
            biopcn_mid.append(crt_summaries[1])
            biopcn_high.append(crt_summaries[2])

        biopcn_low = np.asarray(biopcn_low)
        biopcn_mid = np.asarray(biopcn_mid)
        biopcn_high = np.asarray(biopcn_high)

        # ax.scatter(all_rho["biopcn"], biopcn_mid)
        ax.errorbar(
            all_rho["biopcn"],
            biopcn_mid,
            (biopcn_mid - biopcn_low, biopcn_high - biopcn_mid),
            ls="none",
            marker=".",
        )

        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("final PC loss")

        ax.set_xscale("log")
        ax.set_yscale("log")

        crt_finals = [
            crt_history.validation["pc_loss"][-1]
            for crt_history in all_histories["wb"][1.0]
        ]
        crt_summaries = np.quantile(crt_finals, [0.2, 0.5, 0.8])
        ax.fill_between(
            ax.get_xlim(),
            crt_summaries[0],
            crt_summaries[2],
            color=color_map["wb"],
            alpha=0.2,
            edgecolor="none",
        )
        ax.axhline(crt_summaries[1], c=color_map["wb"], **style_map["wb"])

        # crt_lowest = long_wb_run.validation["pc_loss"][-1]
        # h = ax.axhline(crt_lowest, c="k", label="PC (long run)")

        legend_elements = []
        for net_type in ["biopcn", "wb"]:
            legend_elements.append(
                plt.Line2D(
                    [1],
                    [1],
                    color=color_map[net_type],
                    label=label_map[net_type],
                    **style_map[net_type],
                )
            )

        # legend_elements.append(h)
        ax.legend(handles=legend_elements, frameon=False, loc="upper center")

        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
        ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))

fig.savefig(osp.join(fig_path, f"mnist_{arch}_rho_dep_pc_loss.pdf"))

# %%

ci_level = 0.95
with plt.style.context(paper_style):
    with dv.FigureManager(
        figsize=(2.5, 1.75), despine_kws={"offset": 2}, constrained_layout=True
    ) as (fig, ax):
        biopcn_low = []
        biopcn_mid = []
        biopcn_high = []

        for crt_rho in all_rho["biopcn"]:
            crt_history = all_histories["biopcn"][crt_rho]
            crt_finals = [_.validation["prediction_error"][-1] for _ in crt_history]
            crt_summaries = np.quantile(
                crt_finals, [(1 - ci_level) / 2, 0.5, (1 + ci_level) / 2]
            )
            biopcn_low.append(crt_summaries[0])
            biopcn_mid.append(crt_summaries[1])
            biopcn_high.append(crt_summaries[2])

        biopcn_low = np.asarray(biopcn_low)
        biopcn_mid = np.asarray(biopcn_mid)
        biopcn_high = np.asarray(biopcn_high)

        # ax.scatter(all_rho["biopcn"], biopcn_mid)
        ax.errorbar(
            all_rho["biopcn"],
            biopcn_mid,
            (biopcn_mid - biopcn_low, biopcn_high - biopcn_mid),
            ls="none",
            marker=".",
        )

        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("final prediction error")

        ax.set_xscale("log")
        ax.set_yscale("log")

        crt_finals = [
            crt_history.validation["prediction_error"][-1]
            for crt_history in all_histories["wb"][1.0]
        ]
        crt_summaries = np.quantile(crt_finals, [0.2, 0.5, 0.8])
        ax.fill_between(
            ax.get_xlim(),
            crt_summaries[0],
            crt_summaries[2],
            color=color_map["wb"],
            alpha=0.2,
            edgecolor="none",
        )
        ax.axhline(crt_summaries[1], c=color_map["wb"], **style_map["wb"])

        legend_elements = []
        for net_type in ["biopcn", "wb"]:
            legend_elements.append(
                plt.Line2D(
                    [1],
                    [1],
                    color=color_map[net_type],
                    label=label_map[net_type],
                    **style_map[net_type],
                )
            )

        crt_lowest = long_wb_run.validation["prediction_error"][-1]
        h = ax.axhline(crt_lowest, c="k", label="PC (long run)")

        legend_elements.append(h)
        ax.legend(handles=legend_elements, frameon=False, loc="upper center")

        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y}"))
        ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y}"))

fig.savefig(osp.join(fig_path, f"mnist_{arch}_rho_dep_pred_err.pdf"))

# %%
