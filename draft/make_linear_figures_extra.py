# %% [markdown]
# # Make figures for the linear case, additional datasets

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
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}")
            )

    return fig, ax


# %% [markdown]
# ## Load data

# %%

all_datasets = ["fashionmnist", "cifar10", "cifar100", "lfw"]
all_algo = ["wb", "pcn", "biopcn"]
all_histories = {}

all_arch = {
    "fashionmnist": "many_30_5",
    "cifar10": "many_30_5",
    "cifar100": "many_30_5",
    "lfw": "many_20_5_20",
}
for dataset in tqdm(all_datasets, desc="dataset"):
    arch = all_arch[dataset]
    contexts = {}
    for algo in all_algo:
        value = f"{dataset}_{algo}_{arch}"
        contexts[algo] = value

    histories = {_: [] for _ in contexts}
    base_files = os.listdir("simulations")
    for net_type, context in contexts.items():
        # first find the full file name
        full_context_lst = [name for name in base_files if name.startswith(context)]
        assert len(full_context_lst) == 1

        full_context = full_context_lst[0]

        path = osp.join("simulations", full_context)
        filenames = [
            osp.join(path, f)
            for f in os.listdir(path)
            if f.startswith("history_")
            and f.endswith(".pkl")
            and osp.isfile(os.path.join(path, f))
        ]
        for name in tqdm(filenames, desc=f"{dataset}_{net_type}"):
            with open(name, "rb") as f:
                crt_history = pickle.load(f)
                del crt_history.weight
                del crt_history.constraint
                histories[net_type].append(crt_history)

    all_histories[dataset] = histories

# %% [markdown]
# ## Make all the plots

# %%

max_batches = {_: 3000 for _ in all_histories}
# figsize = None
# figsize = (2.75, 2.00)
figsize = (1.3, 1.0)
for dataset, histories in all_histories.items():
    extra_args = {} if figsize is None else {"figsize": figsize}
    fig, ax = make_plot(histories, max_batch=max_batches[dataset], **extra_args)
    if dataset == "lfw":
        ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    # ax.set_title(dataset)
    if figsize is None:
        figsize_str = ""
    else:
        figsize_str = "_" + "_".join(str(_) for _ in figsize)
    fig.savefig(osp.join(fig_path, f"extra_{dataset}_pc_loss{figsize_str}.pdf"))

# %%

for dataset, histories in all_histories.items():
    if "prediction_error" not in histories["wb"][0].validation:
        continue
    fig, ax = make_plot(
        histories, max_batch=max_batches[dataset], y_var="prediction_error",
    )
    ax.set_ylabel("mean squared error")
    # fig.savefig(osp.join(fig_path, f"linear_{arch}_pred_err.png"), dpi=600)
    fig.savefig(osp.join(fig_path, f"extra_{dataset}_pred_err.pdf"))


# %%

name = osp.join(
    "simulations", "fashionmnist_biopcn_many_30_5_rho0.03_0.03", "history_700.pkl"
)
with open(name, "rb") as f:
    crt_history = pickle.load(f)
    crt_cons_diag = crt_history.constraint

crt_rho = [0.03, 0.03]
_ = show_constraint_diagnostics(crt_cons_diag, layer=1, rho=crt_rho[0])
_ = show_constraint_diagnostics(crt_cons_diag, layer=2, rho=crt_rho[1])


# %%

name = osp.join(
    "simulations", "cifar10_biopcn_many_30_5_rho0.015_0.015", "history_700.pkl"
)
with open(name, "rb") as f:
    crt_history = pickle.load(f)
    crt_cons_diag = crt_history.constraint

crt_rho = [0.015, 0.015]
_ = show_constraint_diagnostics(crt_cons_diag, layer=1, rho=crt_rho[0])
_ = show_constraint_diagnostics(crt_cons_diag, layer=2, rho=crt_rho[1])

# %%

name = osp.join(
    "simulations", "cifar100_biopcn_many_30_5_rho0.0005_0.0005", "history_700.pkl"
)
with open(name, "rb") as f:
    crt_history = pickle.load(f)
    crt_cons_diag = crt_history.constraint

crt_rho = [0.0005, 0.0005]
_ = show_constraint_diagnostics(crt_cons_diag, layer=1, rho=crt_rho[0])
_ = show_constraint_diagnostics(crt_cons_diag, layer=2, rho=crt_rho[1])


# %%

name = osp.join("simulations", "lfw_biopcn_many_20_5_20_rho0.015", "history_700.pkl")
with open(name, "rb") as f:
    crt_history = pickle.load(f)
    crt_cons_diag = crt_history.constraint

crt_rho = [0.015, 0.015]
_ = show_constraint_diagnostics(crt_cons_diag, layer=1, rho=crt_rho[0])
_ = show_constraint_diagnostics(crt_cons_diag, layer=2, rho=crt_rho[1])

# %%
