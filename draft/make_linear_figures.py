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

from cpcn.graph import plot_with_error_interval

# %%

paper_style = [
    "seaborn-paper",
    {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6},
]

label_map = {"wb": "PCN", "biopcn": "BioCPCN", "pcn": "PCN + cons"}
color_map = {"wb": "C3", "pcn": "C0", "biopcn": "C1"}
style_map = {"wb": {"ls": "--"}, "pcn": {}, "biopcn": {}}

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#555CA9",  # liberty
        "#9B4688",  # violet crayola
        "#CC8B8B",  # old rose
        "#353531",  # black olive
        "#000000",  # black
    ]
)

fig_path = "figs"

# %% [markdown]
# ### Learning curves

# %%


def make_plot(
    histories: dict, max_batch: int, batch_size: int, figsize: tuple = (2.75, 2),
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
            ax.set_ylabel("predictive-coding loss")
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

all_arch = ["one", "two", "large-two"]
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

batch_size = 100
max_batches = {"one": 1000, "two": 1000, "large-two": 3000}
for arch, histories in all_histories.items():
    fig, ax = make_plot(histories, batch_size=batch_size, max_batch=max_batches[arch])
    fig.savefig(osp.join(fig_path, f"linear_{arch}.png"), dpi=300)

# %%
