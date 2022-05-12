# %% [markdown]
# # Make figures for the linear case

import os
import os.path as osp

import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import torch
import numpy as np

from tqdm.notebook import tqdm

import pickle

from cpcn.graph import plot_with_error_interval

# %% [markdown]
# ## One hidden layer

contexts = {"pcn": "mnist_pcn_one", "biopcn": "mnist_biopcn_one", "wb": "mnist_wb_one"}
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
    for name in filenames:
        with open(name, "rb") as f:
            histories[net_type].append(pickle.load(f))

# %% [markdown]
# ### Learning curves

# %%

paper_style = [
    "seaborn-paper",
    {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6},
]

max_batch = 1000
# max_batch = 3000

with plt.style.context(paper_style):
    with dv.FigureManager(figsize=(3, 2), despine_kws={"offset": 5}) as (_, ax):
        for i, net_type in enumerate(histories):
            crt_data = [_.validation for _ in histories[net_type]]
            plot_with_error_interval(
                crt_data, mask=crt_data[0]["batch"] < max_batch, c=f"C{i}", lw=1
            )

        for i, net_type in enumerate(histories):
            ax.plot([], [], c=f"C{i}", label=net_type)

        ax.set_xlabel("batch")
        ax.set_ylabel("PC loss")
        ax.legend(frameon=False)

        ax.set_yscale("log")

# %% [markdown]
# ## Two hidden layers

contexts = {"pcn": "mnist_pcn_two", "biopcn": "mnist_biopcn_two", "wb": "mnist_wb_two"}
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
    for name in filenames:
        with open(name, "rb") as f:
            histories[net_type].append(pickle.load(f))

# %% [markdown]
# ### Learning curves

# %%

paper_style = [
    "seaborn-paper",
    {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6},
]

max_batch = 1000
# max_batch = 3000

with plt.style.context(paper_style):
    with dv.FigureManager(figsize=(3, 2), despine_kws={"offset": 5}) as (_, ax):
        for i, net_type in enumerate(histories):
            crt_data = [_.validation for _ in histories[net_type]]
            plot_with_error_interval(
                crt_data, mask=crt_data[0]["batch"] < max_batch, c=f"C{i}", lw=1
            )

        for i, net_type in enumerate(histories):
            ax.plot([], [], c=f"C{i}", label=net_type)

        ax.set_xlabel("batch")
        ax.set_ylabel("PC loss")
        ax.legend(frameon=False)

        ax.set_yscale("log")

# %% [markdown]
# ## Two large hidden layers

contexts = {
    "pcn": "mnist_pcn_large-two",
    "biopcn": "mnist_biopcn_large-two",
    "wb": "mnist_wb_large-two",
}
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
    for name in tqdm(filenames):
        with open(name, "rb") as f:
            crt_history = pickle.load(f)
            del crt_history.weight
            del crt_history.constraint
            histories[net_type].append(crt_history)

# %% [markdown]
# ### Learning curves

# %%

paper_style = [
    "seaborn-paper",
    {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6},
]

max_batch = 3000

with plt.style.context(paper_style):
    with dv.FigureManager(figsize=(3, 2), despine_kws={"offset": 5}) as (_, ax):
        for i, net_type in enumerate(histories):
            crt_data = [_.validation for _ in histories[net_type]]
            plot_with_error_interval(
                crt_data, mask=crt_data[0]["batch"] < max_batch, c=f"C{i}", lw=1
            )

        for i, net_type in enumerate(histories):
            ax.plot([], [], c=f"C{i}", label=net_type)

        ax.set_xlabel("batch")
        ax.set_ylabel("PC loss")
        ax.legend(frameon=False)

        ax.set_yscale("log")

# %%
