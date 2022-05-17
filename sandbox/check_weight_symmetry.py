# %% [markdown]
# # Check whether BioPCN weights are symmetric after learning

import os.path as osp

import matplotlib.pyplot as plt
import pydove as dv

import torch
import numpy as np

import pickle

# %% [markdown]
# ## Load a trained model

# %%

# path = osp.join("..", "draft", "simulations", "mnist_biopcn_large-two")
path = osp.join("..", "draft", "simulations", "mnist_biopcn_two")
filename = osp.join(path, "checkpoints_700.pkl")
with open(filename, "rb") as f:
    checkpoints = pickle.load(f)

# %%

model = checkpoints["model"][-1]
with dv.FigureManager() as (_, ax):
    crt_wa = model.W_a[0].detach()
    crt_wb = model.W_b[1].detach()
    assert crt_wa.shape == crt_wb.shape

    # ax.scatter(crt_wa.numpy().ravel(), crt_wb.numpy().ravel(), alpha=0.02)
    res = dv.regplot(
        crt_wa.numpy().ravel(),
        crt_wb.numpy().ravel(),
        # scatter_kws={"alpha": 0.01},
        ax=ax,
    )

    ax.set_xlabel("$W_a$")
    ax.set_ylabel("$W_b$")

# %%
