# %% [markdown]
# # Look at Whittington&Bogacz runs to see stats of hidden variables

import os.path as osp

import matplotlib.pyplot as plt
import pydove as dv

import torch
import numpy as np

import pickle

base_path = osp.join("simulations")

# %% [markdown]
# ## Look at one layer

# %%

path = osp.join(base_path, "mnist_wb_one")
filename = osp.join(path, "history_700.pkl")
with open(filename, "rb") as f:
    history = pickle.load(f)

# %%
