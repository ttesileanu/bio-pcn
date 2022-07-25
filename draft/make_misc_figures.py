# %% [markdown]
# # Make some miscellaneous figures

import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from cpcn import load_mnist

fig_path = "figs"

# %%

mpl.rcParams["axes.linewidth"] = 0.5

# %% [markdown]
# ## A digit

# %%

mnist = load_mnist(return_loaders=False)

# %%

fig, ax = plt.subplots(figsize=(2, 2))
ax.imshow(mnist["train"][0][0].reshape(28, 28), cmap="gray_r")
ax.set_xticks([])
ax.set_yticks([])

fig.savefig(osp.join(fig_path, "mnist_digit.pdf"))

# %%
