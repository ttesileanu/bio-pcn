"""Define functions for plotting."""

import pydove as dv
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import torch

from types import SimpleNamespace
from typing import Optional, Tuple


def show_constraint_diagnostics(
    diagnostics: dict, n_mat: int = 4, rho: float = 1.0
) -> plt.Figure:
    """Make a figure diagnosing the whitening constraint.

    Uses the diagnostics dictionary returned from `util.get_constraint_diagnostics`.

    NB: This currently only works for a single hidden layer.

    :param diagnostics: dictionary of diagnostic measures
    :param n_mat: number of covariance matrices to display
    :param rho: limit on eigenvalues of covariance matrices
    :return: the figure that was created
    """
    fig = plt.figure(figsize=(7, 7), constrained_layout=True)

    # ensure the number of boxes is even
    n_mat = 2 * ((n_mat + 1) // 2)
    gs = mpl.gridspec.GridSpec(3, n_mat, figure=fig)

    mat_axs = [fig.add_subplot(gs[0, i]) for i in range(n_mat)]
    trace_ax = fig.add_subplot(gs[1:, : n_mat // 2])
    eval_ax = fig.add_subplot(gs[1:, n_mat // 2 :])

    # draw the matrices
    n_cov = len(diagnostics["batch"])
    sel_cov = torch.linspace(0, n_cov - 1, 5).int()

    for ax, i in zip(mat_axs, sel_cov):
        crt_cov = diagnostics["cov:1"][i]
        crt_scale = torch.max(torch.abs(crt_cov))
        ax.imshow(crt_cov, vmin=-crt_scale, vmax=crt_scale, cmap="RdBu_r")
        ax.set_title(f"batch {diagnostics['batch'][i]}")

    # draw the evolution of the trace of the constraint
    trace_ax.axhline(0, c="k", ls="--")
    trace_ax.plot(diagnostics["batch"], diagnostics["trace:1"])
    trace_ax.annotate(
        f"{diagnostics['trace:1'][-1]:.2g}",
        (diagnostics["batch"][-1], diagnostics["trace:1"][-1]),
        xytext=(3, 0),
        textcoords="offset points",
        va="center",
        c="C0",
        fontweight="bold",
    )
    trace_ax.set_xlabel("batch")
    trace_ax.set_ylabel("trace of constraint")

    # draw the evolution of the eigenvalues
    eval_ax.axhline(rho, c="k", ls="--")
    eval_ax.plot(diagnostics["batch"], diagnostics["evals:1"], alpha=0.5, lw=1)
    eval_ax.plot(diagnostics["batch"], diagnostics["max_eval:1"], c="k", lw=2)
    eval_ax.annotate(
        f"{diagnostics['max_eval:1'][-1]:.2g}",
        (diagnostics["batch"][-1], diagnostics["max_eval:1"][-1]),
        xytext=(3, 0),
        textcoords="offset points",
        va="center",
        c="k",
        fontweight="bold",
    )
    eval_ax.set_xlabel("batch")
    eval_ax.set_ylabel("maximum $z$-cov eval")
    eval_ax.set_ylim(0, 5 * rho)

    for ax in [trace_ax, eval_ax]:
        sns.despine(ax=ax, offset=10)
        ax.set_xlim(0, None)

    return fig


def show_learning_curves(
    results: SimpleNamespace,
    axs: Optional[Tuple[plt.Axes]] = None,
    show_train: bool = True,
    show_val: bool = True,
    annotate_val: bool = True,
    labels: Tuple[str] = ("train", "val"),
    colors: Tuple[str] = ("C0", "C1"),
    plot_kwargs: Optional[dict] = None,
) -> plt.Figure:
    """Make plot of predictive-coding loss and classification error rate.
    
    :param results: namespace of results, containing dictionaries `"train"` and
        `"validation"`
    :param axs: tuple of axes in which to make the plots, `(ax_loss, ax_acc)`
    :param show_train: whether to plot the training results
    :param show_val: whether to plot the validation results
    :param annotate_final: whether to annotate the final value
    :param labels: how to label the curves
    :param colors: how to color the curves
    :param plot_kwargs: keyword arguments to pass to plot
    :return: the Matplotlib figure that is created
    """
    # handle defaults
    if axs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    else:
        ax1, ax2 = axs
        fig = ax1.get_figure()

    if plot_kwargs is None:
        plot_kwargs = {"alpha": 0.7, "lw": 1.0}
    else:
        plot_kwargs.setdefault("alpha", 0.7)
        if "linewidth" not in plot_kwargs and "lw" not in plot_kwargs:
            plot_kwargs.setdefault("lw", 1.0)

    if show_train:
        ax1.plot(
            results.train["batch"],
            results.train["pc_loss"],
            label=labels[0],
            c=colors[0],
            **plot_kwargs,
        )

    if show_val:
        ax1.plot(
            results.validation["batch"],
            results.validation["pc_loss"],
            label=labels[1],
            c=colors[1],
            **plot_kwargs,
        )
        if annotate_val:
            ax1.annotate(
                f"{results.validation['pc_loss'][-1]:.2g}",
                (results.validation["batch"][-1], results.validation["pc_loss"][-1]),
                xytext=(3, 0),
                textcoords="offset points",
                c=colors[1],
                va="center",
                fontweight="bold",
            )

    ax1.legend(frameon=False)
    ax1.set_yscale("log")
    ax1.set_xlabel("batch")
    ax1.set_ylabel("predictive-coding loss")

    val_error_rate = 100 * (1.0 - results.validation["accuracy"])

    if show_train:
        train_error_rate = 100 * (1.0 - results.train["accuracy"])
        ax2.plot(
            results.train["batch"],
            train_error_rate,
            label=labels[0],
            c=colors[0],
            **plot_kwargs,
        )

    if show_val:
        ax2.plot(
            results.validation["batch"],
            val_error_rate,
            label=labels[1],
            c=colors[1],
            **plot_kwargs,
        )
        if annotate_val:
            ax2.annotate(
                f"{val_error_rate[-1]:.1f}%",
                (results.validation["batch"][-1], val_error_rate[-1]),
                xytext=(3, 0),
                textcoords="offset points",
                c=colors[0],
                va="center",
                fontweight="bold",
            )

    ax2.legend(frameon=False)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("batch")
    ax2.set_ylabel("error rate (%)")

    for ax in [ax1, ax2]:
        sns.despine(ax=ax, offset=10)
        ax.set_xlim(0, None)

    return fig


def show_weight_evolution(x: torch.Tensor, weights: torch.Tensor, ax: plt.Axes):
    weights = weights.reshape(len(weights), -1)
    n_lines = weights.shape[1]
    alpha = max(min(50 / n_lines, 0.5), 0.01)
    ax.plot(x, weights, c="k", lw=0.5, alpha=alpha)

    ax.set_xlabel("batch")
    ax.set_ylabel("weight")


def show_latent_convergence(
    fast_results: dict, ax: Optional[plt.Axes] = None, var: str = "z"
):
    """Make plots showing how the latent variables converge.

    This draws only the last sample of each batch.
    
    :param fast_results: dictionary containing the evolution of the latent variable
        `"z"` (the name can be overridden using the `var` argument; see below)
    :param ax: axes in which to make the plot; default: current axes
    :param var: name of latent variable
    """
    if ax is None:
        ax = plt.gca()

    cmap = mpl.cm.winter

    # focus on the last sample of each batch
    max_sample = torch.max(fast_results["sample"]).item()
    crt_sel = fast_results["sample"] == max_sample

    batch = fast_results["batch"][crt_sel]
    z = fast_results["z:1"][crt_sel]

    n = len(batch)
    for i in range(n):
        color = cmap(int(cmap.N * (0.2 + 0.8 * i / n)))
        diff = z[i, :, :] - z[i, 0, :]
        diff = diff / torch.max(torch.abs(diff))
        ax.plot(diff, c=color, lw=0.5)

    sm = mpl.cm.ScalarMappable(
        cmap=cmap, norm=mpl.pyplot.Normalize(vmin=0, vmax=torch.max(batch))
    )
    sm.ax = ax
    cbar = dv.colorbar(sm)
    cbar.set_label("batch")

    ax.set_xlabel("fast dynamics iteration")
    ax.set_ylabel("latent $(z - z_0) / \\rm{max}| z - z_0|$")
