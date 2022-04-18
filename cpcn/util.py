"""Define some utilities."""

import torch


def make_onehot(y) -> torch.Tensor:
    y_oh = torch.FloatTensor(y.shape[0], y.max().item() + 1)
    y_oh.zero_()
    y_oh.scatter_(1, y.reshape(-1, 1), 1)

    return y_oh


def evaluate(net, loader) -> tuple:
    """Evaluate PCN or CPCN network on a test set.
    
    :param net: network whose performance to evaluate; should have `loss()` member
    :param loader: test set loader
    :return: tuple (PC_loss, accuracy), where `PC_loss` is the predictive-coding loss,
        as returned by `net.loss()` after a run of `net.forward_constrained()`; the
        accuracy is calculated based on the output from `net.forward()`
    """
    n = 0
    n_correct = 0
    loss = 0
    for x, y in loader:
        y_pred = net.forward(x)
        net.forward_constrained(x, y)
        loss += net.loss().item()

        idx_pred = y_pred.argmax(dim=1)
        n_correct += y[range(len(y)), idx_pred].sum()
        n += len(y)

    avg_loss = loss / n
    frac_correct = n_correct / n
    return avg_loss, frac_correct
