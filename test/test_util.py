from types import SimpleNamespace
import pytest

import torch

from cpcn.util import pretty_size, make_onehot, one_hot_accuracy, dot_accuracy


def test_make_onehot_shape():
    y = torch.LongTensor([1, 3, 2])
    y_oh = make_onehot(y)

    assert y_oh.shape == (len(y), 4)


def test_make_onehot_output():
    y = torch.LongTensor([1, 3, 2])
    exp_y_oh = torch.FloatTensor([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    y_oh = make_onehot(y)

    assert torch.allclose(y_oh, exp_y_oh)


def test_one_hot_accuracy():
    y = torch.FloatTensor([[0.2, -0.1, 1], [1, 0.3, 0.1], [-0.1, 1, 0.1]])
    y_pred = torch.FloatTensor([[0.3, -0.2, 0.5], [0.8, 0.5, 0.1], [-0.3, -0.4, 0.2]])

    expected_accuracy = (y[0, 2] + y[1, 0] + y[2, 2]) / 3

    ns = SimpleNamespace(y=y, fast=SimpleNamespace(y_pred=y_pred))
    accuracy = one_hot_accuracy(ns, None)

    assert pytest.approx(accuracy) == expected_accuracy


def test_dot_accuracy():
    y = torch.FloatTensor([[1, 0, 1], [0, -1, 0], [0.5, 0.5, 0.5]])
    y_pred = torch.FloatTensor([[0.3, -0.2, 0.5], [0.8, 0.5, 0.1], [-0.3, -0.4, 0.2]])

    expected_accuracy = 0
    for crt_y, crt_y_pred in zip(y, y_pred):
        crt_y = crt_y / torch.linalg.norm(crt_y)
        crt_y_pred = crt_y_pred / torch.linalg.norm(crt_y_pred)
        expected_accuracy += 0.5 * (1 + torch.dot(crt_y, crt_y_pred))

    expected_accuracy /= len(y)
    ns = SimpleNamespace(y=y, fast=SimpleNamespace(y_pred=y_pred))
    accuracy = dot_accuracy(ns, None)

    assert pytest.approx(accuracy) == expected_accuracy


def test_pretty_size_small():
    assert pretty_size(23) == "23.0B"


def test_pretty_size_kb():
    assert pretty_size(1024) == "1.0KB"


def test_pretty_size_multiplier():
    assert pretty_size(16384, multiplier=1000) == "16.4KB"


def test_pretty_size_large():
    assert pretty_size(1_234_567_890_123_245, multiplier=1000) == "1234.6TB"
