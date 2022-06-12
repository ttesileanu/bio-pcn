import pytest

import torch

from cpcn.util import pretty_size, make_onehot


def test_make_onehot_shape():
    y = torch.LongTensor([1, 3, 2])
    y_oh = make_onehot(y)

    assert y_oh.shape == (len(y), 4)


def test_make_onehot_output():
    y = torch.LongTensor([1, 3, 2])
    exp_y_oh = torch.FloatTensor([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    y_oh = make_onehot(y)

    assert torch.allclose(y_oh, exp_y_oh)


def test_pretty_size_small():
    assert pretty_size(23) == "23.0B"


def test_pretty_size_kb():
    assert pretty_size(1024) == "1.0KB"


def test_pretty_size_multiplier():
    assert pretty_size(16384, multiplier=1000) == "16.4KB"


def test_pretty_size_large():
    assert pretty_size(1_234_567_890_123_245, multiplier=1000) == "1234.6TB"
