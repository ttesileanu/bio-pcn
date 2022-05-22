from types import SimpleNamespace
import pytest

import torch
import numpy as np

from cpcn.train import Trainer
from unittest.mock import Mock


def generate_loader():
    torch.manual_seed(321)
    n_in = 3
    n_out = 2
    batch_size = 4
    n_batches = 5

    data = [
        (torch.randn((batch_size, n_in)), torch.randn((batch_size, n_out)))
        for _ in range(n_batches)
    ]
    return data


@pytest.fixture
def loader():
    return generate_loader()


@pytest.fixture
def trainer():
    trainer = Trainer(generate_loader())
    return trainer


def test_trainer_iterates_have_x_and_y_members(trainer):
    for batch in trainer(3):
        assert hasattr(batch, "x")
        assert hasattr(batch, "y")


def test_iterating_through_trainer_yields_correct_x_and_y(loader):
    trainer = Trainer(loader)
    for batch, (x, y) in zip(trainer(len(loader)), loader):
        assert torch.allclose(batch.x, x)
        assert torch.allclose(batch.y, y)


def test_trainer_call_returns_sequence_with_correct_len(trainer):
    n = 2
    assert len(trainer(n)) == n


def test_trainer_call_returns_sequence_with_right_no_of_elems(trainer):
    n = 5
    data = [_ for _ in trainer(n)]

    assert len(data) == n


def test_trainer_len_equals_length_of_loader(loader):
    trainer = Trainer(loader)
    assert len(trainer) == len(loader)


def test_trainer_repeats_dataset_if_necessary(loader):
    n = 2 * len(loader)

    trainer = Trainer(loader)
    data = [_ for _ in trainer(n)]
    expected = 2 * [_ for _ in loader]

    assert len(data) == len(expected)
    for batch, (x, y) in zip(data, expected):
        assert torch.allclose(batch.x, x)
        assert torch.allclose(batch.y, y)


def test_batch_feed_calls_net_relax(trainer):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    for batch in trainer(1):
        batch.feed(net)

    net.relax.assert_called_once()


def test_batch_feed_calls_net_relax_with_x_and_y(trainer):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    for batch in trainer(1):
        batch.feed(net)

    net.relax.assert_called_once_with(batch.x, batch.y)


def test_batch_feed_sends_other_kwargs_to_net_relax(trainer):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    foo = 3.5
    for batch in trainer(1):
        batch.feed(net, foo=foo)

    net.relax.assert_called_once_with(batch.x, batch.y, foo=foo)


def test_batch_feed_return_contains_results_from_relax_in_fast(trainer):
    net = Mock()
    ret_val = "test"
    net.relax.return_value = ret_val
    for batch in trainer(1):
        ns = batch.feed(net)

    assert ns.fast == ret_val


def test_batch_feed_calls_calculate_weight_grad_with_fast(trainer):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    for batch in trainer(1):
        ns = batch.feed(net)

    net.calculate_weight_grad.assert_called_once_with(ns.fast)


def test_batch_contains_batch_index(trainer):
    n = 10
    for i, batch in enumerate(trainer(n)):
        assert i == batch.idx


def test_batch_every(trainer):
    n = 12
    s = 3
    for batch in trainer(n):
        assert batch.every(s) == ((batch.idx % s) == 0)


def test_batch_count(trainer):
    n = 25
    c = 13
    idxs = np.linspace(0, n - 1, c).astype(int)
    for batch in trainer(n):
        assert batch.count(c) == (batch.idx in idxs)
