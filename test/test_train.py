from types import SimpleNamespace
import pytest

import torch
import numpy as np

from cpcn import train
from cpcn.train import Trainer
from cpcn.track import Tracker
from unittest.mock import Mock, patch


def generate_loader(
    n_in: int = 3, n_out: int = 2, batch_size: int = 4, n_batches: int = 5
):
    torch.manual_seed(321)
    data = [
        (torch.randn((batch_size, n_in)), torch.randn((batch_size, n_out)))
        for _ in range(n_batches)
    ]
    return data


@pytest.fixture
def loader():
    return generate_loader()


@pytest.fixture
def val_loader():
    return generate_loader(batch_size=10, n_batches=10)


@pytest.fixture
def val_loader_one():
    return generate_loader(batch_size=20, n_batches=1)


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


def test_batch_has_training_true(trainer):
    for batch in trainer(1):
        assert hasattr(batch, "training")
        assert batch.training


def test_iterate_evaluate_goes_through_val_dataset(trainer, val_loader):
    data = [_ for _ in trainer.evaluate(val_loader)]
    expected = [_ for _ in val_loader]
    for crt_batch, (crt_x, crt_y) in zip(data, expected):
        assert torch.allclose(crt_batch.x, crt_x)
        assert torch.allclose(crt_batch.y, crt_y)


def test_evaluate_batch_feed_calls_net_relax(trainer, val_loader_one):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    for batch in trainer.evaluate(val_loader_one):
        batch.feed(net)

    net.relax.assert_called_once()


def test_evaluate_batch_feed_calls_net_relax_with_x_and_y(trainer, val_loader_one):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    for batch in trainer.evaluate(val_loader_one):
        batch.feed(net)

    net.relax.assert_called_once_with(batch.x, batch.y)


def test_evaluate_batch_feed_sends_other_kwargs_to_net_relax(trainer, val_loader_one):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    foo = 3.5
    for batch in trainer.evaluate(val_loader_one):
        batch.feed(net, foo=foo)

    net.relax.assert_called_once_with(batch.x, batch.y, foo=foo)


def test_evaluate_batch_feed_return_contains_results_from_relax_in_fast(
    trainer, val_loader_one
):
    net = Mock()
    ret_val = "test"
    net.relax.return_value = ret_val
    for batch in trainer.evaluate(val_loader_one):
        ns = batch.feed(net)

    assert ns.fast == ret_val


def test_evaluate_batch_feed_calls_does_not_call_calculate_weight_grad(
    trainer, val_loader_one
):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    for batch in trainer.evaluate(val_loader_one):
        batch.feed(net)

    net.calculate_weight_grad.assert_not_called()


def test_evaluate_batch_contains_batch_index(trainer, val_loader):
    for i, batch in enumerate(trainer.evaluate(val_loader)):
        assert i == batch.idx


def test_evaluate_batch_every(trainer, val_loader):
    s = 3
    for batch in trainer.evaluate(val_loader):
        assert batch.every(s) == ((batch.idx % s) == 0)


def test_evaluate_batch_count(trainer, val_loader):
    c = 6
    idxs = np.linspace(0, len(val_loader) - 1, c).astype(int)
    for batch in trainer.evaluate(val_loader):
        assert batch.count(c) == (batch.idx in idxs)


def test_evaluate_batch_has_training_false(trainer, val_loader):
    for batch in trainer.evaluate(val_loader):
        assert hasattr(batch, "training")
        assert not batch.training


def test_evaluate_run_iterates_and_feeds_through_all_dataset(trainer, val_loader):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    trainer.evaluate(val_loader).run(net)

    assert net.relax.call_count == len(val_loader)


def test_trainer_has_tracker(trainer):
    assert hasattr(trainer, "tracker")
    assert isinstance(trainer.tracker, Tracker)


def test_trainer_history_is_shared_with_tracker_history(trainer):
    assert trainer.tracker.history is trainer.history


def test_tracker_index_name_is_batch(trainer):
    assert trainer.tracker.index_name == "batch"


def test_report_calls_tracker_report(trainer):
    trainer = Trainer(generate_loader())
    with patch.object(trainer.tracker, "report", wraps=trainer.tracker.report) as mock:
        for batch in trainer(1):
            batch.report.test("foo", 2.0)

        mock.test.assert_called()


def test_report_fills_in_batch(trainer):
    for batch in trainer(3):
        batch.report.test("foo", 2.0)

    assert hasattr(trainer.history, "test")
    assert "batch" in trainer.history.test
    assert len(trainer.history.test["batch"]) == 3
    assert torch.allclose(trainer.history.test["batch"], torch.arange(3))


def test_report_fills_in_sample(trainer):
    for batch in trainer(3):
        batch.report.test("foo", 2.0)

    assert len(trainer.history.test["sample"]) == 3

    batch_size = len(batch.x)
    assert torch.allclose(trainer.history.test["sample"], torch.arange(3) * batch_size)


def test_end_of_iteration_calls_tracker_finalized(trainer):
    for batch in trainer(2):
        batch.report.test("foo", 1.0)

    assert trainer.tracker.finalized


def test_end_of_evaluate_iteration_does_not_finalize_tracker(trainer, val_loader):
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    trainer.evaluate(val_loader).run(net)

    assert not trainer.tracker.finalized


def test_batch_len(trainer):
    for batch in trainer(3):
        assert len(batch) == len(batch.x)


def test_batch_keeps_track_of_sample_index(trainer):
    crt_sample = 0
    for batch in trainer(10):
        assert batch.sample_idx == crt_sample
        crt_sample += len(batch)
