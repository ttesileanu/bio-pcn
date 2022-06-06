from types import SimpleNamespace
import pytest

import torch
import numpy as np

from cpcn.train import Trainer, DivergenceError, DivergenceWarning
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
def mock_net():
    net = Mock()
    net.relax.return_value = SimpleNamespace()
    net.pc_loss.return_value = torch.tensor(0.0)

    return net


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


def test_batch_feed_calls_net_relax(trainer, mock_net):
    for batch in trainer(1):
        batch.feed(mock_net)

    mock_net.relax.assert_called_once()


def test_batch_feed_calls_net_relax_with_x_and_y(trainer, mock_net):
    for batch in trainer(1):
        batch.feed(mock_net)

    mock_net.relax.assert_called_once_with(batch.x, batch.y)


def test_batch_feed_sends_other_kwargs_to_net_relax(trainer, mock_net):
    foo = 3.5
    for batch in trainer(1):
        batch.feed(mock_net, foo=foo)

    mock_net.relax.assert_called_once_with(batch.x, batch.y, foo=foo)


def test_batch_feed_return_contains_results_from_relax_in_fast(trainer, mock_net):
    ret_val = "test"
    mock_net.relax.return_value = ret_val
    for batch in trainer(1):
        ns = batch.feed(mock_net)

    assert ns.fast == ret_val


def test_batch_feed_calls_calculate_weight_grad_with_fast(trainer, mock_net):
    for batch in trainer(1):
        ns = batch.feed(mock_net)

    mock_net.calculate_weight_grad.assert_called_once_with(ns.fast)


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


def test_iterate_evaluate_goes_through_val_dataset(trainer, val_loader):
    train_batch = next(iter(trainer(1)))
    data = [_ for _ in train_batch.evaluate(val_loader)]
    expected = [_ for _ in val_loader]
    for crt_batch, (crt_x, crt_y) in zip(data, expected):
        assert torch.allclose(crt_batch.x, crt_x)
        assert torch.allclose(crt_batch.y, crt_y)


def test_evaluate_batch_feed_calls_net_relax(trainer, val_loader_one, mock_net):
    train_batch = next(iter(trainer(1)))
    for batch in train_batch.evaluate(val_loader_one):
        batch.feed(mock_net)

    mock_net.relax.assert_called_once()


def test_evaluate_batch_feed_calls_net_relax_with_x_and_y(
    trainer, val_loader_one, mock_net
):
    train_batch = next(iter(trainer(1)))
    for batch in train_batch.evaluate(val_loader_one):
        batch.feed(mock_net)

    mock_net.relax.assert_called_once_with(batch.x, batch.y)


def test_evaluate_batch_feed_sends_other_kwargs_to_net_relax(
    trainer, val_loader_one, mock_net
):
    foo = 3.5
    train_batch = next(iter(trainer(1)))
    for batch in train_batch.evaluate(val_loader_one):
        batch.feed(mock_net, foo=foo)

    mock_net.relax.assert_called_once_with(batch.x, batch.y, foo=foo)


def test_evaluate_batch_feed_return_contains_results_from_relax_in_fast(
    trainer, val_loader_one, mock_net
):
    ret_val = "test"
    mock_net.relax.return_value = ret_val
    train_batch = next(iter(trainer(1)))
    for batch in train_batch.evaluate(val_loader_one):
        ns = batch.feed(mock_net)

    assert ns.fast == ret_val


def test_evaluate_batch_feed_calls_does_not_call_calculate_weight_grad(
    trainer, val_loader_one, mock_net
):
    train_batch = next(iter(trainer(1)))
    for batch in train_batch.evaluate(val_loader_one):
        batch.feed(mock_net)

    mock_net.calculate_weight_grad.assert_not_called()


def test_evaluate_run_iterates_and_feeds_through_all_dataset(
    trainer, val_loader, mock_net
):
    train_batch = next(iter(trainer(1)))
    train_batch.evaluate(val_loader).run(mock_net)

    assert mock_net.relax.call_count == len(val_loader)


def test_trainer_has_tracker(trainer):
    assert hasattr(trainer, "tracker")
    assert isinstance(trainer.tracker, Tracker)


def test_trainer_history_is_shared_with_tracker_history(trainer):
    assert trainer.tracker.history is trainer.history


def test_tracker_index_name_is_batch(trainer):
    assert trainer.tracker.index_name == "batch"


def test_report_fills_in_batch(trainer):
    for batch in trainer(3):
        batch.test.report("foo", 2.0)

    assert hasattr(trainer.history, "test")
    assert "batch" in trainer.history.test
    assert len(trainer.history.test["batch"]) == 3
    assert torch.allclose(trainer.history.test["batch"], torch.arange(3))


def test_report_fills_in_sample(trainer):
    for batch in trainer(3):
        batch.test.report("foo", 2.0)

    assert len(trainer.history.test["sample"]) == 3

    batch_size = len(batch.x)
    assert torch.allclose(trainer.history.test["sample"], torch.arange(3) * batch_size)


def test_end_of_iteration_calls_tracker_finalized(trainer):
    for batch in trainer(2):
        batch.test.report("foo", 1.0)

    assert trainer.tracker.finalized


def test_end_of_evaluate_iteration_does_not_finalize_tracker(
    trainer, val_loader, mock_net
):
    train_batch = next(iter(trainer(1)))
    train_batch.evaluate(val_loader).run(mock_net)

    assert not trainer.tracker.finalized


def test_batch_len(trainer):
    for batch in trainer(3):
        assert len(batch) == len(batch.x)


def test_batch_keeps_track_of_sample_index(trainer):
    crt_sample = 0
    for batch in trainer(10):
        assert batch.sample_idx == crt_sample
        crt_sample += len(batch)


def test_batch_terminate_ends_iteration(trainer):
    count = 0
    n = 5
    for batch in trainer(10):
        count += 1
        if batch.idx == n - 1:
            batch.terminate()

    assert count == n


def test_batch_terminate_only_terminates_at_the_end_of_the_for_loop(trainer):
    count = 0
    for batch in trainer(5):
        batch.terminate()
        count += 1

    assert count == 1


@pytest.mark.parametrize(
    "value",
    [
        np.inf,
        np.nan,
        torch.FloatTensor([0.0, np.inf]),
        [1.0, np.nan],
        [torch.FloatTensor([0.2, 0.3]), torch.FloatTensor([np.inf, np.nan])],
    ],
)
def test_report_with_check_invalid(trainer, value):
    k = 3
    count = 0
    trainer.invalid_action = "stop"
    if torch.is_tensor(value):
        normal = torch.ones_like(value)
    elif hasattr(value, "__len__"):
        if torch.is_tensor(value[0]):
            normal = [torch.ones_like(_) for _ in value]
        else:
            normal = [0 for _ in value]
    else:
        normal = 0.0
    for batch in trainer(5):
        count += 1
        if batch.idx != k:
            batch.test.report("foo", normal, check_invalid=True)
        else:
            batch.test.report("foo", value, check_invalid=True)

    assert count == k + 1


def test_report_check_invalid_raises_if_invalid_action_is_raise(trainer):
    trainer.invalid_action = "raise"
    with pytest.raises(DivergenceError):
        for batch in trainer(2):
            batch.test.report("foo", np.nan, check_invalid=True)


def test_default_invalid_action_is_none(trainer):
    count = 0
    n = 3
    for batch in trainer(n):
        count += 1
        batch.test.report("foo", np.nan, check_invalid=True)

    assert count == n


def test_batch_terminate_divergence_error_raises(trainer):
    with pytest.raises(DivergenceError):
        for batch in trainer(2):
            batch.terminate(divergence_error=True)


@pytest.mark.parametrize("warn_type", ["warn", "warn+stop"])
def test_report_check_invalid_warns_if_invalid_action_is_warn_or_warn_stop(
    trainer, warn_type
):
    trainer.invalid_action = warn_type
    with pytest.warns(DivergenceWarning):
        for batch in trainer(2):
            batch.test.report("foo", np.nan, check_invalid=True)


def test_report_check_invalid_does_not_stop_if_invalid_action_is_warn(trainer):
    count = 0
    n = 3
    trainer.invalid_action = "warn"
    with pytest.warns(DivergenceWarning):
        for batch in trainer(n):
            count += 1
            batch.test.report("foo", np.nan, check_invalid=True)

    assert count == n


@pytest.mark.parametrize("warn_type", ["stop", "warn+stop"])
def test_report_check_invalid_stops_if_invalid_action_is_stop_or_warn_stop(
    trainer, warn_type
):
    count = 0
    trainer.invalid_action = warn_type
    for batch in trainer(3):
        count += 1
        batch.test.report("foo", np.nan, check_invalid=True)

    assert count == 1


def test_trainer_invalid_action_in_constructor():
    trainer = Trainer(generate_loader(), invalid_action="raise")
    with pytest.raises(DivergenceError):
        for batch in trainer(2):
            batch.test.report("foo", np.nan, check_invalid=True)


@pytest.mark.parametrize("name", ["all_train", "train", "validation"])
def test_metric_history_namespaces_automatically_made(
    trainer, name, val_loader, mock_net
):
    for batch in trainer(1):
        batch.feed(mock_net)
        batch.evaluate(val_loader).run(mock_net)

    assert hasattr(trainer.history, name)


@pytest.mark.parametrize("name", ["all_train", "train", "validation"])
def test_pc_loss_metric_automatically_made(trainer, name, val_loader, mock_net):
    for batch in trainer(1):
        batch.feed(mock_net)
        batch.evaluate(val_loader).run(mock_net)

    assert "pc_loss" in getattr(trainer.history, name)


def test_pc_loss_correctly_registered_in_all_train(trainer, mock_net):
    losses = [3.5, 2.0, 1.2]
    mock_net.pc_loss.side_effect = losses
    for batch in trainer(len(losses)):
        batch.feed(mock_net)

    assert torch.allclose(
        trainer.history.all_train["pc_loss"], torch.FloatTensor(losses)
    )


def test_validation_pc_loss_correctly_registered(trainer, val_loader, mock_net):
    rng = np.random.default_rng(1)
    losses = rng.uniform(size=len(val_loader))
    mock_net.pc_loss.side_effect = losses
    for batch in trainer(1):
        for eval_batch in batch.evaluate(val_loader):
            eval_batch.feed(mock_net)

    assert len(trainer.history.validation["pc_loss"]) == 1

    avg_loss = np.mean(losses)
    assert pytest.approx(trainer.history.validation["pc_loss"].item()) == avg_loss


def test_train_pc_loss_correctly_averaged(trainer, val_loader, mock_net):
    n = 20
    step = 5

    rng = np.random.default_rng(1)
    losses = rng.uniform(size=n)
    mock_net.pc_loss.side_effect = losses
    for batch in trainer(n):
        if batch.every(step):
            for _ in batch.evaluate(val_loader):
                pass
        batch.feed(mock_net)

    k = n // step
    assert len(trainer.history.train["pc_loss"]) == k - 1

    for i in range(k - 1):
        avg_loss = np.mean(losses[i * step : (i + 1) * step])
        assert pytest.approx(trainer.history.train["pc_loss"][i].item()) == avg_loss


def test_check_invalid_with_multi_parameter_report(trainer):
    trainer.invalid_action = "raise"
    with pytest.raises(DivergenceError):
        for batch in trainer(2):
            batch.test.report({"a": 0.0, "b": np.nan}, check_invalid=True)
