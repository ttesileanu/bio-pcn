from types import SimpleNamespace
import pytest

import torch
import numpy as np

from cpcn.train import Trainer, DivergenceError, DivergenceWarning, multi_lr
from cpcn.track import Tracker
from unittest.mock import Mock


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
    net.relax.return_value = SimpleNamespace(z=[torch.FloatTensor([0.0])])
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
    ret_val = SimpleNamespace(z=[], foo="test")
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
    ret_val = SimpleNamespace(z=[], foo="test")
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


def test_report_fills_in_batch(trainer):
    for batch in trainer(3):
        batch.test.report("foo", 2.0)

    assert hasattr(trainer.history, "test")
    assert "batch" in trainer.history.test
    assert len(trainer.history.test["batch"]) == 3
    np.testing.assert_allclose(trainer.history.test["batch"], np.arange(3))


def test_report_fills_in_sample(trainer):
    for batch in trainer(3):
        batch.test.report("foo", 2.0)

    assert len(trainer.history.test["sample"]) == 3

    batch_size = len(batch.x)
    np.testing.assert_allclose(
        trainer.history.test["sample"], np.arange(3) * batch_size
    )


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
        np.array([np.nan, -np.inf]),
    ],
)
def test_report_with_check_invalid(trainer, value):
    k = 3
    count = 0
    trainer.invalid_action = "stop"
    if torch.is_tensor(value):
        normal = torch.ones_like(value)
    elif isinstance(value, np.ndarray):
        normal = np.ones_like(value)
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
    losses = [torch.tensor(_) for _ in [3.5, 2.0, 1.2]]
    mock_net.pc_loss.side_effect = losses
    for batch in trainer(len(losses)):
        batch.feed(mock_net)

    np.testing.assert_allclose(trainer.history.all_train["pc_loss"], losses)


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


def test_metrics_contains_pc_loss_by_default(trainer):
    assert "pc_loss" in trainer.metrics


def test_removing_pc_loss_from_metrics_removes_default_reporting(
    trainer, mock_net, val_loader
):
    trainer.metrics.pop("pc_loss")
    for batch in trainer(10):
        batch.feed(mock_net)
        if batch.every(3):
            batch.evaluate(val_loader).run(mock_net)

    assert not hasattr(trainer.history, "all_train")
    for field in ["train", "validation"]:
        crt_dict = getattr(trainer.history, field)
        assert len(crt_dict) == 0


def test_removing_pc_loss_from_metrics_but_having_different_one_removes_it_from_dict(
    trainer, mock_net, val_loader
):
    trainer.metrics = {"nop": lambda *args, **kwargs: 0.0}
    for batch in trainer(10):
        batch.feed(mock_net)
        if batch.every(2):
            batch.evaluate(val_loader).run(mock_net)

    for field in ["all_train", "train", "validation"]:
        crt_dict = getattr(trainer.history, field)
        assert "pc_loss" not in crt_dict
        assert "nop" in crt_dict


@pytest.mark.parametrize("field", ["all_train", "train", "validation"])
def test_custom_metric_stores_values(trainer, field, val_loader):
    def custom_metric(ns, net) -> float:
        norm_z = sum(np.linalg.norm(_) for _ in ns.fast.z)
        norm_w = np.trace(net.W)
        return norm_z + norm_w

    net = Mock()
    net.W = torch.FloatTensor([[0.3, 0.5], [-0.2, 0.1]])
    net.relax.side_effect = lambda x, y: SimpleNamespace(z=[2 * x, x.T @ y])
    trainer.metrics = {"custom": custom_metric}

    for batch in trainer(10):
        if batch.every(2):
            for eval_batch in batch.evaluate(val_loader):
                eval_ns = eval_batch.feed(net)
                batch.test_validation.accumulate("custom", custom_metric(eval_ns, net))
            batch.test_validation.report_accumulated()
            batch.test_train.report_accumulated()

        ns = batch.feed(net)
        batch.test_all_train.report("custom", custom_metric(ns, net))
        batch.test_train.accumulate("custom", custom_metric(ns, net))

    auto_dict = getattr(trainer.history, field)
    test_dict = getattr(trainer.history, "test_" + field)
    np.testing.assert_allclose(auto_dict["custom"], test_dict["custom"])


@pytest.fixture()
def trainer_with_meld() -> SimpleNamespace:
    trainer = Trainer(generate_loader())
    net = Mock()
    net.pc_loss.return_value = torch.tensor(0.0)

    x, y = next(iter(trainer.loader))
    a = x.clone()
    b = torch.hstack((x.clone(), y.clone()))
    net.relax.return_value = SimpleNamespace(z=[a, b])

    n = 3
    for batch in trainer(n):
        ns = batch.feed(net)
        batch.latent.report_batch("z", ns.fast.z)

    return SimpleNamespace(trainer=trainer, a=a, b=b, n=n)


def test_trainer_report_batch_reports_correct_batches(trainer_with_meld):
    history = trainer_with_meld.trainer.history
    k = len(trainer_with_meld.a)
    n = trainer_with_meld.n

    expected_batch = np.repeat(np.arange(n), k)
    np.testing.assert_equal(history.latent["batch"], expected_batch)


def test_trainer_report_batch_reports_correct_samples(trainer_with_meld):
    history = trainer_with_meld.trainer.history
    k = len(trainer_with_meld.a)
    n = trainer_with_meld.n

    expected_sample = np.arange(n * k)
    np.testing.assert_equal(history.latent["sample"], expected_sample)


def test_trainer_report_batch_reports_correct_z0(trainer_with_meld):
    history = trainer_with_meld.trainer.history
    a = trainer_with_meld.a
    n = trainer_with_meld.n

    expected_z0 = np.vstack(n * [a.numpy()])
    np.testing.assert_allclose(history.latent["z:0"], expected_z0)


def test_trainer_report_batch_reports_correct_z1(trainer_with_meld):
    history = trainer_with_meld.trainer.history
    b = trainer_with_meld.b
    n = trainer_with_meld.n

    expected_z1 = np.vstack(n * [b.numpy()])
    np.testing.assert_allclose(history.latent["z:1"], expected_z1)


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_trainer_repr(trainer, kind):
    s = {"repr": repr, "str": str}[kind](trainer)
    assert s.startswith("Trainer(")
    assert s.endswith(")")


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_eval_iterable_repr(trainer, val_loader, kind):
    batch = next(iter(trainer(1)))
    eval_iterable = batch.evaluate(val_loader)

    s = {"repr": repr, "str": str}[kind](eval_iterable)
    assert s.startswith("EvaluationIterable(")
    assert s.endswith(")")


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_eval_batch_repr(trainer, val_loader, kind):
    batch = next(iter(trainer(1)))
    eval_iterable = batch.evaluate(val_loader)
    eval_batch = next(iter(eval_iterable))

    s = {"repr": repr, "str": str}[kind](eval_batch)
    assert s.startswith("EvaluationBatch(")
    assert s.endswith(")")


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_train_iterable_repr(trainer, kind):
    train_iterable = trainer(1)

    s = {"repr": repr, "str": str}[kind](train_iterable)
    assert s.startswith("TrainingIterable(")
    assert s.endswith(")")


@pytest.mark.parametrize("kind", ["repr", "str"])
def test_train_batch_repr(trainer, kind):
    batch = next(iter(trainer(1)))

    s = {"repr": repr, "str": str}[kind](batch)
    assert s.startswith("TrainingBatch(")
    assert s.endswith(")")


def test_multi_lr_uses_all_param_groups():
    param_groups = [
        {"name": "foo", "params": torch.FloatTensor([2.0])},
        {"name": "boo", "params": torch.FloatTensor([[3.0, -0.5], [2.0, 0.3]])},
    ]
    optim = multi_lr(torch.optim.Adam, param_groups, {})

    assert len(optim.param_groups) == len(param_groups)

    original_names = set(_["name"] for _ in param_groups)
    optim_names = set(_["name"] for _ in optim.param_groups)
    assert original_names == optim_names


def test_multi_lr_learning_rates_are_correct():
    param_groups = [
        {"name": "foo", "params": torch.FloatTensor([2.0])},
        {"name": "boo", "params": torch.FloatTensor([[3.0, -0.5], [2.0, 0.3]])},
    ]
    lr_factors = {"foo": 123, "boo": 0.023}
    lr = 0.0032
    optim = multi_lr(torch.optim.SGD, param_groups, lr_factors, lr=lr)

    for param_dict in optim.param_groups:
        expected_lr = lr * lr_factors[param_dict["name"]]
        assert pytest.approx(param_dict["lr"]) == expected_lr


def test_multi_lr_layered_variable():
    param_groups = [
        {"name": "foo:0", "params": torch.FloatTensor([2.0])},
        {"name": "foo:1", "params": torch.FloatTensor([[3.0, -0.5], [2.0, 0.3]])},
        {"name": "boo:0", "params": torch.FloatTensor([1.0, 3.0])},
        {"name": "boo:1", "params": torch.FloatTensor([1.0, -3.0])},
    ]
    lr_factors = {"foo": 1.5, "boo:0": 0.7}
    lr = 0.0032
    optim = multi_lr(torch.optim.SGD, param_groups, lr_factors, lr=lr)

    for param_dict in optim.param_groups:
        if param_dict["name"].startswith("foo:"):
            assert pytest.approx(param_dict["lr"]) == lr * lr_factors["foo"]
        elif param_dict["name"] == "boo:0":
            assert pytest.approx(param_dict["lr"]) == lr * lr_factors["boo:0"]
        else:
            assert pytest.approx(param_dict["lr"]) == lr
