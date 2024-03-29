import pytest

import numpy as np

from types import SimpleNamespace
from unittest.mock import Mock
from cpcn.tqdm_wrapper import tqdmw


class MockTrainerIterable(list):
    trainer = SimpleNamespace(
        history=SimpleNamespace(
            train={
                "batch": [np.array([10])],
                "pc_loss": [np.array([0.3])],
                "foo": [np.array([1.5])],
            },
            all_train={
                "batch": [np.array([1, 3, 5])],
                "pc_loss": [np.array([0.2, 0.4, 0.3])],
                "foo": [np.array([0.5, 0.1, 0.2])],
            },
            validation={"batch": [np.array([10])], "pc_loss": [np.array([0.4])]},
        ),
        metrics={"pc_loss": None},
    )
    _epoch = 0


@pytest.fixture()
def mock_train_it():
    trainer = MockTrainerIterable([0.5, 1.3, 0.7, 4.3, 0.23, 6.5])
    return trainer


def test_behaves_same_as_original_iterator(mock_train_it):
    l_again = list(tqdmw(mock_train_it, tqdm=Mock()))
    assert list(mock_train_it) == l_again


def test_calls_tqdm(mock_train_it):
    mock = Mock()
    # need to iterate through tqdmw, otherwise the loop is never entered!
    list(tqdmw(mock_train_it, tqdm=mock))
    mock.assert_called()


def test_calls_update_on_iteration(mock_train_it):
    pbar = Mock()
    mock = Mock(return_value=pbar)
    list(tqdmw(mock_train_it, tqdm=mock))
    assert len(pbar.update.call_args_list) == len(mock_train_it)


def test_calls_close_at_end_of_iteration(mock_train_it):
    pbar = Mock()
    mock = Mock(return_value=pbar)
    for _ in tqdmw(mock_train_it, tqdm=mock):
        mock.close.assert_not_called()
    pbar.close.assert_called()


def test_calls_set_postfix(mock_train_it):
    pbar = Mock()
    mock = Mock(return_value=pbar)
    list(tqdmw(mock_train_it, tqdm=mock))
    pbar.set_postfix.assert_called()


def test_calls_set_postfix_with_refresh_false(mock_train_it):
    pbar = Mock()
    mock = Mock(return_value=pbar)
    list(tqdmw(mock_train_it, tqdm=mock))
    assert "refresh" in pbar.set_postfix.call_args[1]
    assert not pbar.set_postfix.call_args[1]["refresh"]


def test_calls_set_postfix_with_one_arg(mock_train_it):
    pbar = Mock()
    mock = Mock(return_value=pbar)
    list(tqdmw(mock_train_it, tqdm=mock))
    assert len(pbar.set_postfix.call_args[0]) == 1


def test_set_postfix_contains_val_pc_loss(mock_train_it):
    pbar = Mock()
    mock = Mock(return_value=pbar)
    list(tqdmw(mock_train_it, tqdm=mock))
    assert "val pc_loss" in pbar.set_postfix.call_args[0][0]


def test_set_postfix_contains_all_metrics(mock_train_it):
    pbar = Mock()
    mock = Mock(return_value=pbar)

    mock_train_it.trainer.metrics["foo"] = None
    list(tqdmw(mock_train_it, tqdm=mock))
    assert "val pc_loss" in pbar.set_postfix.call_args[0][0]
    assert "val foo" in pbar.set_postfix.call_args[0][0]


def test_does_not_fail_with_valid_data(mock_train_it):
    pbar = Mock()
    mock = Mock(return_value=pbar)
    list(tqdmw(mock_train_it, tqdm=mock))
    value = pbar.set_postfix.call_args[0][0]["val pc_loss"]
    assert "???" not in value


def test_progress_info_contains_epoch(mock_train_it):
    epoch = 123
    mock_train_it._epoch = epoch

    pbar = Mock()
    mock = Mock(return_value=pbar)
    list(tqdmw(mock_train_it, tqdm=mock))

    assert "epoch" in pbar.set_postfix.call_args[0][0]
    value = pbar.set_postfix.call_args[0][0]["epoch"]
    assert int(value) == epoch
