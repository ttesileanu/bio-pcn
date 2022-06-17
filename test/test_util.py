from types import SimpleNamespace
import pytest

import torch
import numpy as np

from cpcn.util import pretty_size, make_onehot, one_hot_accuracy, dot_accuracy
from cpcn.util import get_constraint_diagnostics


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


@pytest.fixture()
def mock_latent():
    # a mock for testing get_constraint_diagnostics
    n_batch = 53

    latent = {}
    latent["batch"] = np.arange(n_batch)

    rng = np.random.default_rng(0)
    latent["z:0"] = rng.normal(size=(n_batch, 3))
    latent["z:1"] = rng.normal(size=(n_batch, 5))
    latent["z:2"] = rng.normal(size=(n_batch, 4))

    return latent


def test_get_constraint_diagnostics_returns_correct_batch(mock_latent):
    every = 12
    cons_diag = get_constraint_diagnostics(mock_latent, every=every)

    expected_batch = np.arange(len(mock_latent["batch"]) // every) * every
    np.testing.assert_equal(cons_diag["batch"], expected_batch)


def test_get_constraint_diagnostics_returns_correct_covariances(mock_latent):
    every = 8
    cons_diag = get_constraint_diagnostics(mock_latent, every=every)

    n_batch = len(mock_latent["batch"])
    for i in range(n_batch // every):
        start = i * every
        stop = (i + 1) * every
        for k in range(3):
            values = mock_latent[f"z:{k}"][start:stop, :]
            cov = values.T @ values / len(values)
            np.testing.assert_allclose(cons_diag[f"cov:{k}"][i], cov)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_get_constraint_diagnostics_returns_correct_trace(mock_latent, layer):
    rho = 0.3
    cons_diag = get_constraint_diagnostics(mock_latent, rho=rho)
    for cov, tr in zip(cons_diag[f"cov:{layer}"], cons_diag[f"trace:{layer}"]):
        assert pytest.approx(tr) == np.trace(cov - rho * np.eye(len(cov)))


def test_get_constraint_diagnostics_with_per_layer_rho():
    n_batch = 23

    latent = {}
    latent["batch"] = np.arange(n_batch)

    rng = np.random.default_rng(1)
    latent["z:0"] = rng.normal(size=(n_batch, 3))
    latent["z:1"] = rng.normal(size=(n_batch, 5))
    latent["z:2"] = rng.normal(size=(n_batch, 4))
    latent["z:3"] = rng.normal(size=(n_batch, 2))

    rho = [0.9, 0.3, 1.2, 0.8]
    cons_diag = get_constraint_diagnostics(latent, rho=rho)

    k = 1
    for layer in range(4):
        cov = cons_diag[f"cov:{layer}"][k]
        tr = cons_diag[f"trace:{layer}"][k]
        assert pytest.approx(tr) == np.trace(cov - rho[layer] * np.eye(len(cov)))


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_get_constraint_diagnostics_evals(mock_latent, layer):
    cons_diag = get_constraint_diagnostics(mock_latent)
    for cov, evals in zip(cons_diag[f"cov:{layer}"], cons_diag[f"evals:{layer}"]):
        expected = np.linalg.eigh(cov)[0]
        np.testing.assert_allclose(expected, evals)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_get_constraint_diagnostics_max_eval(mock_latent, layer):
    cons_diag = get_constraint_diagnostics(mock_latent)
    for ev, max in zip(cons_diag[f"evals:{layer}"], cons_diag[f"max_eval:{layer}"]):
        assert pytest.approx(np.max(ev)) == max


def test_get_constraint_diagnostics_var_has_effect(mock_latent):
    cons_diag = get_constraint_diagnostics(mock_latent, var="foo")
    assert "batch" in cons_diag
    assert len(cons_diag) == 1
