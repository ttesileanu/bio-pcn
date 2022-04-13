import pytest

from cpcn.linear import LinearCPCNetwork

import torch


@pytest.fixture
def net():
    net = LinearCPCNetwork([3, 4, 5, 2])
    return net


@pytest.fixture
def net_inter_dims():
    net = LinearCPCNetwork([3, 4, 5, 2], inter_dims=[2, 7])
    return net


@pytest.fixture
def net_no_bias_a():
    net = LinearCPCNetwork([5, 3, 4, 3], bias_a=False)
    return net


@pytest.fixture
def net_no_bias_b():
    net = LinearCPCNetwork([2, 6, 2], bias_b=False)
    return net


def test_default_interneuron_dims_match_pyramidal(net):
    assert len(net.inter_dims) == len(net.pyr_dims) - 2
    assert all(net.inter_dims == net.pyr_dims[1:-1])


def test_number_of_layers(net):
    # number of hidden layers
    n = len(net.pyr_dims)
    D = n - 2

    # weights and biases
    assert len(net.W_a) == D
    assert len(net.W_b) == D
    assert len(net.Q) == D
    assert len(net.M) == D
    assert len(net.h_a) == D
    assert len(net.h_b) == D

    # currents
    assert len(net.a) == D
    assert len(net.b) == D
    assert len(net.n) == D

    # input, hidden, and output activations
    assert len(net.z) == n


def test_apical_weight_sizes(net):
    # apical weights are feedback: start at first hidden layer, end with output layer
    assert net.W_a[0].shape == (5, 4)
    assert net.W_a[1].shape == (2, 5)


def test_basal_weight_sizes(net):
    # basal weights are feedforward: start at input layer, end at last hidden layer
    assert net.W_b[0].shape == (4, 3)
    assert net.W_b[1].shape == (5, 4)


def test_interneuron_weight_sizes(net):
    assert net.Q[0].shape == (4, 4)
    assert net.Q[1].shape == (5, 5)


def test_interneuron_weight_sizes_special_inter(net_inter_dims):
    assert net_inter_dims.Q[0].shape == (2, 4)
    assert net_inter_dims.Q[1].shape == (7, 5)


def test_lateral_weight_sizes(net):
    assert net.M[0].shape == (4, 4)
    assert net.M[1].shape == (5, 5)


def test_apical_bias_sizes(net):
    assert net.h_a[0].shape == (5,)
    assert net.h_a[1].shape == (2,)


def test_basal_bias_sizes(net):
    assert net.h_b[0].shape == (4,)
    assert net.h_b[1].shape == (5,)


def test_no_bias_a(net_no_bias_a):
    assert net_no_bias_a.h_a is None
    assert net_no_bias_a.h_b is not None


def test_no_bias_b(net_no_bias_b):
    assert net_no_bias_b.h_b is None
    assert net_no_bias_b.h_a is not None


def test_z_sizes(net):
    net.forward(torch.FloatTensor([0.3, -0.2, 0.5]))
    assert [len(_) for _ in net.z] == [3, 4, 5, 2]


def test_all_z_not_none_after_forward(net):
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for z in net.z:
        assert z is not None
