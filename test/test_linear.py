import pytest

from types import SimpleNamespace

from cpcn.linear import LinearCPCNetwork
from cpcn.pcn import PCNetwork

import torch
import numpy as np


@pytest.fixture
def net():
    # ensure non-trivial conductances
    net = LinearCPCNetwork(
        [3, 4, 5, 2], g_a=[0.4, 0.8], g_b=[1.2, 0.5], c_m=[0.3, 0.7], l_s=[2.5, 1.8]
    )
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


def test_all_z_nonzero_after_forward(net):
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for z in net.z:
        assert torch.max(torch.abs(z)) > 1e-4


def test_all_z_change_during_forward(net):
    # set some starting values for x
    net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))

    old_z = [_.clone() for _ in net.z]
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for old, new in zip(old_z, net.z):
        assert not torch.any(torch.isclose(old, new))


def test_forward_constrained_starts_with_forward(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])

    net.z_it = 0
    net.forward_constrained(x, torch.FloatTensor([0.3, -0.4]))

    old_z = [_.clone() for _ in net.z]
    net.forward(x)

    for old, new in zip(old_z[:-1], net.z[:-1]):
        assert torch.all(torch.isclose(old, new))


def test_all_z_nonzero_after_forward_constrained(net):
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    for z in net.z:
        assert torch.max(torch.abs(z)) > 1e-4


def test_all_z_change_during_forward_constrained(net):
    # set some starting values for x
    net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))

    old_z = [_.clone() for _ in net.z]
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    for old, new in zip(old_z, net.z):
        assert not torch.all(torch.isclose(old, new))


def test_interneuron_current_calculation(net):
    net.forward(torch.FloatTensor([-0.3, 0.1, 0.4]))
    net.calculate_currents()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.Q[i] @ net.z[i + 1]
        assert torch.all(torch.isclose(net.n[i], expected))


def test_basal_current_calculation(net):
    net.forward(torch.FloatTensor([-0.3, 0.1, 0.4]))
    net.calculate_currents()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.W_b[i] @ net.z[i] + net.h_b[i]
        assert torch.all(torch.isclose(net.b[i], expected))


def test_basal_current_calculation_no_bias(net_no_bias_b):
    net = net_no_bias_b
    net.forward(torch.FloatTensor([-0.3, 0.1]))
    net.calculate_currents()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.W_b[i] @ net.z[i]
        assert torch.all(torch.isclose(net.b[i], expected))


def test_apical_current_calculation(net):
    net.forward(torch.FloatTensor([-0.3, 0.1, 0.4]))
    net.calculate_currents()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.W_a[i].T @ (net.z[i + 2] - net.h_a[i]) - net.Q[i].T @ net.n[i]
        assert torch.all(torch.isclose(net.a[i], expected))


def test_apical_current_calculation_no_bias(net):
    net.forward(torch.FloatTensor([-0.3, 0.1, 0.4]))
    net.calculate_currents()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.W_a[i].T @ net.z[i + 2] - net.Q[i].T @ net.n[i]
        assert torch.all(torch.isclose(net.a[i], expected))


@pytest.mark.parametrize("which", ["W_a", "W_b", "Q", "M"])
def test_initial_params_same_when_torch_seed_is_same(which: str):
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = LinearCPCNetwork(dims)

    old = [_.clone().detach() for _ in getattr(net, which)]

    torch.manual_seed(seed)
    net = LinearCPCNetwork(dims)

    new = [_.clone().detach() for _ in getattr(net, which)]

    for crt_old, crt_new in zip(old, new):
        assert torch.allclose(crt_old, crt_new)


@pytest.mark.parametrize("which", ["W_a", "W_b", "Q", "M"])
def test_initial_params_change_for_subsequent_calls_if_seed_not_reset(which: str):
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = LinearCPCNetwork(dims)

    old = [_.clone().detach() for _ in getattr(net, which)]

    net = LinearCPCNetwork(dims)
    new = [_.clone().detach() for _ in getattr(net, which)]

    for crt_old, crt_new in zip(old, new):
        assert not torch.any(torch.isclose(crt_old, crt_new))


def test_initial_biases_are_zero(net):
    for h in net.h_a + net.h_b:
        assert torch.all(torch.isclose(h, torch.FloatTensor([0])))


def test_z_grad(net):
    net.forward(torch.FloatTensor([0.2, -0.5, 0.3]))
    net.calculate_currents()
    net.calculate_z_grad()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        hidden_apical = net.g_a[i] * net.a[i]
        hidden_basal = net.g_b[i] * net.b[i]
        hidden_lateral = net.c_m[i] * net.M[i] @ net.z[i + 1]
        hidden_leak = net.l_s[i] * net.z[i + 1]

        expected = hidden_apical + hidden_basal - hidden_lateral - hidden_leak

        assert torch.all(torch.isclose(net.z[i + 1].grad, -expected))


def test_no_z_grad_for_input_and_output(net):
    net.forward(torch.FloatTensor([0.2, -0.5, 0.3]))
    net.calculate_currents()
    net.calculate_z_grad()

    # ensure gradient is zero for input and output layers
    for i in [0, -1]:
        assert net.z[i].grad is None


def test_pc_loss_function(net):
    net.forward(torch.FloatTensor([0.2, -0.5, 0.3]))
    loss = net.pc_loss().item()

    expected = 0
    D = len(net.pyr_dims) - 2
    for i in range(D):
        err_apical = net.z[i + 2] - net.W_a[i] @ net.z[i + 1] - net.h_a[i]
        apical = 0.5 * net.g_a[i] * torch.linalg.norm(err_apical) ** 2

        err_basal = net.z[i + 1] - net.W_b[i] @ net.z[i] - net.h_b[i]
        basal = 0.5 * net.g_b[i] * torch.linalg.norm(err_basal) ** 2

        expected += (apical + basal).item()

    assert loss == pytest.approx(expected)


def test_apical_weight_gradient(net):
    net.g_a[0] = 1.0
    net.g_a[1] = 1.0

    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    net.forward_constrained(x, y)
    net.calculate_weight_grad()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = torch.outer(net.z[i + 2] - net.h_a[i], net.z[i + 1]) - net.W_a[i]
        assert torch.all(torch.isclose(net.W_a[i].grad, -expected))


def test_apical_weight_gradient_scaling_with_apical_conductance(net):
    net.g_a[0] = 1.0
    net.g_a[1] = 1.0

    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    net.forward_constrained(x, y)
    net.calculate_weight_grad()

    old_W_a_grads = [_.grad.clone().detach() for _ in net.W_a]

    # if i don't update any other variables, the delta W should be prop to g
    net.g_a[0] = 0.5
    net.g_a[1] = 1.3

    net.calculate_weight_grad()

    for old, new, g in zip(old_W_a_grads, net.W_a, net.g_a):
        assert torch.all(torch.isfinite(new.grad))
        assert torch.all(torch.isclose(new.grad, g * old))


def test_basal_weight_gradient(net):
    x = torch.FloatTensor([-0.4, 0.5, 0.3])
    y = torch.FloatTensor([0.2, 0.3])
    net.forward_constrained(x, y)
    net.calculate_weight_grad()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        plateau = net.g_a[i] * torch.outer(net.a[i], net.z[i])
        lateral = net.c_m[i] * net.M[i] @ net.z[i + 1]
        self = (net.l_s[i] - net.g_b[i]) * net.z[i + 1]
        hebbian = torch.outer(self + lateral, net.z[i])

        assert torch.all(torch.isclose(-net.W_b[i].grad, plateau - hebbian))


def test_interneuron_weight_gradient(net):
    x = torch.FloatTensor([-0.4, 0.5, 0.3])
    y = torch.FloatTensor([0.2, 0.3])
    net.forward_constrained(x, y)
    net.calculate_weight_grad()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = torch.outer(net.n[i], net.z[i + 1]) - net.Q[i]
        assert torch.all(torch.isclose(net.Q[i].grad, -net.g_a[i] * expected))


def test_lateral_weight_gradient(net):
    x = torch.FloatTensor([-0.4, 0.5, 0.3])
    y = torch.FloatTensor([0.2, 0.3])
    net.forward_constrained(x, y)
    net.calculate_weight_grad()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = torch.outer(net.z[i + 1], net.z[i + 1]) - net.M[i]
        assert torch.all(torch.isclose(net.M[i].grad, -net.c_m[i] * expected))


def test_on_shell_after_forward_constrained(net):
    # i.e., z reaches stationarity
    x = torch.FloatTensor([-0.4, 0.5, 0.3])
    y = torch.FloatTensor([0.2, 0.3])
    net.z_it = 1000
    net.forward_constrained(x, y)

    net.calculate_currents()

    D = len(net.pyr_dims) - 2
    for i in range(D):
        apical = net.g_a[i] * net.a[i]
        basal = net.g_b[i] * net.b[i]
        lateral = net.c_m[i] * net.M[i] @ net.z[i + 1]
        leak = net.l_s[i] * net.z[i + 1]
        rhs = apical + basal - lateral - leak

        assert torch.max(torch.abs(rhs)) < 1e-5


@pytest.mark.parametrize("which", ["g_a", "g_b", "c_m", "l_s"])
def test_allow_tensor_conductances_in_constructor(which):
    kwargs = {which: torch.FloatTensor([1.3, 2.5])}
    pcn = LinearCPCNetwork([2, 5, 4, 3], **kwargs)
    assert getattr(pcn, which).shape == (2,)


@pytest.mark.parametrize("which", ["g_a", "g_b", "c_m", "l_s"])
def test_allow_scalar_tensor_conductances_in_constructor(which):
    kwargs = {which: torch.FloatTensor([1.3])}
    pcn = LinearCPCNetwork([2, 5, 4, 3], **kwargs)
    assert getattr(pcn, which).shape == (2,)


def test_cpcn_pc_loss_matches_pcn_loss_with_appropriate_params():
    torch.manual_seed(2)

    dims = [2, 4, 3, 7, 3]
    variances = [0.5, 1.2, 0.7, 0.3]

    pcn = PCNetwork(dims, variances=variances, activation=lambda _: _)

    # match CPCN conductances to PCN variances
    variances = torch.FloatTensor(variances)
    g_a = 0.5 / variances[1:]
    g_b = 0.5 / variances[:-1]

    g_a[-1] *= 2
    g_b[0] *= 2
    cpcn = LinearCPCNetwork(dims, g_a=g_a, g_b=g_b, c_m=0, l_s=g_b)

    # match the weights
    D = len(dims) - 2
    for i in range(D):
        cpcn.W_a[i] = pcn.W[i + 1].clone().detach()
        cpcn.W_b[i] = pcn.W[i].clone().detach()

    # pass some data through the network to set the neural activities
    pcn.forward_constrained(
        torch.FloatTensor([-0.3, 0.2]), torch.FloatTensor([0.4, -0.2, 0.5])
    )

    # copy the neural activations over to CPCN
    for i in range(len(dims)):
        cpcn.z[i] = pcn.z[i].clone().detach()

    # now calculate and compare loss
    pcn_loss = pcn.loss().item()
    cpcn_loss = cpcn.pc_loss().item()

    assert pcn_loss == pytest.approx(cpcn_loss)


def test_init_params_with_numpy_scalar():
    net = LinearCPCNetwork([2, 3, 4], g_a=np.asarray([0.5]))


def test_to_returns_self(net):
    assert net.to(torch.device("cpu")) is net


def test_repr(net):
    s = repr(net)

    assert s.startswith("LinearCPCNetwork(")
    assert s.endswith(")")


def test_str(net):
    s = str(net)

    assert s.startswith("LinearCPCNetwork(")
    assert s.endswith(")")


def test_a_size_when_inter_different_from_pyr(net_inter_dims):
    D = len(net_inter_dims.inter_dims)
    for i in range(D):
        dim = net_inter_dims.pyr_dims[i + 1]
        assert len(net_inter_dims.a[i]) == dim


def test_b_size_when_inter_different_from_pyr(net_inter_dims):
    D = len(net_inter_dims.inter_dims)
    for i in range(D):
        dim = net_inter_dims.pyr_dims[i + 1]
        assert len(net_inter_dims.b[i]) == dim


def test_n_size_when_inter_different_from_pyr(net_inter_dims):
    D = len(net_inter_dims.inter_dims)
    for i in range(D):
        dim = net_inter_dims.inter_dims[i]
        assert len(net_inter_dims.n[i]) == dim


def test_run_on_batch(net):
    x = torch.FloatTensor([[-0.2, -0.3, 0.5], [0.1, 0.5, -1.2]])
    y = torch.FloatTensor([[1.2, -0.5], [-0.3, -0.4]])
    net.forward_constrained(x, y)

    batch_z = [_.clone().detach() for _ in net.z]

    for i in range(len(x)):
        net.forward_constrained(x[i], y[i])
        for old_z, new_z in zip(batch_z, net.z):
            assert torch.allclose(old_z[i], new_z)


def test_pc_loss_on_batch(net):
    x = torch.FloatTensor([[-0.2, -0.3, 0.5], [0.1, 0.5, -1.2]])
    y = torch.FloatTensor([[1.2, -0.5], [-0.3, -0.4]])
    net.forward_constrained(x, y)
    batch_loss = net.pc_loss()

    loss = 0
    for i in range(len(x)):
        net.forward_constrained(x[i], y[i])
        loss += net.pc_loss()

    assert batch_loss.item() == pytest.approx(loss.item() / len(x))


def test_weight_grad_on_batch(net):
    x = torch.FloatTensor([[-0.2, -0.3, 0.5], [0.1, 0.5, -1.2]])
    y = torch.FloatTensor([[1.2, -0.5], [-0.3, -0.4]])
    net.forward_constrained(x, y)
    net.calculate_weight_grad()

    batch_W_grad = [_.grad.clone().detach() for _ in net.W_a + net.W_b]

    expected_grads = [torch.zeros_like(_) for _ in batch_W_grad]
    for i in range(len(x)):
        net.forward_constrained(x[i], y[i])
        net.calculate_weight_grad()
        for k, new_W in enumerate(net.W_a + net.W_b):
            expected_grads[k] += new_W.grad

    for old, new in zip(batch_W_grad, expected_grads):
        assert torch.allclose(old, new / len(x))


def test_bias_grad_on_batch(net):
    x = torch.FloatTensor([[-0.2, -0.3, 0.5], [0.1, 0.5, -1.2]])
    y = torch.FloatTensor([[1.2, -0.5], [-0.3, -0.4]])
    net.forward_constrained(x, y)
    net.calculate_weight_grad()

    batch_h_grad = [_.grad.clone().detach() for _ in net.h_a + net.h_b]

    expected_grads = [torch.zeros_like(_) for _ in batch_h_grad]
    for i in range(len(x)):
        net.forward_constrained(x[i], y[i])
        net.calculate_weight_grad()
        for k, new_h in enumerate(net.h_a + net.h_b):
            expected_grads[k] += new_h.grad

    for old, new in zip(batch_h_grad, expected_grads):
        assert torch.allclose(old, new / len(x))


def test_weights_change_when_optimizing_slow_parameters(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    old_Ws = [_.clone().detach() for _ in net.W_a + net.W_b + net.Q + net.M]

    optimizer = torch.optim.Adam(net.slow_parameters(), lr=1.0)
    net.forward_constrained(x0, y0)
    net.calculate_weight_grad()
    optimizer.step()

    new_Ws = net.W_a + net.W_b + net.Q + net.M
    for old_W, new_W in zip(old_Ws, new_Ws):
        assert not torch.any(torch.isclose(old_W, new_W))


def test_biases_change_when_optimizing_slow_parameters(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    old_hs = [_.clone().detach() for _ in net.h_a + net.h_b]

    optimizer = torch.optim.Adam(net.slow_parameters(), lr=1.0)
    net.forward_constrained(x0, y0)
    net.calculate_weight_grad()
    optimizer.step()

    new_hs = net.h_a + net.h_b
    for old_h, new_h in zip(old_hs, new_hs):
        assert not torch.any(torch.isclose(old_h, new_h))


def test_no_nan_or_inf_after_a_few_learning_steps(net):
    torch.manual_seed(0)

    optimizer = torch.optim.Adam(net.slow_parameters(), lr=1e-3)
    for i in range(4):
        x = torch.Tensor(3).uniform_()
        y = torch.Tensor(2).uniform_()
        net.forward_constrained(x, y)
        net.calculate_weight_grad()
        optimizer.step()

    for W in net.W_a + net.W_b + net.Q + net.M:
        assert torch.all(torch.isfinite(W))

    for h in net.h_a + net.h_b:
        assert torch.all(torch.isfinite(h))

    for z in net.z:
        assert torch.all(torch.isfinite(z))


def linear_cpcn_loss(net: LinearCPCNetwork, reduction: str = "sum") -> torch.Tensor:
    D = len(net.inter_dims)
    batch_size = 1 if net.z[0].ndim == 1 else len(net.z[0])
    loss = torch.zeros(batch_size)
    batch_outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
    for i in range(D):
        z = net.z[i + 1]

        mu = net.z[i] @ net.W_b[i].T + net.h_b[i]
        error = z - mu
        basal = 0.5 * net.g_b[i] * (error ** 2).sum(dim=-1)

        diff = net.z[i + 2] - net.h_a[i]
        cross_term = (diff * (z @ net.W_a[i].T)).sum(dim=-1)
        wa_reg = torch.trace(net.W_a[i] @ net.W_a[i].T)
        apical0 = (diff ** 2).sum(dim=-1) - 2 * cross_term + wa_reg
        apical = 0.5 * net.g_a[i] * apical0

        # need to flip this for Q because we're maximizing!!
        # but note that this would lead to incorrect dynamics for the latents!
        # (we get away with it because we don't use this objective for z dynamics)
        z_cons = batch_outer(z, z) - net.rho[i] * torch.eye(z.shape[-1])
        q_prod = net.Q[i].T @ net.Q[i]
        constraint0 = (q_prod @ z_cons).diagonal(dim1=-1, dim2=-2).sum(dim=-1)
        constraint = -0.5 * net.g_a[i] * constraint0

        alpha = net.l_s[i] - net.g_b[i]
        z_reg = 0.5 * alpha * (z ** 2).sum(dim=-1)

        m_prod = (z * (z @ net.M[i].T)).sum(dim=-1)
        m_reg = torch.trace(net.M[i] @ net.M[i].T)
        lateral0 = -2 * m_prod + m_reg
        lateral = 0.5 * net.c_m[i] * lateral0

        loss += basal + apical + constraint + z_reg + lateral

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction != "none":
        raise ValueError("unknown reduction type")

    return loss


@pytest.fixture
def data() -> tuple:
    x = torch.FloatTensor([[-0.1, 0.2, 0.4], [0.5, 0.3, 0.2], [-1.0, 2.3, 0.1]])
    y = torch.FloatTensor([[0.3, -0.4], [0.1, 0.2], [-0.5, 1.2]])
    return SimpleNamespace(x=x, y=y)


def test_cpcn_loss_reduction_none(net, data):
    net.forward_constrained(data.x, data.y)
    loss = linear_cpcn_loss(net, reduction="none")

    for i, (crt_x, crt_y) in enumerate(zip(data.x, data.y)):
        net.forward_constrained(crt_x, crt_y)
        crt_loss = linear_cpcn_loss(net)

        assert loss[i].item() == pytest.approx(crt_loss.item())


def test_cpcn_loss_reduction_sum(net, data):
    net.forward_constrained(data.x, data.y)
    loss = linear_cpcn_loss(net, reduction="sum")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        net.forward_constrained(crt_x, crt_y)
        expected += linear_cpcn_loss(net)

    assert loss.item() == pytest.approx(expected.item())


def test_cpcn_loss_reduction_mean(net, data):
    net.forward_constrained(data.x, data.y)
    loss = linear_cpcn_loss(net, reduction="mean")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        net.forward_constrained(crt_x, crt_y)
        expected += linear_cpcn_loss(net)

    expected /= len(data.x)
    assert loss.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("var", ["W_a", "W_b", "h_a", "h_b", "Q", "M"])
def test_weight_gradients_match_autograd_from_loss(net, var):
    net.z_it = 1000

    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])

    net.forward_constrained(x0, y0)
    net.calculate_weight_grad()

    manual_grads = [_.grad.clone().detach() for _ in getattr(net, var)]

    for param in net.slow_parameters():
        param.requires_grad_()
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    loss = linear_cpcn_loss(net)
    loss.backward()

    loss_grads = [_.grad.clone().detach() for _ in getattr(net, var)]

    for from_manual, from_loss in zip(manual_grads, loss_grads):
        assert torch.allclose(from_manual, from_loss, rtol=1e-2, atol=1e-5)


def test_forward_constrained_returns_empty_namespace_by_default(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.forward_constrained(x, y)

    assert len(ns.__dict__) == 0


def test_forward_constrained_loss_profile_is_sequence_of_correct_length(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.forward_constrained(x, y, pc_loss_profile=True)

    assert len(ns.pc_loss) == net.z_it


def test_forward_constrained_loss_profile_is_sequence_of_positive_numbers(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.forward_constrained(x, y, pc_loss_profile=True)

    assert min(ns.pc_loss) > 0


def test_forward_constrained_latent_profile_batch_index_added(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.forward_constrained(x, y, latent_profile=True)
    for var in ns.latent.__dict__:
        crt_data = getattr(ns.latent, var)
        for x in crt_data:
            assert x.ndim == 3
            assert x.shape[1] == 1


def test_forward_constrained_latent_profile_has_z_a_b_n(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.forward_constrained(x, y, latent_profile=True)

    assert set(ns.latent.__dict__.keys()) == {"z", "a", "b", "n"}


@pytest.mark.parametrize("var", ["z", "a", "b", "n"])
def test_forward_constrained_latent_profile_has_correct_length(net, var):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.forward_constrained(x, y, latent_profile=True)
    crt_data = getattr(ns.latent, var)
    for x in crt_data:
        assert len(x) == net.z_it


def test_forward_constrained_latent_profile_first_layer_of_z_is_input(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.forward_constrained(x, y, latent_profile=True)
    assert torch.max(torch.abs(ns.latent.z[0] - x)) < 1e-5


def test_forward_constrained_latent_profile_last_layer_of_z_is_output(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.forward_constrained(x, y, latent_profile=True)
    assert torch.max(torch.abs(ns.latent.z[-1] - y)) < 1e-5


@pytest.mark.parametrize("var", ["z", "a", "b", "n"])
def test_forward_constrained_latent_profile_row_matches_shorter_run(net, var):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.forward_constrained(x, y, latent_profile=True)

    new_it = net.z_it // 2
    net.z_it = new_it
    net.forward_constrained(x, y)

    long_data = getattr(ns.latent, var)
    for i, x in enumerate(long_data):
        short_data = getattr(net, var)
        assert torch.allclose(short_data[i], x[new_it - 1, 0])


@pytest.mark.parametrize("var", ["z", "a", "b", "n"])
def test_forward_constrained_latent_profile_row_matches_shorter_run_interdims(
    net_inter_dims, var
):
    net = net_inter_dims
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.forward_constrained(x, y, latent_profile=True)

    new_it = net.z_it // 2
    net.z_it = new_it
    net.forward_constrained(x, y)

    long_data = getattr(ns.latent, var)
    for i, x in enumerate(long_data):
        short_data = getattr(net, var)
        assert torch.allclose(short_data[i], x[new_it - 1, 0])


@pytest.mark.parametrize("var", ["z", "a", "b", "n"])
def test_forward_constrained_latent_profile_with_batch(net, var):
    x = torch.FloatTensor([[-0.1, 0.2, 0.4], [0.5, 0.3, 0.2]])
    y = torch.FloatTensor([[0.3, -0.4], [0.1, 0.2]])
    ns = net.forward_constrained(x, y, latent_profile=True)

    long_data = getattr(ns.latent, var)
    for k in range(len(x)):
        crt_x = x[k]
        crt_y = y[k]
        crt_ns = net.forward_constrained(crt_x, crt_y, latent_profile=True)

        short_data = getattr(crt_ns.latent, var)
        for x1, x2 in zip(long_data, short_data):
            assert torch.allclose(x1[:, [k], :], x2, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("var", ["a", "b", "n"])
def test_currents_are_consistent_with_z_after_forward_constraint(net, var):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    net.forward_constrained(x, y)

    old = [_.clone().detach() for _ in getattr(net, var)]
    net.calculate_currents()

    for a, b in zip(old, getattr(net, var)):
        assert torch.allclose(a, b)


def test_forward_not_inplace_does_not_change_z(net):
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )
    old_z = [_.detach().clone() for _ in net.z]

    net.forward(torch.FloatTensor([0.1, 0.2, 0.3]), inplace=False)
    for old, new in zip(old_z, net.z):
        assert torch.allclose(old, new)


def test_forward_not_inplace_return_matches_in_place_values(net):
    x = torch.FloatTensor([0.1, 0.2, 0.3])
    net.forward(x)
    inplace_z = [_.detach().clone() for _ in net.z]

    # scramble the z's
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    # check what happens without inplace
    new_z = net.forward(x, inplace=False)
    for old, new in zip(inplace_z, new_z):
        assert torch.allclose(old, new)


def test_forward_not_inplace_return_matches_in_place_values_batch(net):
    x = torch.FloatTensor([[0.1, 0.2, -0.3], [-0.5, 0.3, 0.2]])
    net.forward(x)
    inplace_z = [_.detach().clone() for _ in net.z]

    # scramble the z's
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    # check what happens without inplace
    new_z = net.forward(x, inplace=False)
    for old, new in zip(inplace_z, new_z):
        assert torch.allclose(old, new)


def test_forward_always_returns_all_layers_of_z(net):
    z = net.forward(torch.FloatTensor([0.1, 0.2, 0.3]))

    for x, y in zip(z, net.z):
        assert torch.allclose(x, y)


def test_loss_reduction_none(net, data):
    net.forward_constrained(data.x, data.y)
    loss = net.pc_loss(reduction="none")

    for i, (crt_x, crt_y) in enumerate(zip(data.x, data.y)):
        net.forward_constrained(crt_x, crt_y)
        crt_loss = net.pc_loss()

        assert loss[i].item() == pytest.approx(crt_loss.item())


def test_loss_reduction_sum(net, data):
    net.forward_constrained(data.x, data.y)
    loss = net.pc_loss(reduction="sum")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        net.forward_constrained(crt_x, crt_y)
        expected += net.pc_loss()

    assert loss.item() == pytest.approx(expected.item())


def test_loss_reduction_mean(net, data):
    net.forward_constrained(data.x, data.y)
    loss = net.pc_loss(reduction="mean")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        net.forward_constrained(crt_x, crt_y)
        expected += net.pc_loss()

    expected /= len(data.x)
    assert loss.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("var", ["W_a", "W_b", "h_a", "h_b", "Q", "M"])
@pytest.mark.parametrize("red", ["sum", "mean"])
def test_weight_gradients_match_autograd_from_loss_batch(net, data, var, red):
    net.z_it = 1000

    net.forward_constrained(data.x, data.y)
    net.calculate_weight_grad(reduction=red)

    manual_grads = [_.grad.clone().detach() for _ in getattr(net, var)]

    for param in net.slow_parameters():
        param.requires_grad_()
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    loss = linear_cpcn_loss(net, reduction=red)
    loss.backward()

    loss_grads = [_.grad.clone().detach() for _ in getattr(net, var)]

    for from_manual, from_loss in zip(manual_grads, loss_grads):
        assert torch.allclose(from_manual, from_loss, rtol=1e-2, atol=1e-5)


def test_train(net):
    net.train()
    assert net.training


def test_eval(net):
    net.eval()
    assert not net.training


@pytest.fixture
def net_nontrivial_constraint():
    # ensure non-trivial conductances *and* non-trivial constraints
    net = LinearCPCNetwork(
        [3, 4, 5, 2],
        g_a=[0.4, 0.8],
        g_b=[1.2, 0.5],
        c_m=[0.3, 0.7],
        l_s=[2.5, 1.8],
        rho=[0.7, 1.5],
    )
    return net


def test_foward_constrained_unaffected_by_nontrivial_constraint(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    net.forward_constrained(x, y)

    old_z = [_.detach().clone() for _ in net.z]

    net.rho[0] = 0.7
    net.rho[1] = 1.5
    net.forward_constrained(x, y)

    for old, new in zip(old_z, net.z):
        assert torch.allclose(old, new)


def test_q_gradient_with_nontrivial_constraint(net_nontrivial_constraint, data):
    net = net_nontrivial_constraint
    net.forward_constrained(data.x, data.y)
    net.calculate_weight_grad()

    outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
    for i in range(len(net.inter_dims)):
        expected = net.g_a[i] * (
            net.rho[i] * net.Q[i] - outer(net.n[i], net.z[i + 1])
        ).mean(dim=0)
        assert torch.allclose(net.Q[i].grad, expected)


def test_q_gradient_with_nontrivial_constraint_vs_autograd(net_nontrivial_constraint):
    net = net_nontrivial_constraint

    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    net.forward_constrained(x, y)
    net.calculate_weight_grad()

    manual_grads = [_.grad.clone().detach() for _ in net.Q]

    for param in net.slow_parameters():
        param.requires_grad_()
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    loss = linear_cpcn_loss(net)
    loss.backward()

    loss_grads = [_.grad.clone().detach() for _ in net.Q]

    for from_manual, from_loss in zip(manual_grads, loss_grads):
        assert torch.allclose(from_manual, from_loss)
