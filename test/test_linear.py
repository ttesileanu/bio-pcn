import pytest

from types import SimpleNamespace

from cpcn.linear import LinearBioPCN
from cpcn.pcn import PCNetwork

import torch
import numpy as np


@pytest.fixture
def net():
    # ensure non-trivial conductances
    net = LinearBioPCN(
        [3, 4, 5, 2], g_a=[0.4, 0.8], g_b=[1.2, 0.5], c_m=[0.3, 0.7], l_s=[2.5, 1.8]
    )
    return net


@pytest.fixture
def net_inter_dims():
    net = LinearBioPCN([3, 4, 5, 2], inter_dims=[2, 7])
    return net


@pytest.fixture
def net_no_bias_a():
    net = LinearBioPCN([5, 3, 4, 3], bias_a=False)
    return net


@pytest.fixture
def net_no_bias_b():
    net = LinearBioPCN([2, 6, 2], bias_b=False)
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
    z = net.forward(torch.FloatTensor([0.3, -0.2, 0.5]))
    assert [len(_) for _ in z] == [3, 4, 5, 2]


def test_all_z_nonzero_after_forward(net):
    z = net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for crt_z in z:
        assert torch.max(torch.abs(crt_z)) > 1e-4


def test_all_z_change_during_forward(net):
    # set some starting values for x
    old_z = net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))
    new_z = net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for old, new in zip(old_z, new_z):
        assert not torch.any(torch.isclose(old, new))


def test_relax_starts_with_forward(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])

    net.z_it = 0
    ns = net.relax(x, torch.FloatTensor([0.3, -0.4]))
    new_z = net.forward(x)

    for old, new in zip(ns.z[:-1], new_z[:-1]):
        assert torch.all(torch.isclose(old, new))


def test_all_z_nonzero_after_relax(net):
    ns = net.relax(torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4]))

    for z in ns.z:
        assert torch.max(torch.abs(z)) > 1e-4


def test_all_z_change_during_relax(net):
    # set some starting values for x
    old_z = net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))
    ns = net.relax(torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4]))

    for old, new in zip(old_z, ns.z):
        assert not torch.all(torch.isclose(old, new))


def test_interneuron_current_calculation(net):
    z = net.forward(torch.FloatTensor([-0.3, 0.1, 0.4]))
    _, _, n = net.calculate_currents(z)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.Q[i] @ z[i + 1]
        assert torch.all(torch.isclose(n[i], expected))


def test_basal_current_calculation(net):
    z = net.forward(torch.FloatTensor([-0.3, 0.1, 0.4]))
    _, b, _ = net.calculate_currents(z)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.W_b[i] @ z[i] + net.h_b[i]
        assert torch.all(torch.isclose(b[i], expected))


def test_basal_current_calculation_no_bias(net_no_bias_b):
    net = net_no_bias_b
    z = net.forward(torch.FloatTensor([-0.3, 0.1]))
    _, b, _ = net.calculate_currents(z)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.W_b[i] @ z[i]
        assert torch.all(torch.isclose(b[i], expected))


def test_apical_current_calculation(net):
    z = net.forward(torch.FloatTensor([-0.3, 0.1, 0.4]))
    a, _, n = net.calculate_currents(z)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.W_a[i].T @ (z[i + 2] - net.h_a[i]) - net.Q[i].T @ n[i]
        assert torch.all(torch.isclose(a[i], expected))


def test_apical_current_calculation_no_bias(net):
    z = net.forward(torch.FloatTensor([-0.3, 0.1, 0.4]))
    a, _, n = net.calculate_currents(z)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = net.W_a[i].T @ z[i + 2] - net.Q[i].T @ n[i]
        assert torch.all(torch.isclose(a[i], expected))


@pytest.mark.parametrize("which", ["W_a", "W_b", "Q", "M"])
def test_initial_params_same_when_torch_seed_is_same(which: str):
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = LinearBioPCN(dims)

    old = [_.clone().detach() for _ in getattr(net, which)]

    torch.manual_seed(seed)
    net = LinearBioPCN(dims)

    new = [_.clone().detach() for _ in getattr(net, which)]

    for crt_old, crt_new in zip(old, new):
        assert torch.allclose(crt_old, crt_new)


@pytest.mark.parametrize("which", ["W_a", "W_b", "Q", "M"])
def test_initial_params_change_for_subsequent_calls_if_seed_not_reset(which: str):
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = LinearBioPCN(dims)

    old = [_.clone().detach() for _ in getattr(net, which)]

    net = LinearBioPCN(dims)
    new = [_.clone().detach() for _ in getattr(net, which)]

    for crt_old, crt_new in zip(old, new):
        assert not torch.any(torch.isclose(crt_old, crt_new))


def test_initial_biases_are_zero(net):
    for h in net.h_a + net.h_b:
        assert torch.all(torch.isclose(h, torch.FloatTensor([0])))


def test_z_grad(net):
    z = net.forward(torch.FloatTensor([0.2, -0.5, 0.3]))
    a, b, n = net.calculate_currents(z)
    net.calculate_z_grad(z, a, b, n)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        hidden_apical = net.g_a[i] * a[i]
        hidden_basal = net.g_b[i] * b[i]
        hidden_lateral = net.c_m[i] * net.M[i] @ z[i + 1]
        hidden_leak = net.l_s[i] * z[i + 1]

        expected = hidden_apical + hidden_basal - hidden_lateral - hidden_leak

        assert torch.all(torch.isclose(z[i + 1].grad, -expected))


def test_pc_loss_function(net):
    z = net.forward(torch.FloatTensor([0.2, -0.5, 0.3]))
    loss = net.pc_loss(z).item()

    expected = 0
    D = len(net.pyr_dims) - 2
    for i in range(D):
        err_apical = z[i + 2] - net.W_a[i] @ z[i + 1] - net.h_a[i]
        apical = 0.5 * net.g_a[i] * torch.linalg.norm(err_apical) ** 2

        err_basal = z[i + 1] - net.W_b[i] @ z[i] - net.h_b[i]
        basal = 0.5 * net.g_b[i] * torch.linalg.norm(err_basal) ** 2

        expected += (apical + basal).item()

    assert loss == pytest.approx(expected)


def test_apical_weight_gradient(net):
    net.g_a[0] = 1.0
    net.g_a[1] = 1.0

    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = torch.outer(ns.z[i + 2] - net.h_a[i], ns.z[i + 1]) - net.W_a[i]
        assert torch.all(torch.isclose(net.W_a[i].grad, -expected))


def test_apical_weight_gradient_scaling_with_apical_conductance(net):
    net.g_a[0] = 1.0
    net.g_a[1] = 1.0

    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    old_W_a_grads = [_.grad.clone().detach() for _ in net.W_a]

    # if i don't update any other variables, the delta W should be prop to g
    net.g_a[0] = 0.5
    net.g_a[1] = 1.3

    net.calculate_weight_grad(ns)

    for old, new, g in zip(old_W_a_grads, net.W_a, net.g_a):
        assert torch.all(torch.isfinite(new.grad))
        assert torch.all(torch.isclose(new.grad, g * old))


def test_basal_weight_gradient(net):
    x = torch.FloatTensor([-0.4, 0.5, 0.3])
    y = torch.FloatTensor([0.2, 0.3])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        plateau = net.g_a[i] * torch.outer(ns.a[i], ns.z[i])
        lateral = net.c_m[i] * net.M[i] @ ns.z[i + 1]
        self = (net.l_s[i] - net.g_b[i]) * ns.z[i + 1]
        hebbian = torch.outer(self + lateral, ns.z[i])

        assert torch.all(torch.isclose(-net.W_b[i].grad, plateau - hebbian))


def test_interneuron_weight_gradient(net):
    x = torch.FloatTensor([-0.4, 0.5, 0.3])
    y = torch.FloatTensor([0.2, 0.3])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = torch.outer(ns.n[i], ns.z[i + 1]) - net.Q[i]
        assert torch.all(torch.isclose(net.Q[i].grad, -net.g_a[i] * expected))


def test_lateral_weight_gradient(net):
    x = torch.FloatTensor([-0.4, 0.5, 0.3])
    y = torch.FloatTensor([0.2, 0.3])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        expected = torch.outer(ns.z[i + 1], ns.z[i + 1]) - net.M[i]
        assert torch.all(torch.isclose(net.M[i].grad, -net.c_m[i] * expected))


def test_on_shell_after_relax(net):
    # i.e., z reaches stationarity
    x = torch.FloatTensor([-0.4, 0.5, 0.3])
    y = torch.FloatTensor([0.2, 0.3])
    net.z_it = 1000
    ns = net.relax(x, y)

    D = len(net.pyr_dims) - 2
    for i in range(D):
        apical = net.g_a[i] * ns.a[i]
        basal = net.g_b[i] * ns.b[i]
        lateral = net.c_m[i] * net.M[i] @ ns.z[i + 1]
        leak = net.l_s[i] * ns.z[i + 1]
        rhs = apical + basal - lateral - leak

        assert torch.max(torch.abs(rhs)) < 1e-5


@pytest.mark.parametrize("which", ["g_a", "g_b", "c_m", "l_s"])
def test_allow_tensor_conductances_in_constructor(which):
    kwargs = {which: torch.FloatTensor([1.3, 2.5])}
    pcn = LinearBioPCN([2, 5, 4, 3], **kwargs)
    assert getattr(pcn, which).shape == (2,)


@pytest.mark.parametrize("which", ["g_a", "g_b", "c_m", "l_s"])
def test_allow_scalar_tensor_conductances_in_constructor(which):
    kwargs = {which: torch.FloatTensor([1.3])}
    pcn = LinearBioPCN([2, 5, 4, 3], **kwargs)
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
    cpcn = LinearBioPCN(dims, g_a=g_a, g_b=g_b, c_m=0, l_s=g_b)

    # match the weights
    D = len(dims) - 2
    for i in range(D):
        cpcn.W_a[i] = pcn.W[i + 1].clone().detach()
        cpcn.W_b[i] = pcn.W[i].clone().detach()

    # pass some data through the network to set the neural activities
    ns1 = pcn.relax(torch.FloatTensor([-0.3, 0.2]), torch.FloatTensor([0.4, -0.2, 0.5]))

    # now calculate and compare loss
    pcn_loss = pcn.loss(ns1.z).item()
    cpcn_loss = cpcn.pc_loss(ns1.z).item()

    assert pcn_loss == pytest.approx(cpcn_loss)


def test_init_params_with_numpy_scalar():
    net = LinearBioPCN([2, 3, 4], g_a=np.asarray([0.5]))


def test_to_returns_self(net):
    assert net.to(torch.device("cpu")) is net


def test_repr(net):
    s = repr(net)

    assert s.startswith("LinearBioPCN(")
    assert s.endswith(")")


def test_str(net):
    s = str(net)

    assert s.startswith("LinearBioPCN(")
    assert s.endswith(")")


def test_a_size_when_inter_different_from_pyr(net_inter_dims):
    net = net_inter_dims
    ns = net.relax(torch.FloatTensor([0.2, 0.3, 0.4]), torch.FloatTensor([-0.5, 0.5]))
    D = len(net.inter_dims)
    for i in range(D):
        dim = net.pyr_dims[i + 1]
        assert len(ns.a[i]) == dim


def test_b_size_when_inter_different_from_pyr(net_inter_dims):
    net = net_inter_dims
    ns = net.relax(torch.FloatTensor([0.2, 0.3, 0.4]), torch.FloatTensor([-0.5, 0.5]))
    D = len(net.inter_dims)
    for i in range(D):
        dim = net.pyr_dims[i + 1]
        assert len(ns.b[i]) == dim


def test_n_size_when_inter_different_from_pyr(net_inter_dims):
    net = net_inter_dims
    ns = net.relax(torch.FloatTensor([0.2, 0.3, 0.4]), torch.FloatTensor([-0.5, 0.5]))
    D = len(net.inter_dims)
    for i in range(D):
        dim = net.inter_dims[i]
        assert len(ns.n[i]) == dim


def test_run_on_batch(net):
    x = torch.FloatTensor([[-0.2, -0.3, 0.5], [0.1, 0.5, -1.2]])
    y = torch.FloatTensor([[1.2, -0.5], [-0.3, -0.4]])
    ns_batch = net.relax(x, y)

    for i in range(len(x)):
        ns = net.relax(x[i], y[i])
        for old_z, new_z in zip(ns_batch.z, ns.z):
            assert torch.allclose(old_z[i], new_z)


def test_pc_loss_on_batch(net):
    x = torch.FloatTensor([[-0.2, -0.3, 0.5], [0.1, 0.5, -1.2]])
    y = torch.FloatTensor([[1.2, -0.5], [-0.3, -0.4]])
    ns = net.relax(x, y)
    batch_loss = net.pc_loss(ns.z)

    loss = 0
    for i in range(len(x)):
        ns = net.relax(x[i], y[i])
        loss += net.pc_loss(ns.z)

    assert batch_loss.item() == pytest.approx(loss.item() / len(x))


def test_weight_grad_on_batch(net):
    x = torch.FloatTensor([[-0.2, -0.3, 0.5], [0.1, 0.5, -1.2]])
    y = torch.FloatTensor([[1.2, -0.5], [-0.3, -0.4]])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    batch_W_grad = [_.grad.clone().detach() for _ in net.W_a + net.W_b]

    expected_grads = [torch.zeros_like(_) for _ in batch_W_grad]
    for i in range(len(x)):
        ns = net.relax(x[i], y[i])
        net.calculate_weight_grad(ns)
        for k, new_W in enumerate(net.W_a + net.W_b):
            expected_grads[k] += new_W.grad

    for old, new in zip(batch_W_grad, expected_grads):
        assert torch.allclose(old, new / len(x))


def test_bias_grad_on_batch(net):
    x = torch.FloatTensor([[-0.2, -0.3, 0.5], [0.1, 0.5, -1.2]])
    y = torch.FloatTensor([[1.2, -0.5], [-0.3, -0.4]])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    batch_h_grad = [_.grad.clone().detach() for _ in net.h_a + net.h_b]

    expected_grads = [torch.zeros_like(_) for _ in batch_h_grad]
    for i in range(len(x)):
        ns = net.relax(x[i], y[i])
        net.calculate_weight_grad(ns)
        for k, new_h in enumerate(net.h_a + net.h_b):
            expected_grads[k] += new_h.grad

    for old, new in zip(batch_h_grad, expected_grads):
        assert torch.allclose(old, new / len(x))


def test_weights_change_when_optimizing_slow_parameters(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    old_Ws = [_.clone().detach() for _ in net.W_a + net.W_b + net.Q + net.M]

    optimizer = torch.optim.Adam(net.slow_parameters(), lr=1.0)
    ns = net.relax(x0, y0)
    net.calculate_weight_grad(ns)
    optimizer.step()

    new_Ws = net.W_a + net.W_b + net.Q + net.M
    for old_W, new_W in zip(old_Ws, new_Ws):
        assert not torch.any(torch.isclose(old_W, new_W))


def test_biases_change_when_optimizing_slow_parameters(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    old_hs = [_.clone().detach() for _ in net.h_a + net.h_b]

    optimizer = torch.optim.Adam(net.slow_parameters(), lr=1.0)
    ns = net.relax(x0, y0)
    net.calculate_weight_grad(ns)
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
        ns = net.relax(x, y)
        net.calculate_weight_grad(ns)
        optimizer.step()

    for W in net.W_a + net.W_b + net.Q + net.M:
        assert torch.all(torch.isfinite(W))

    for h in net.h_a + net.h_b:
        assert torch.all(torch.isfinite(h))

    for z in ns.z:
        assert torch.all(torch.isfinite(z))


def linear_cpcn_loss(
    net: LinearBioPCN, z: list, reduction: str = "sum"
) -> torch.Tensor:
    D = len(net.inter_dims)
    batch_size = 1 if z[0].ndim == 1 else len(z[0])
    loss = torch.zeros(batch_size)
    batch_outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
    for i in range(D):
        crt_z = z[i + 1]

        mu = z[i] @ net.W_b[i].T + net.h_b[i]
        error = crt_z - mu
        basal = 0.5 * net.g_b[i] * (error ** 2).sum(dim=-1)

        diff = z[i + 2] - net.h_a[i]
        cross_term = (diff * (crt_z @ net.W_a[i].T)).sum(dim=-1)
        wa_reg = net.rho[i] * torch.trace(net.W_a[i] @ net.W_a[i].T)
        apical0 = (diff ** 2).sum(dim=-1) - 2 * cross_term + wa_reg
        apical = 0.5 * net.g_a[i] * apical0

        # need to flip this for Q because we're maximizing!!
        # but note that this would lead to incorrect dynamics for the latents!
        # (we get away with it because we don't use this objective for z dynamics)
        z_cons = batch_outer(crt_z, crt_z) - net.rho[i] * torch.eye(crt_z.shape[-1])
        q_prod = net.Q[i].T @ net.Q[i]
        constraint0 = (q_prod @ z_cons).diagonal(dim1=-1, dim2=-2).sum(dim=-1)
        constraint = -0.5 * net.g_a[i] * constraint0

        alpha = net.l_s[i] - net.g_b[i]
        z_reg = 0.5 * alpha * (crt_z ** 2).sum(dim=-1)

        m_prod = (crt_z * (crt_z @ net.M[i].T)).sum(dim=-1)
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
    ns = net.relax(data.x, data.y)
    loss = linear_cpcn_loss(net, ns.z, reduction="none")

    for i, (crt_x, crt_y) in enumerate(zip(data.x, data.y)):
        ns = net.relax(crt_x, crt_y)
        crt_loss = linear_cpcn_loss(net, ns.z)

        assert loss[i].item() == pytest.approx(crt_loss.item())


def test_cpcn_loss_reduction_sum(net, data):
    ns = net.relax(data.x, data.y)
    loss = linear_cpcn_loss(net, ns.z, reduction="sum")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        ns = net.relax(crt_x, crt_y)
        expected += linear_cpcn_loss(net, ns.z)

    assert loss.item() == pytest.approx(expected.item())


def test_cpcn_loss_reduction_mean(net, data):
    ns = net.relax(data.x, data.y)
    loss = linear_cpcn_loss(net, ns.z, reduction="mean")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        ns = net.relax(crt_x, crt_y)
        expected += linear_cpcn_loss(net, ns.z)

    expected /= len(data.x)
    assert loss.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("var", ["W_a", "W_b", "h_a", "h_b", "Q", "M"])
def test_weight_gradients_match_autograd_from_loss(net, var):
    net.z_it = 1000

    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])

    ns = net.relax(x0, y0)
    net.calculate_weight_grad(ns)

    manual_grads = [_.grad.clone().detach() for _ in getattr(net, var)]

    for param in net.slow_parameters():
        param.requires_grad_()
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    loss = linear_cpcn_loss(net, ns.z)
    loss.backward()

    loss_grads = [_.grad.clone().detach() for _ in getattr(net, var)]

    for from_manual, from_loss in zip(manual_grads, loss_grads):
        assert torch.allclose(from_manual, from_loss, rtol=1e-2, atol=1e-5)


def test_relax_returns_empty_profile_namespace_by_default(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.relax(x, y)

    assert len(ns.profile.__dict__) == 0


def test_relax_loss_profile_is_sequence_of_correct_length(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.relax(x, y, pc_loss_profile=True)

    assert len(ns.profile.pc_loss) == net.z_it


def test_relax_loss_profile_is_sequence_of_positive_numbers(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.relax(x, y, pc_loss_profile=True)

    assert min(ns.profile.pc_loss) > 0


def test_relax_latent_profile_batch_index_added(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.relax(x, y, latent_profile=True)
    for var in ns.profile.__dict__:
        crt_data = getattr(ns.profile, var)
        for x in crt_data:
            assert x.ndim == 3
            assert x.shape[1] == 1


def test_relax_latent_profile_has_z_a_b_n(net):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.relax(x, y, latent_profile=True)

    assert set(ns.profile.__dict__.keys()) == {"z", "a", "b", "n"}


@pytest.mark.parametrize("var", ["z", "a", "b", "n"])
def test_relax_latent_profile_has_correct_length(net, var):
    x = torch.FloatTensor([0.2, -0.5, 0.3])
    y = torch.FloatTensor([0.5, -0.3])
    ns = net.relax(x, y, latent_profile=True)
    crt_data = getattr(ns.profile, var)
    for x in crt_data:
        assert len(x) == net.z_it


def test_relax_latent_profile_first_layer_of_z_is_input(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y, latent_profile=True)
    assert torch.max(torch.abs(ns.profile.z[0] - x)) < 1e-5


def test_relax_latent_profile_last_layer_of_z_is_output(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y, latent_profile=True)
    assert torch.max(torch.abs(ns.profile.z[-1] - y)) < 1e-5


@pytest.mark.parametrize("var", ["z", "a", "b", "n"])
def test_relax_latent_profile_row_matches_shorter_run(net, var):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y, latent_profile=True)

    new_it = net.z_it // 2
    net.z_it = new_it
    ns_short = net.relax(x, y)

    long_data = getattr(ns.profile, var)
    for i, x in enumerate(long_data):
        short_data = getattr(ns_short, var)
        assert torch.allclose(short_data[i], x[new_it - 1, 0])


@pytest.mark.parametrize("var", ["z", "a", "b", "n"])
def test_relax_latent_profile_row_matches_shorter_run_interdims(net_inter_dims, var):
    net = net_inter_dims
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y, latent_profile=True)

    new_it = net.z_it // 2
    net.z_it = new_it
    ns_short = net.relax(x, y)

    long_data = getattr(ns.profile, var)
    for i, x in enumerate(long_data):
        short_data = getattr(ns_short, var)
        assert torch.allclose(short_data[i], x[new_it - 1, 0])


@pytest.mark.parametrize("var", ["z", "a", "b", "n"])
def test_relax_latent_profile_with_batch(net, var):
    x = torch.FloatTensor([[-0.1, 0.2, 0.4], [0.5, 0.3, 0.2]])
    y = torch.FloatTensor([[0.3, -0.4], [0.1, 0.2]])
    ns = net.relax(x, y, latent_profile=True)

    long_data = getattr(ns.profile, var)
    for k in range(len(x)):
        crt_x = x[k]
        crt_y = y[k]
        crt_ns = net.relax(crt_x, crt_y, latent_profile=True)

        short_data = getattr(crt_ns.profile, var)
        for x1, x2 in zip(long_data, short_data):
            assert torch.allclose(x1[:, [k], :], x2, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("var", ["a", "b", "n"])
def test_currents_are_consistent_with_z_after_forward_constraint(net, var):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y)

    new = SimpleNamespace()
    new.a, new.b, new.n = net.calculate_currents(ns.z)

    for a, b in zip(getattr(ns, var), getattr(new, var)):
        assert torch.allclose(a, b)


def test_forward_always_returns_all_layers_of_z(net):
    z = net.forward(torch.FloatTensor([0.1, 0.2, 0.3]))
    assert len(z) == len(net.pyr_dims)


def test_loss_reduction_none(net, data):
    ns = net.relax(data.x, data.y)
    loss = net.pc_loss(ns.z, reduction="none")

    for i, (crt_x, crt_y) in enumerate(zip(data.x, data.y)):
        ns = net.relax(crt_x, crt_y)
        crt_loss = net.pc_loss(ns.z)

        assert loss[i].item() == pytest.approx(crt_loss.item())


def test_loss_reduction_sum(net, data):
    ns = net.relax(data.x, data.y)
    loss = net.pc_loss(ns.z, reduction="sum")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        ns = net.relax(crt_x, crt_y)
        expected += net.pc_loss(ns.z)

    assert loss.item() == pytest.approx(expected.item())


def test_loss_reduction_mean(net, data):
    ns = net.relax(data.x, data.y)
    loss = net.pc_loss(ns.z, reduction="mean")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        ns = net.relax(crt_x, crt_y)
        expected += net.pc_loss(ns.z)

    expected /= len(data.x)
    assert loss.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("var", ["W_a", "W_b", "h_a", "h_b", "Q", "M"])
@pytest.mark.parametrize("red", ["sum", "mean"])
def test_weight_gradients_match_autograd_from_loss_batch(net, data, var, red):
    net.z_it = 1000

    ns = net.relax(data.x, data.y)
    net.calculate_weight_grad(ns, reduction=red)

    manual_grads = [_.grad.clone().detach() for _ in getattr(net, var)]

    for param in net.slow_parameters():
        param.requires_grad_()
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    loss = linear_cpcn_loss(net, ns.z, reduction=red)
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
    net = LinearBioPCN(
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
    ns_old = net.relax(x, y)

    net.rho[0] = 0.7
    net.rho[1] = 1.5
    ns_new = net.relax(x, y)

    for old, new in zip(ns_old.z, ns_new.z):
        assert torch.allclose(old, new)


def test_q_gradient_with_nontrivial_constraint(net_nontrivial_constraint, data):
    net = net_nontrivial_constraint
    ns = net.relax(data.x, data.y)
    net.calculate_weight_grad(ns)

    outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
    for i in range(len(net.inter_dims)):
        expected = net.g_a[i] * (
            net.rho[i] * net.Q[i] - outer(ns.n[i], ns.z[i + 1])
        ).mean(dim=0)
        assert torch.allclose(net.Q[i].grad, expected)


def test_q_gradient_with_nontrivial_constraint_vs_autograd(net_nontrivial_constraint):
    net = net_nontrivial_constraint

    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    manual_grads = [_.grad.clone().detach() for _ in net.Q]

    for param in net.slow_parameters():
        param.requires_grad_()
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    loss = linear_cpcn_loss(net, ns.z)
    loss.backward()

    loss_grads = [_.grad.clone().detach() for _ in net.Q]

    for from_manual, from_loss in zip(manual_grads, loss_grads):
        assert torch.allclose(from_manual, from_loss)


def test_wa_gradient_with_nontrivial_constraint(net_nontrivial_constraint, data):
    net = net_nontrivial_constraint
    ns = net.relax(data.x, data.y)
    net.calculate_weight_grad(ns)

    outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
    for i in range(len(net.inter_dims)):
        diff = ns.z[i + 2] - net.h_a[i]
        expected = net.g_a[i] * (
            net.rho[i] * net.W_a[i] - outer(diff, ns.z[i + 1])
        ).mean(dim=0)
        assert torch.allclose(net.W_a[i].grad, expected)


def test_wa_gradient_with_nontrivial_constraint_vs_autograd(net_nontrivial_constraint):
    net = net_nontrivial_constraint

    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y)
    net.calculate_weight_grad(ns)

    manual_grads = [_.grad.clone().detach() for _ in net.W_a]

    for param in net.slow_parameters():
        param.requires_grad_()
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    loss = linear_cpcn_loss(net, ns.z)
    loss.backward()

    loss_grads = [_.grad.clone().detach() for _ in net.W_a]

    for from_manual, from_loss in zip(manual_grads, loss_grads):
        assert torch.allclose(from_manual, from_loss)


@pytest.mark.parametrize("var", ["W_a", "W_b", "Q", "M"])
def test_init_scale(var):
    scale_name = var.lower().replace("_", "") + "0_scale"

    seed = 1
    scale = 1.35
    dims = [5, 3, 4, 3]
    torch.manual_seed(seed)
    net1 = LinearBioPCN(dims)

    torch.manual_seed(seed)
    net2 = LinearBioPCN(dims, **{scale_name: scale})

    for theta1, theta2 in zip(getattr(net1, var), getattr(net2, var)):
        assert torch.allclose(theta2, scale * theta1)


def test_default_c_m_is_zero():
    net = LinearBioPCN([2, 5, 3])
    assert net.c_m[0] == 0


@pytest.mark.parametrize("var", ["W_a", "W_b", "Q", "M"])
def test_initial_param_scale_depends_on_out_only(var):
    torch.manual_seed(2)
    n = 500
    m = 50
    p = 25
    net = LinearBioPCN([m, n, m], inter_dims=[p])

    value = getattr(net, var)[0]
    sigma = torch.std(value).item()

    n_in, n_out = value.shape
    # sqrt(3) from stdev of uniform distribution
    expected = np.sqrt(1 / n_out) / np.sqrt(3)

    n_elem = value.numel()
    tol = 4 / np.sqrt(n_elem)

    assert sigma == pytest.approx(expected, rel=tol)


@pytest.mark.parametrize("var", ["W_a", "W_b", "Q", "M"])
def test_initial_param_scale_xavier(var):
    torch.manual_seed(3)
    n = 500
    m = 50
    p = 25
    net = LinearBioPCN([m, n, m], inter_dims=[p], init_scale_type="xavier_uniform")

    value = getattr(net, var)[0]
    sigma = torch.std(value).item()

    n_in, n_out = value.shape
    # sqrt(3) from stdev of uniform distribution
    expected = np.sqrt(6 / (n_in + n_out)) / np.sqrt(3)

    n_elem = value.numel()
    tol = 4 / np.sqrt(n_elem)

    assert sigma == pytest.approx(expected, rel=tol)


def test_slow_parameter_groups_is_iterable_of_dicts_each_with_params_member(net):
    param_groups = net.slow_parameter_groups()
    for d in param_groups:
        assert "params" in d


def test_slow_parameter_groups_returns_dicts_with_name_key(net):
    param_groups = net.slow_parameter_groups()
    for d in param_groups:
        assert "name" in d


def test_slow_parameter_groups_lists_the_same_parameters_as_slow_parameters(net):
    params1 = set(net.slow_parameters())
    params2 = set(sum([_["params"] for _ in net.slow_parameter_groups()], []))

    for x in params1:
        assert x in params2, "element of params missing from param_groups"
    for x in params2:
        assert x in params1, "element of param_groups missing from params"


@pytest.mark.parametrize("var", ["W_a", "W_b", "Q", "M", "h_a", "h_b"])
def test_slow_parameter_groups_contains_expected_names(net, var):
    params = net.slow_parameter_groups()
    names = [_["name"] for _ in params]

    assert var in names


def test_slow_parameters_no_bias_a(net_no_bias_a):
    params = net_no_bias_a.slow_parameter_groups()
    names = [_["name"] for _ in params]

    assert "h_a" not in names
    assert "h_b" in names


def test_slow_parameters_no_bias_b(net_no_bias_b):
    params = net_no_bias_b.slow_parameter_groups()
    names = [_["name"] for _ in params]

    assert "h_b" not in names
    assert "h_a" in names


def test_from_pcn_with_match_weights():
    torch.manual_seed(2)

    dims = [2, 4, 3, 7, 3]
    variances = [0.5, 1.2, 0.7, 0.3]

    pcn = PCNetwork(dims, variances=variances, activation=lambda _: _)
    cpcn = LinearBioPCN.from_pcn(pcn, match_weights=True)

    # pass some data through the network to set the neural activities
    ns1 = pcn.relax(torch.FloatTensor([-0.3, 0.2]), torch.FloatTensor([0.4, -0.2, 0.5]))

    # now calculate and compare loss
    pcn_loss = pcn.loss(ns1.z).item()
    cpcn_loss = cpcn.pc_loss(ns1.z).item()

    assert pcn_loss == pytest.approx(cpcn_loss)


def test_from_pcn_without_match_weights():
    torch.manual_seed(2)

    dims = [2, 4, 3, 7, 3]
    variances = [0.5, 1.2, 0.7, 0.3]

    pcn = PCNetwork(dims, variances=variances, activation=lambda _: _)
    cpcn = LinearBioPCN.from_pcn(pcn)

    # ensure weights differ
    for i in range(len(dims) - 2):
        assert torch.max(torch.abs(cpcn.W_a[i] - pcn.W[i + 1])) > 0.01
        assert torch.max(torch.abs(cpcn.W_b[i] - pcn.W[i])) > 0.01


def test_from_pcn_raises_if_nontrivial_activation():
    pcn = PCNetwork([2, 5, 10], activation=torch.tanh)
    with pytest.raises(ValueError):
        LinearBioPCN.from_pcn(pcn)


def test_from_pcn_does_not_raise_if_nontrival_activation_but_copy_activation_false():
    pcn = PCNetwork([2, 5, 10], activation=torch.tanh)
    LinearBioPCN.from_pcn(pcn, check_activation=False)


def test_from_pcn_copies_over_Q_for_constrained_pcn_if_match_weights_is_true():
    torch.manual_seed(3)

    dims = [2, 4, 3, 8]

    pcn = PCNetwork(dims, activation=lambda _: _, constrained=True)
    cpcn = LinearBioPCN.from_pcn(pcn, match_weights=True)

    for i in range(len(dims) - 2):
        assert torch.allclose(cpcn.Q[i], pcn.Q[i])


def test_from_pcn_copies_over_rho_for_constrained_pcn():
    torch.manual_seed(3)

    dims = [2, 4, 3, 8]

    rho = [0.3, 0.5]
    pcn = PCNetwork(dims, activation=lambda _: _, constrained=True, rho=rho)
    cpcn = LinearBioPCN.from_pcn(pcn)

    np.testing.assert_allclose(cpcn.rho, pcn.rho)


def test_from_pcn_leaves_rho_untouched_for_constrained_pcn():
    torch.manual_seed(3)

    dims = [2, 4, 3, 8]

    rho = [0.3, 0.5]
    pcn = PCNetwork(dims, activation=lambda _: _, constrained=False, rho=rho)
    cpcn = LinearBioPCN.from_pcn(pcn)

    assert torch.max(torch.abs(cpcn.rho - pcn.rho)) > 0.1


def test_from_pcn_copies_over_biases_when_match_weights_is_true():
    torch.manual_seed(2)

    dims = [2, 4, 3, 7, 3]
    variances = [0.5, 1.2, 0.7, 0.3]

    pcn = PCNetwork(dims, variances=variances, activation=lambda _: _)
    with torch.no_grad():
        for h in pcn.h:
            h.uniform_()

    cpcn = LinearBioPCN.from_pcn(pcn, match_weights=True)

    # pass some data through the network to set the neural activities
    ns1 = pcn.relax(torch.FloatTensor([-0.3, 0.2]), torch.FloatTensor([0.4, -0.2, 0.5]))

    # now calculate and compare loss
    pcn_loss = pcn.loss(ns1.z).item()
    cpcn_loss = cpcn.pc_loss(ns1.z).item()

    assert pcn_loss == pytest.approx(cpcn_loss)


def test_from_pcn_copies_over_state_of_bias():
    torch.manual_seed(3)

    pcn = PCNetwork([2, 5, 3, 7], activation=lambda _: _, bias=False)
    cpcn = LinearBioPCN.from_pcn(pcn)

    assert not cpcn.bias_a
    assert not cpcn.bias_b


def test_from_pcn_additional_kwargs_go_to_biopcn_constructor():
    torch.manual_seed(3)

    rho = torch.FloatTensor([0.5, 0.7])
    pcn = PCNetwork([2, 5, 3, 7], activation=lambda _: _)
    cpcn = LinearBioPCN.from_pcn(pcn, rho=rho)

    np.testing.assert_allclose(cpcn.rho, rho)


def test_from_pcn_additional_kwargs_override_conductances():
    torch.manual_seed(3)

    pcn = PCNetwork([2, 5, 3, 7], activation=lambda _: _)
    g_a = 0.75
    cpcn = LinearBioPCN.from_pcn(pcn, g_a=g_a)

    assert torch.max(torch.abs(cpcn.g_a - g_a)) < 1e-5


def test_from_pcn_additional_kwargs_override_no_bias():
    torch.manual_seed(3)

    pcn = PCNetwork([2, 5, 3, 7], activation=lambda _: _, bias=False)
    cpcn = LinearBioPCN.from_pcn(pcn, match_weights=True, bias_a=True)

    assert cpcn.bias_a
    assert not cpcn.bias_b


def test_from_pcn_additional_kwargs_override_yes_bias():
    torch.manual_seed(3)

    dims = [2, 5, 3, 7]
    pcn = PCNetwork(dims, activation=lambda _: _, bias=True)
    cpcn = LinearBioPCN.from_pcn(pcn, match_weights=True, bias_a=False)

    assert not cpcn.bias_a
    assert cpcn.bias_b

    for i in range(len(dims) - 2):
        assert torch.allclose(cpcn.h_b[i], pcn.h[i])


@pytest.mark.parametrize("var", ["g_a", "g_b", "l_s", "c_m", "rho"])
def test_parameters_not_tensors(net_nontrivial_constraint, var):
    net = net_nontrivial_constraint
    for x in getattr(net, var):
        assert not torch.is_tensor(x)


@pytest.mark.parametrize("var", ["g_a", "g_b", "l_s", "c_m", "rho"])
def test_parameters_not_tensors_even_if_fed_tensors(var):
    net = LinearBioPCN([2, 5, 3, 4], **{var: torch.FloatTensor([0.3, 1.2])})
    for x in getattr(net, var):
        assert not torch.is_tensor(x)


def test_relax_works_when_called_inside_no_grad(net):
    with torch.no_grad():
        net.relax(torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4]))
