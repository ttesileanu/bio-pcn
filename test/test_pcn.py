import pytest

from types import SimpleNamespace

from cpcn.pcn import PCNetwork

import torch


@pytest.fixture
def net():
    torch.manual_seed(20392)
    net = PCNetwork([3, 4, 2])
    return net


@pytest.fixture
def net_nb():
    torch.manual_seed(127824)
    net = PCNetwork([3, 4, 2], bias=False)
    return net


def test_number_of_layers(net):
    assert len(net.W) == 2
    assert len(net.h) == 2


def test_weight_sizes(net):
    assert net.W[0].shape == (4, 3)
    assert net.W[1].shape == (2, 4)


def test_z_sizes(net):
    z = net.forward(torch.FloatTensor([0.3, -0.2, 0.5]))
    assert [len(_) for _ in z] == [3, 4, 2]


def test_all_zs_nonzero_after_relax(net):
    ns = net.relax(torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4]))

    for z in ns.z:
        assert torch.max(torch.abs(z)) > 1e-4


def test_all_zs_change_during_relax(net):
    # set some starting values for z
    old_z = net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))
    ns = net.relax(torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4]))

    for old, new in zip(old_z, ns.z):
        assert not torch.all(torch.isclose(old, new))


def test_all_zs_not_nonzero_after_forward(net):
    z = net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for crt_z in z:
        assert torch.max(torch.abs(crt_z)) > 1e-4


def test_all_zs_change_during_forward(net):
    # set some starting values for z
    old_z = net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))
    new_z = net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for old, new in zip(old_z, new_z):
        assert not torch.any(torch.isclose(old, new))


def test_forward_result_is_stationary_point_of_relax(net):
    x0 = torch.FloatTensor([0.5, -0.7, 0.2])
    old_z = net.forward(x0)

    ns = net.relax(old_z[0], old_z[-1])

    for old, new in zip(old_z, ns.z):
        assert torch.allclose(old, new)


def test_weights_and_biases_change_when_optimizing_parameters(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    old_Ws = [_.clone().detach() for _ in net.W]
    old_bs = [_.clone().detach() for _ in net.h]

    optimizer = torch.optim.Adam(net.parameters(), lr=1.0)
    ns = net.relax(x0, y0)

    optimizer.zero_grad()
    loss = net.loss(ns.z)
    loss.backward()
    optimizer.step()

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_bs, net.W, net.h):
        assert not torch.any(torch.isclose(old_W, new_W))
        assert not torch.any(torch.isclose(old_b, new_b))


def test_loss_is_nonzero_after_relax(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    ns = net.relax(x0, y0)

    assert net.loss(ns.z).abs().item() > 1e-6


def test_forward_does_not_change_weights_and_biases(net):
    old_Ws = [_.clone().detach() for _ in net.W]
    old_hs = [_.clone().detach() for _ in net.h]
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_hs, net.W, net.h):
        assert torch.allclose(old_W, new_W)
        assert torch.allclose(old_b, new_b)


def test_relax_does_not_change_weights_and_biases(net):
    old_Ws = [_.clone().detach() for _ in net.W]
    old_hs = [_.clone().detach() for _ in net.h]
    net.relax(torch.FloatTensor([0.3, -0.4, 0.2]), torch.FloatTensor([-0.2, 0.2]))

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_hs, net.W, net.h):
        assert torch.allclose(old_W, new_W)
        assert torch.allclose(old_b, new_b)


def test_loss_does_not_change_weights_and_biases(net):
    # ensure the z variables have valid values assigned to them
    z = net.forward(torch.FloatTensor([0.1, 0.2, 0.3]))

    old_Ws = [_.clone().detach() for _ in net.W]
    old_hs = [_.clone().detach() for _ in net.h]
    net.loss(z)

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_hs, net.W, net.h):
        assert torch.allclose(old_W, new_W)
        assert torch.allclose(old_b, new_b)


def test_no_nan_or_inf_after_a_few_learning_steps(net):
    torch.manual_seed(0)

    optimizer = torch.optim.Adam(net.parameters())
    for i in range(4):
        x = torch.Tensor(3).uniform_()
        y = torch.Tensor(2).uniform_()
        ns = net.relax(x, y)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    for W, h in zip(net.W, net.h):
        assert torch.all(torch.isfinite(W))
        assert torch.all(torch.isfinite(h))

    for z in ns.z:
        assert torch.all(torch.isfinite(z))


def test_forward_output_depends_on_input(net):
    y1 = [_.detach().clone() for _ in net.forward(torch.FloatTensor([0.1, 0.3, -0.2]))]
    y2 = [_.detach().clone() for _ in net.forward(torch.FloatTensor([-0.5, 0.1, 0.2]))]

    for a, b in zip(y1, y2):
        assert not torch.allclose(a, b)


def test_forward_sets_first_element_of_z_to_input_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    z = net.forward(x0)
    assert torch.allclose(z[0], x0)


def test_relax_sets_first_element_of_z_to_input_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    y0 = torch.FloatTensor([0.5, -0.2])
    ns = net.relax(x0, y0)
    assert torch.allclose(ns.z[0], x0)


def test_relax_sets_last_element_of_z_to_output_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    y0 = torch.FloatTensor([0.5, -0.2])
    ns = net.relax(x0, y0)
    assert torch.allclose(ns.z[-1], y0)


def test_initialize_values_same_when_torch_seed_is_same():
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    old_Ws = [_.clone().detach() for _ in net.W]
    old_bs = [_.clone().detach() for _ in net.h]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    new_Ws = [_.clone().detach() for _ in net.W]
    new_bs = [_.clone().detach() for _ in net.h]

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_bs, new_Ws, new_bs):
        assert torch.allclose(old_W, new_W)
        assert torch.allclose(old_b, new_b)


def test_initialize_weights_change_for_subsequent_calls_if_seed_not_reset():
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    var1 = [_.clone().detach() for _ in net.W]

    net = PCNetwork(dims)
    var2 = [_.clone().detach() for _ in net.W]

    for old, new in zip(var1, var2):
        assert not torch.any(torch.isclose(old, new))


def test_weights_reproducible_for_same_seed_after_learning():
    seed = 321
    dims = [2, 6, 5, 3]

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    # do some learning
    torch.manual_seed(seed)
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.parameters())
    for crt_x, crt_y in zip(x, y):
        ns = net.relax(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    var1 = [_.clone().detach() for _ in net.W]

    # reset and do the learning again
    torch.manual_seed(seed)
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.parameters())
    for crt_x, crt_y in zip(x, y):
        ns = net.relax(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    var2 = [_.clone().detach() for _ in net.W]

    for old, new in zip(var1, var2):
        assert torch.allclose(old, new)


def test_learning_effects_are_different_for_subsequent_runs():
    seed = 321
    dims = [2, 6, 5, 3]

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    # do some learning
    torch.manual_seed(seed)
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.parameters())
    for crt_x, crt_y in zip(x, y):
        ns = net.relax(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    var1 = [_.clone().detach() for _ in net.W]

    # reset and do the learning again -- without resetting random seed this time!
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.parameters())
    for crt_x, crt_y in zip(x, y):
        ns = net.relax(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    var2 = [_.clone().detach() for _ in net.W]

    for old, new in zip(var1, var2):
        assert not torch.allclose(old, new)


def test_training_with_batches_of_size_one():
    seed = 100
    dims = [2, 6, 5, 3]
    variances = [0.5, 1.5, 2.7]
    lr = 0.2

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    # do some learning
    torch.manual_seed(seed)
    net = PCNetwork(dims, variances=variances)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for crt_x, crt_y in zip(x, y):
        ns = net.relax(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    test_x = torch.FloatTensor([0.5, 0.2])
    out = net.forward(test_x)

    # do the same learning with batches of size 1
    torch.manual_seed(seed)
    net = PCNetwork(dims, variances=variances)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for crt_x, crt_y in zip(x, y):
        crt_x_batch = crt_x[None, :]
        crt_y_batch = crt_y[None, :]
        ns = net.relax(crt_x_batch, crt_y_batch)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    test_x = torch.FloatTensor([0.5, 0.2])
    out_batch = net.forward(test_x)

    for crt_out, crt_out_batch in zip(out, out_batch):
        assert torch.allclose(crt_out, crt_out_batch)


def test_training_with_batches_of_nontrivial_size():
    seed = 200
    dims = [2, 6, 5]
    variances = [0.5, 1.5]
    lr = 1e-4
    z_it = 10

    n_samples = 5

    torch.manual_seed(seed)
    x = torch.normal(0, 1, size=(n_samples, dims[0]))
    y = torch.normal(0, 1, size=(n_samples, dims[-1]))

    # do some learning
    torch.manual_seed(seed)
    kwargs = {"variances": variances, "z_it": z_it}
    net = PCNetwork(dims, **kwargs)

    # gradients are averaged over batch samples by default, *almost* equivalent to lower
    # lr when training sample by sample
    optimizer = torch.optim.SGD(net.parameters(), lr=lr / n_samples)
    for crt_x, crt_y in zip(x, y):
        ns = net.relax(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    test_x = torch.FloatTensor([0.5, -0.2])
    out = net.forward(test_x)

    # do the same learning with batches of size 1
    torch.manual_seed(seed)
    net = PCNetwork(dims, **kwargs)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for crt_x1, crt_x2, crt_y1, crt_y2 in zip(x[::2], x[1::2], y[::2], y[1::2]):
        crt_x_batch = torch.vstack((crt_x1, crt_x2))
        crt_y_batch = torch.vstack((crt_y1, crt_y2))
        ns = net.relax(crt_x_batch, crt_y_batch)

        optimizer.zero_grad()
        net.loss(ns.z).backward()
        optimizer.step()

    test_x = torch.FloatTensor([0.5, -0.2])
    out_batch = net.forward(test_x)

    # we don't expect these to be super close -- when we make the steps one by one, the
    # gradients at subsequent batches change due to earlier steps
    for crt_out, crt_out_batch in zip(out, out_batch):
        assert torch.allclose(crt_out, crt_out_batch, rtol=0.01, atol=1e-5)


def test_relax_returns_empty_profile_by_default(net):
    ns = net.relax(torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4]))
    assert len(ns.profile.__dict__) == 0


def test_relax_loss_profile_is_sequence_of_correct_length(net):
    ns = net.relax(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        pc_loss_profile=True,
    )

    assert len(ns.profile.pc_loss) == net.z_it


def test_relax_loss_profile_is_sequence_of_positive_numbers(net):
    ns = net.relax(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        pc_loss_profile=True,
    )

    assert min(ns.profile.pc_loss) > 0


def test_relax_loss_profile_has_last_elem_smaller_than_first(net):
    ns = net.relax(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        pc_loss_profile=True,
    )

    assert ns.profile.pc_loss[-1] < ns.profile.pc_loss[0]


def test_relax_latent_profile_batch_index_added(net):
    ns = net.relax(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        latent_profile=True,
    )
    for z in ns.profile.z:
        assert z.ndim == 3
        assert z.shape[1] == 1


def test_relax_latent_profile_has_correct_length(net):
    ns = net.relax(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        latent_profile=True,
    )
    for z in ns.profile.z:
        assert len(z) == net.z_it


def test_relax_latent_profile_first_layer_is_input(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y, latent_profile=True)
    assert torch.max(torch.abs(ns.profile.z[0] - x)) < 1e-5


def test_relax_latent_profile_last_layer_is_output(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y, latent_profile=True)
    assert torch.max(torch.abs(ns.profile.z[-1] - y)) < 1e-5


def test_relax_latent_profile_row_matches_shorter_run(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns0 = net.relax(x, y, latent_profile=True)

    new_it = 2
    net.z_it = new_it
    ns = net.relax(x, y)

    for i, z in enumerate(ns0.profile.z):
        assert torch.allclose(ns.z[i], z[new_it - 1, 0])


def test_relax_latent_profile_with_batch(net):
    x = torch.FloatTensor([[-0.1, 0.2, 0.4], [0.5, 0.3, 0.2]])
    y = torch.FloatTensor([[0.3, -0.4], [0.1, 0.2]])
    ns = net.relax(x, y, latent_profile=True)

    for k in range(len(x)):
        crt_x = x[k]
        crt_y = y[k]
        crt_ns = net.relax(crt_x, crt_y, latent_profile=True)

        for z1, z2 in zip(ns.profile.z, crt_ns.profile.z):
            assert torch.allclose(z1[:, [k], :], z2, atol=1e-5, rtol=1e-5)


def test_to_returns_self(net):
    assert net.to(torch.device("cpu")) is net


def test_repr(net):
    s = repr(net)

    assert s.startswith("PCNetwork(")
    assert s.endswith(")")


def test_str(net):
    s = str(net)

    assert s.startswith("PCNetwork(")
    assert s.endswith(")")


def test_pc_loss_matches_loss(net):
    ns = net.relax(torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4]))

    assert net.loss(ns.z).item() == pytest.approx(net.pc_loss(ns.z).item())


@pytest.fixture
def data() -> tuple:
    x = torch.FloatTensor([[-0.1, 0.2, 0.4], [0.5, 0.3, 0.2], [-1.0, 2.3, 0.1]])
    y = torch.FloatTensor([[0.3, -0.4], [0.1, 0.2], [-0.5, 1.2]])
    return SimpleNamespace(x=x, y=y)


@pytest.mark.parametrize("red", ["sum", "mean"])
def test_calculate_weight_grad_matches_backward_on_loss(net, red, data):
    ns = net.relax(data.x, data.y)
    net.calculate_weight_grad(ns, reduction=red)

    old_grad = [_.grad.clone().detach() for _ in net.parameters()]

    for param in net.parameters():
        if param.grad is not None:
            param.grad.zero_()

    loss = net.loss(ns.z, reduction=red)
    loss.backward()

    for old, new_param in zip(old_grad, net.parameters()):
        assert torch.allclose(old, new_param.grad)


def test_forward_result_is_stationary_point_of_relax_nobias(net_nb):
    x0 = torch.FloatTensor([0.5, -0.7, 0.2])
    old_z = net_nb.forward(x0)

    ns = net_nb.relax(old_z[0], old_z[-1])

    for old, new in zip(old_z, ns.z):
        assert torch.allclose(old, new)


def test_forward_maps_zero_to_zero_when_nobias(net_nb):
    x0 = torch.FloatTensor([0.0, 0.0, 0.0])
    z = net_nb.forward(x0)

    assert torch.max(torch.abs(z[-1])) < 1e-5


def test_loss_reduction_none(net, data):
    ns = net.relax(data.x, data.y)
    loss = net.loss(ns.z, reduction="none")

    for i, (crt_x, crt_y) in enumerate(zip(data.x, data.y)):
        ns = net.relax(crt_x, crt_y)
        crt_loss = net.loss(ns.z)

        assert loss[i].item() == pytest.approx(crt_loss.item())


def test_loss_reduction_sum(net, data):
    ns = net.relax(data.x, data.y)
    loss = net.loss(ns.z, reduction="sum")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        ns = net.relax(crt_x, crt_y)
        expected += net.loss(ns.z)

    assert loss.item() == pytest.approx(expected.item())


def test_loss_reduction_mean(net, data):
    ns = net.relax(data.x, data.y)
    loss = net.loss(ns.z, reduction="mean")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        ns = net.relax(crt_x, crt_y)
        expected += net.loss(ns.z)

    expected /= len(data.x)
    assert loss.item() == pytest.approx(expected.item())


@pytest.fixture
def net_constraint():
    torch.manual_seed(20392)
    net = PCNetwork([3, 4, 5, 2], constrained=True, rho=[0.2, 1.5])
    return net


def test_q_gradient_with_constraint(net_constraint, data):
    net = net_constraint
    ns = net.relax(data.x, data.y)
    net.calculate_weight_grad(ns)

    outer = lambda a, b: a.unsqueeze(-1) @ b.unsqueeze(-2)
    for i in range(len(net.dims) - 2):
        fct = net.activation[i]
        if fct == "tanh":
            fct = torch.tanh
        elif fct == "none":
            fct = lambda _: _
        fz = fct(ns.z[i + 1])
        n = fz @ net.Q[i].T
        expected = (1 / net.variances[i]) * (net.rho[i] * net.Q[i] - outer(n, fz)).mean(
            dim=0
        )
        assert torch.allclose(net.Q[i].grad, expected, atol=1e-6)


@pytest.mark.parametrize("red", ["sum", "mean"])
def test_calculate_weight_grad_matches_backward_on_loss_constraint(
    net_constraint, red, data
):
    net = net_constraint
    ns = net.relax(data.x, data.y)
    net.calculate_weight_grad(ns, reduction=red)

    old_grad = [_.grad.clone().detach() for _ in net.parameters()]

    for param in net.parameters():
        if param.grad is not None:
            param.grad.zero_()

    loss = net.loss(ns.z, reduction=red)
    loss.backward()
    for Q in net.Q:
        Q.grad = -Q.grad

    for old, new_param in zip(old_grad, net.parameters()):
        assert torch.allclose(old, new_param.grad)


def test_loss_ignore_constraint_works(data):
    seed = 20392
    torch.manual_seed(seed)
    dims = [3, 4, 5, 2]
    rho = [0.2, 1.5]
    net1 = PCNetwork(dims, constrained=True, rho=rho)
    ns1 = net1.relax(data.x, data.y)
    loss1 = net1.loss(ns1.z, ignore_constraint=True)

    torch.manual_seed(seed)
    net2 = PCNetwork(dims, constrained=True, rho=rho)
    ns2 = net2.relax(data.x, data.y)
    net2.constrained = False
    loss2 = net2.loss(ns2.z)

    assert loss1.item() == pytest.approx(loss2.item())


def test_pc_loss_ignores_constraint_by_default(net_constraint, data):
    ns = net_constraint.relax(data.x, data.y)
    loss1 = net_constraint.loss(ns.z)
    loss1_i = net_constraint.loss(ns.z, ignore_constraint=True)

    assert loss1.item() != pytest.approx(loss1_i.item())

    loss2 = net_constraint.pc_loss(ns.z)
    assert loss2.item() == pytest.approx(loss1_i.item())


def get_sample_z_gradient(net):
    # calculate the gradient but don't update z
    net.fast_optimizer = torch.optim.SGD
    net.z_it = 1
    net.z_lr = 0

    x = torch.FloatTensor([-0.1, 0.3, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.relax(x, y)

    # XXX after one step only the last hidden layer should have non-zero gradient when
    #     we start with state from `net.forward()`... that should be enough for this
    #     test, though
    grad_z = [_.grad.detach().clone() for _ in ns.z[1:-1]]

    return ns.z, grad_z


@pytest.fixture
def net_ntv():  # non-trivial variances
    torch.manual_seed(20392)
    net = PCNetwork([3, 4, 5, 2], variances=[0.2, 1.5, 0.9], activation=torch.tanh)
    return net


def test_z_dynamics(net_ntv):
    net = net_ntv
    z, grad_z = get_sample_z_gradient(net)
    activation = torch.tanh
    der_activation = lambda _: 1 / torch.cosh(_) ** 2
    for i in range(1, len(net.dims) - 1):
        mu = activation(z[i - 1]) @ net.W[i - 1].T + net.h[i - 1]
        diff = z[i] - mu
        grad1 = diff / net.variances[i - 1]

        mu = activation(z[i]) @ net.W[i].T + net.h[i]
        diff = z[i + 1] - mu
        der = der_activation(z[i])
        grad2 = der * (diff @ net.W[i]) / net.variances[i]
        expected = grad1 - grad2
        assert torch.allclose(expected, grad_z[i - 1])


def test_z_dynamics_uses_correct_sign_for_constraint(net_constraint):
    net = net_constraint
    activation = torch.tanh
    variances = [1.5, 0.3, 2.5]
    for i in range(len(net.activation)):
        net.activation[i] = activation
        net.variances[i] = variances[i]

    z, grad_z = get_sample_z_gradient(net)
    der_activation = lambda _: 1 / torch.cosh(_) ** 2
    for i in range(1, len(net.dims) - 1):
        mu = activation(z[i - 1]) @ net.W[i - 1].T + net.h[i - 1]
        diff = z[i] - mu
        grad1 = diff / net.variances[i - 1]

        mu = activation(z[i]) @ net.W[i].T + net.h[i]
        diff = z[i + 1] - mu
        der = der_activation(z[i])
        grad2 = der * (diff @ net.W[i]) / net.variances[i]

        fz = activation(z[i])
        grad3 = der * (fz @ net.Q[i - 1].T @ net.Q[i - 1]) / net.variances[i - 1]

        expected = grad1 - grad2 + grad3
        assert torch.allclose(expected, grad_z[i - 1])


def test_parameter_groups_is_iterable_of_dicts_each_with_params_member(net):
    param_groups = net.parameter_groups()
    for d in param_groups:
        assert "params" in d


def test_parameter_groups_returns_dicts_with_name_key(net):
    param_groups = net.parameter_groups()
    for d in param_groups:
        assert "name" in d


def test_parameter_groups_lists_the_same_parameters_as_parameters(net):
    params1 = set(net.parameters())
    params2 = set(sum([_["params"] for _ in net.parameter_groups()], []))

    for x in params1:
        assert x in params2, "element of params missing from param_groups"
    for x in params2:
        assert x in params1, "element of param_groups missing from params"


@pytest.mark.parametrize("var", ["W", "h"])
def test_parameter_groups_contains_expected_names(net, var):
    params = net.parameter_groups()
    names = [_["name"] for _ in params]

    for layer in range(len(net.dims) - 1):
        assert f"{var}:{layer}" in names


def test_parameters_no_bias(net_nb):
    params = net_nb.parameter_groups()
    names = [_["name"] for _ in params]

    assert "b" not in names


def test_parameters_constraint(net_constraint):
    params = net_constraint.parameter_groups()
    names = [_["name"] for _ in params]

    for layer in range(len(net_constraint.dims) - 2):
        assert f"Q:{layer}" in names


def test_parameters_no_constraint(net):
    params = net.parameter_groups()
    names = [_["name"] for _ in params]

    assert not any(_.startswith("Q:") for _ in names)


@pytest.mark.parametrize("var", ["variances", "rho"])
def test_parameters_not_tensors(net_ntv, var):
    for x in getattr(net_ntv, var):
        assert not torch.is_tensor(x)


@pytest.mark.parametrize("var", ["variances", "rho"])
def test_parameters_not_tensors_even_if_fed_tensors(var):
    net = PCNetwork(
        [2, 5, 3, 4],
        variances=torch.FloatTensor([0.5, 1.3, 2.4]),
        rho=torch.FloatTensor([0.3]),
        constrained=True,
    )
    for x in getattr(net, var):
        assert not torch.is_tensor(x)


def test_relax_works_when_called_inside_no_grad(net):
    with torch.no_grad():
        net.relax(torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4]))


def test_choose_activation_function_using_string():
    seed = 43
    dims = [2, 5, 4]

    torch.manual_seed(seed)
    net1 = PCNetwork(dims, activation="relu")

    torch.manual_seed(seed)
    net2 = PCNetwork(dims, activation=torch.relu)

    x = torch.FloatTensor([0.3, -0.5])
    y = torch.FloatTensor([1.3, 0.5, -0.5, 0.8])

    ns1 = net1.relax(x, y)
    ns2 = net2.relax(x, y)

    for z1, z2 in zip(ns1.z, ns2.z):
        assert torch.allclose(z1, z2)


def test_default_activation_function_is_tanh():
    seed = 43
    dims = [2, 5, 4]

    torch.manual_seed(seed)
    net1 = PCNetwork(dims)

    torch.manual_seed(seed)
    net2 = PCNetwork(dims, activation="tanh")

    x = torch.FloatTensor([0.3, -0.5])
    y = torch.FloatTensor([1.3, 0.5, -0.5, 0.8])

    ns1 = net1.relax(x, y)
    ns2 = net2.relax(x, y)

    for z1, z2 in zip(ns1.z, ns2.z):
        assert torch.allclose(z1, z2)


def test_clone(net_ntv):
    net = net_ntv

    torch.manual_seed(0)
    net_clone = net.clone()

    n_steps = 4

    x = torch.Tensor(n_steps, 3).uniform_()
    y = torch.Tensor(n_steps, 2).uniform_()

    for crt_net in [net, net_clone]:
        optimizer = torch.optim.Adam(crt_net.parameters())
        for i in range(4):
            ns = crt_net.relax(x[i], y[i])
            crt_net.calculate_weight_grad(ns)
            optimizer.step()

    for param1, param2 in zip(net.parameters(), net_clone.parameters()):
        assert torch.allclose(param1, param2)


def test_clone_with_constraint(net_constraint):
    net = net_constraint

    torch.manual_seed(0)
    net_clone = net.clone()

    n_steps = 4

    x = torch.Tensor(n_steps, 3).uniform_()
    y = torch.Tensor(n_steps, 2).uniform_()

    for crt_net in [net, net_clone]:
        optimizer = torch.optim.Adam(crt_net.parameters())
        for i in range(4):
            ns = crt_net.relax(x[i], y[i])
            crt_net.calculate_weight_grad(ns)
            optimizer.step()

    for param1, param2 in zip(net.parameters(), net_clone.parameters()):
        assert torch.allclose(param1, param2)


def test_relax_returns_y_pred_equal(net, data):
    ns = net.relax(data.x, data.y)
    assert hasattr(ns, "y_pred")


def test_relax_y_pred_equal_to_last_layer_from_forward(net, data):
    fwd_z = net.forward(data.x)
    ns = net.relax(data.x, data.y)

    assert torch.allclose(ns.y_pred, fwd_z[-1])
