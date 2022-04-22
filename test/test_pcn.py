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
    assert len(net.b) == 2
    assert len(net.z) == 3


def test_weight_sizes(net):
    assert net.W[0].shape == (4, 3)
    assert net.W[1].shape == (2, 4)


def test_z_sizes(net):
    net.forward(torch.FloatTensor([0.3, -0.2, 0.5]))
    assert [len(_) for _ in net.z] == [3, 4, 2]


def test_all_zs_nonzero_after_forward_constrained(net):
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    for z in net.z:
        assert torch.max(torch.abs(z)) > 1e-4


def test_all_zs_change_during_forward_constrained(net):
    # set some starting values for z
    net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))

    old_z = [_.clone() for _ in net.z]
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    for old, new in zip(old_z, net.z):
        assert not torch.all(torch.isclose(old, new))


def test_all_zs_not_nonzero_after_forward(net):
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for z in net.z:
        assert torch.max(torch.abs(z)) > 1e-4


def test_all_zs_change_during_forward(net):
    # set some starting values for z
    net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))

    old_z = [_.clone() for _ in net.z]
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for old, new in zip(old_z, net.z):
        assert not torch.any(torch.isclose(old, new))


def test_forward_result_is_stationary_point_of_forward_constrained(net):
    x0 = torch.FloatTensor([0.5, -0.7, 0.2])
    net.forward(x0)

    old_z = [_.clone().detach() for _ in net.z]
    net.forward_constrained(old_z[0], old_z[-1])

    for old, new in zip(old_z, net.z):
        assert torch.allclose(old, new)


def test_weights_and_biases_change_when_optimizing_slow_parameters(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    old_Ws = [_.clone().detach() for _ in net.W]
    old_bs = [_.clone().detach() for _ in net.b]

    optimizer = torch.optim.Adam(net.slow_parameters(), lr=1.0)
    net.forward_constrained(x0, y0)

    optimizer.zero_grad()
    loss = net.loss()
    loss.backward()
    optimizer.step()

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_bs, net.W, net.b):
        assert not torch.any(torch.isclose(old_W, new_W))
        assert not torch.any(torch.isclose(old_b, new_b))


def test_loss_is_nonzero_after_forward_constrained(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    net.forward_constrained(x0, y0)

    assert net.loss().abs().item() > 1e-6


def test_forward_does_not_change_weights_and_biases(net):
    old_Ws = [_.clone().detach() for _ in net.W]
    old_bs = [_.clone().detach() for _ in net.b]
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_bs, net.W, net.b):
        assert torch.allclose(old_W, new_W)
        assert torch.allclose(old_b, new_b)


def test_forward_constrained_does_not_change_weights_and_biases(net):
    old_Ws = [_.clone().detach() for _ in net.W]
    old_bs = [_.clone().detach() for _ in net.b]
    net.forward_constrained(
        torch.FloatTensor([0.3, -0.4, 0.2]), torch.FloatTensor([-0.2, 0.2])
    )

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_bs, net.W, net.b):
        assert torch.allclose(old_W, new_W)
        assert torch.allclose(old_b, new_b)


def test_loss_does_not_change_weights_and_biases(net):
    # ensure the z variables have valid values assigned to them
    net.forward(torch.FloatTensor([0.1, 0.2, 0.3]))

    old_Ws = [_.clone().detach() for _ in net.W]
    old_bs = [_.clone().detach() for _ in net.b]
    net.loss()

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_bs, net.W, net.b):
        assert torch.allclose(old_W, new_W)
        assert torch.allclose(old_b, new_b)


def test_no_nan_or_inf_after_a_few_learning_steps(net):
    torch.manual_seed(0)

    optimizer = torch.optim.Adam(net.slow_parameters())
    for i in range(4):
        x = torch.Tensor(3).uniform_()
        y = torch.Tensor(2).uniform_()
        net.forward_constrained(x, y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    for W, b in zip(net.W, net.b):
        assert torch.all(torch.isfinite(W))
        assert torch.all(torch.isfinite(b))

    for z in net.z:
        assert torch.all(torch.isfinite(z))


def test_forward_output_depends_on_input(net):
    y1 = [_.detach().clone() for _ in net.forward(torch.FloatTensor([0.1, 0.3, -0.2]))]
    y2 = [_.detach().clone() for _ in net.forward(torch.FloatTensor([-0.5, 0.1, 0.2]))]

    for a, b in zip(y1, y2):
        assert not torch.allclose(a, b)


def test_forward_sets_first_element_of_z_to_input_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    net.forward(x0)
    assert torch.allclose(net.z[0], x0)


def test_forward_constrained_sets_first_element_of_z_to_input_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    y0 = torch.FloatTensor([0.5, -0.2])
    net.forward_constrained(x0, y0)
    assert torch.allclose(net.z[0], x0)


def test_forward_constrained_sets_last_element_of_z_to_output_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    y0 = torch.FloatTensor([0.5, -0.2])
    net.forward_constrained(x0, y0)
    assert torch.allclose(net.z[-1], y0)


def test_initialize_values_same_when_torch_seed_is_same():
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    old_Ws = [_.clone().detach() for _ in net.W]
    old_bs = [_.clone().detach() for _ in net.b]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    new_Ws = [_.clone().detach() for _ in net.W]
    new_bs = [_.clone().detach() for _ in net.b]

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
    optimizer = torch.optim.Adam(net.slow_parameters())
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    var1 = [_.clone().detach() for _ in net.W]

    # reset and do the learning again
    torch.manual_seed(seed)
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.slow_parameters())
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
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
    optimizer = torch.optim.Adam(net.slow_parameters())
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    var1 = [_.clone().detach() for _ in net.W]

    # reset and do the learning again -- without resetting random seed this time!
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.slow_parameters())
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
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

    optimizer = torch.optim.SGD(net.slow_parameters(), lr=lr)
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    test_x = torch.FloatTensor([0.5, 0.2])
    out = net.forward(test_x)

    # do the same learning with batches of size 1
    torch.manual_seed(seed)
    net = PCNetwork(dims, variances=variances)

    optimizer = torch.optim.SGD(net.slow_parameters(), lr=lr)
    for crt_x, crt_y in zip(x, y):
        crt_x_batch = crt_x[None, :]
        crt_y_batch = crt_y[None, :]
        net.forward_constrained(crt_x_batch, crt_y_batch)

        optimizer.zero_grad()
        net.loss().backward()
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
    it_inference = 10

    n_samples = 50

    torch.manual_seed(seed)
    x = torch.normal(0, 1, size=(n_samples, dims[0]))
    y = torch.normal(0, 1, size=(n_samples, dims[-1]))

    # do some learning
    torch.manual_seed(seed)
    kwargs = {"variances": variances, "it_inference": it_inference}
    net = PCNetwork(dims, **kwargs)

    # gradients are averaged over batch samples by default, equivalent to lower lr when
    # training sample by sample
    optimizer = torch.optim.SGD(net.slow_parameters(), lr=lr / n_samples)
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    test_x = torch.FloatTensor([0.5, -0.2])
    out = net.forward(test_x)

    # do the same learning with batches of size 1
    torch.manual_seed(seed)
    net = PCNetwork(dims, **kwargs)

    optimizer = torch.optim.SGD(net.slow_parameters(), lr=lr)
    for crt_x1, crt_x2, crt_y1, crt_y2 in zip(x[::2], x[1::2], y[::2], y[1::2]):
        crt_x_batch = torch.vstack((crt_x1, crt_x2))
        crt_y_batch = torch.vstack((crt_y1, crt_y2))
        net.forward_constrained(crt_x_batch, crt_y_batch)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    test_x = torch.FloatTensor([0.5, -0.2])
    out_batch = net.forward(test_x)

    # we don't expect these to be super close -- when we make the steps one by one, the
    # gradients at subsequent batches change due to earlier steps
    for crt_out, crt_out_batch in zip(out, out_batch):
        assert torch.allclose(crt_out, crt_out_batch, rtol=0.01, atol=1e-5)


def test_forward_constrained_returns_empty_namespace_by_default(net):
    ns = net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )
    assert len(ns.__dict__) == 0


def test_forward_constrained_loss_profile_is_sequence_of_correct_length(net):
    ns = net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        pc_loss_profile=True,
    )

    assert len(ns.pc_loss) == net.it_inference


def test_forward_constrained_loss_profile_is_sequence_of_positive_numbers(net):
    ns = net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        pc_loss_profile=True,
    )

    assert min(ns.pc_loss) > 0


def test_forward_constrained_loss_profile_has_last_elem_smaller_than_first(net):
    ns = net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        pc_loss_profile=True,
    )

    assert ns.pc_loss[-1] < ns.pc_loss[0]


def test_forward_constrained_latent_profile_batch_index_added(net):
    ns = net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        latent_profile=True,
    )
    for z in ns.latent.z:
        assert z.ndim == 3
        assert z.shape[1] == 1


def test_forward_constrained_latent_profile_has_correct_length(net):
    ns = net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]),
        torch.FloatTensor([0.3, -0.4]),
        latent_profile=True,
    )
    for z in ns.latent.z:
        assert len(z) == net.it_inference


def test_forward_constrained_latent_profile_first_layer_is_input(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.forward_constrained(x, y, latent_profile=True)
    assert torch.max(torch.abs(ns.latent.z[0] - x)) < 1e-5


def test_forward_constrained_latent_profile_last_layer_is_output(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.forward_constrained(x, y, latent_profile=True)
    assert torch.max(torch.abs(ns.latent.z[-1] - y)) < 1e-5


def test_forward_constrained_latent_profile_row_matches_shorter_run(net):
    x = torch.FloatTensor([-0.1, 0.2, 0.4])
    y = torch.FloatTensor([0.3, -0.4])
    ns = net.forward_constrained(x, y, latent_profile=True)

    new_it = 2
    net.it_inference = new_it
    net.forward_constrained(x, y)

    for i, z in enumerate(ns.latent.z):
        assert torch.allclose(net.z[i], z[new_it - 1, 0])


def test_forward_constrained_latent_profile_with_batch(net):
    x = torch.FloatTensor([[-0.1, 0.2, 0.4], [0.5, 0.3, 0.2]])
    y = torch.FloatTensor([[0.3, -0.4], [0.1, 0.2]])
    ns = net.forward_constrained(x, y, latent_profile=True)

    for k in range(len(x)):
        crt_x = x[k]
        crt_y = y[k]
        crt_ns = net.forward_constrained(crt_x, crt_y, latent_profile=True)

        for z1, z2 in zip(ns.latent.z, crt_ns.latent.z):
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
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    assert net.loss().item() == pytest.approx(net.pc_loss().item())


@pytest.fixture
def data() -> tuple:
    x = torch.FloatTensor([[-0.1, 0.2, 0.4], [0.5, 0.3, 0.2], [-1.0, 2.3, 0.1]])
    y = torch.FloatTensor([[0.3, -0.4], [0.1, 0.2], [-0.5, 1.2]])
    return SimpleNamespace(x=x, y=y)


@pytest.mark.parametrize("red", ["sum", "mean"])
def test_calculate_weight_grad_matches_backward_on_loss(net, red, data):
    net.forward_constrained(data.x, data.y)
    net.calculate_weight_grad(reduction=red)

    old_grad = [_.grad.clone().detach() for _ in net.slow_parameters()]

    for param in net.slow_parameters():
        if param.grad is not None:
            param.grad.zero_()

    loss = net.loss(reduction=red)
    loss.backward()

    for old, new_param in zip(old_grad, net.slow_parameters()):
        assert torch.allclose(old, new_param.grad)


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


def test_forward_result_is_stationary_point_of_forward_constrained_nobias(net_nb):
    x0 = torch.FloatTensor([0.5, -0.7, 0.2])
    net_nb.forward(x0)

    old_z = [_.clone().detach() for _ in net_nb.z]
    net_nb.forward_constrained(old_z[0], old_z[-1])

    for old, new in zip(old_z, net_nb.z):
        assert torch.allclose(old, new)


def test_forward_maps_zero_to_zero_when_nobias(net_nb):
    x0 = torch.FloatTensor([0.0, 0.0, 0.0])
    net_nb.forward(x0)

    assert torch.max(torch.abs(net_nb.z[-1])) < 1e-5


def test_loss_reduction_none(net, data):
    net.forward_constrained(data.x, data.y)
    loss = net.loss(reduction="none")

    for i, (crt_x, crt_y) in enumerate(zip(data.x, data.y)):
        net.forward_constrained(crt_x, crt_y)
        crt_loss = net.loss()

        assert loss[i].item() == pytest.approx(crt_loss.item())


def test_loss_reduction_sum(net, data):
    net.forward_constrained(data.x, data.y)
    loss = net.loss(reduction="sum")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        net.forward_constrained(crt_x, crt_y)
        expected += net.loss()

    assert loss.item() == pytest.approx(expected.item())


def test_loss_reduction_mean(net, data):
    net.forward_constrained(data.x, data.y)
    loss = net.loss(reduction="mean")

    expected = 0
    for crt_x, crt_y in zip(data.x, data.y):
        net.forward_constrained(crt_x, crt_y)
        expected += net.loss()

    expected /= len(data.x)
    assert loss.item() == pytest.approx(expected.item())
