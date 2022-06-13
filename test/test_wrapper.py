import pytest

import torch

from unittest.mock import Mock
from types import SimpleNamespace

from cpcn.wrapper import PCWrapper


@pytest.fixture
def data() -> tuple:
    x = torch.FloatTensor([[-0.1, 0.2, 0.4], [0.5, 0.3, 0.2], [-1.0, 2.3, 0.1]])
    y = torch.FloatTensor([[0.3, -0.4], [0.1, 0.2], [-0.5, 1.2]])
    return SimpleNamespace(x=x, y=y)


def get_mock_net():
    net = Mock()
    net.relax.side_effect = lambda x, y, **kwargs: SimpleNamespace(
        z=[torch.ones(len(x), 1), x, torch.linalg.norm(y)]
    )
    net.pc_loss.return_value = torch.tensor(0.0)
    net.parameters.return_value = [torch.FloatTensor([2.3])]
    net.parameter_groups.return_value = [
        {"name": "a", "params": torch.FloatTensor([2.3])}
    ]
    net.dims = [3, 3, 2]

    return net


@pytest.fixture
def mock_net():
    return get_mock_net()


def test_wrapper_relax_calls_net_relax(mock_net, data):
    wrapper = PCWrapper(mock_net, lambda _: _)

    wrapper.relax(data.x, data.y)

    mock_net.relax.assert_called()


def test_wrapper_relax_calls_predictor(mock_net, data):
    predictor = Mock(return_value=torch.tensor(0.0))
    wrapper = PCWrapper(mock_net, predictor)
    wrapper.relax(data.x, data.y)

    predictor.assert_called()


def test_wrapper_relax_yields_y_pred_from_predictor_call(mock_net, data):
    ret_val = 1.3
    predictor = Mock(return_value=torch.tensor(ret_val))
    wrapper = PCWrapper(mock_net, predictor)
    ns = wrapper.relax(data.x, data.y)

    assert pytest.approx(ns.y_pred.item()) == ret_val


def test_wrapper_parameters_includes_predictor_parameters(mock_net):
    predictor = Mock(return_value=torch.tensor(0.0))
    w = torch.FloatTensor([-0.5, 0.3])
    predictor.parameters.return_value = [w]

    wrapper = PCWrapper(mock_net, predictor)
    params = wrapper.parameters()

    assert any(_ is w for _ in params)


def test_wrapper_parameter_groups_includes_predictor_parameters(mock_net):
    predictor = Mock(return_value=torch.tensor(0.0))
    w = torch.FloatTensor([-0.5, 0.3])
    v = torch.FloatTensor([0.5, -0.3])
    predictor.parameters.return_value = [w, v]

    wrapper = PCWrapper(mock_net, predictor)
    params = wrapper.parameter_groups()

    names = [_["name"] for _ in params]

    assert "predictor" in names
    idx = names.index("predictor")

    assert len(params[idx]["params"]) == 2
    assert any(_ is v for _ in params[idx]["params"])
    assert any(_ is w for _ in params[idx]["params"])


def get_wrapped(loss=None):
    net = get_mock_net()
    predictor = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())

    kwargs = {}
    if loss is not None:
        kwargs["loss"] = loss
    wrapper = PCWrapper(net, predictor, **kwargs)

    return wrapper


def test_wrapper_default_loss_is_mse(data):
    seed = 1
    lr = 1e-3

    torch.manual_seed(seed)
    wrapper1 = get_wrapped()
    optim1 = torch.optim.SGD(wrapper1.parameters(), lr=lr)
    ns1 = wrapper1.relax(data.x, data.y)
    wrapper1.calculate_weight_grad(ns1)
    optim1.step()

    torch.manual_seed(seed)
    wrapper2 = get_wrapped(loss=torch.nn.MSELoss())
    optim2 = torch.optim.SGD(wrapper2.parameters(), lr=lr)
    ns2 = wrapper2.relax(data.x, data.y)
    wrapper2.calculate_weight_grad(ns2)
    optim2.step()

    for param1, param2 in zip(wrapper1.parameters(), wrapper2.parameters()):
        assert torch.allclose(param1, param2)


def test_wrapper_calculate_weight_grad_does_backward_on_loss(data):
    loss_val = Mock()
    loss = Mock(return_value=loss_val)
    wrapper = get_wrapped(loss=loss)

    ns = wrapper.relax(data.x, data.y)
    wrapper.calculate_weight_grad(ns)

    loss_val.backward.assert_called()


def test_wrapper_calculate_weight_grad_calls_pc_loss_weight_grad(mock_net, data):
    wrapper = PCWrapper(mock_net, "linear")
    ns = wrapper.relax(data.x, data.y)
    wrapper.calculate_weight_grad(ns)

    mock_net.calculate_weight_grad.assert_called()


def test_wrapper_different_dim(mock_net, data):
    predictor = Mock(side_effect=lambda x: x)

    wrapper = PCWrapper(mock_net, predictor, dim=0)
    ns = wrapper.relax(data.x, data.y)

    assert ns.y_pred.shape == (len(data.x), 1)


def test_wrapper_default_dim_is_next_to_last(mock_net, data):
    predictor = Mock(side_effect=lambda x: x)

    wrapper = PCWrapper(mock_net, predictor)
    ns = wrapper.relax(data.x, data.y)

    assert ns.y_pred.shape == data.x.shape


@pytest.mark.parametrize("kind", ["linear", "linear-relu", "linear-softmax"])
def test_wrapper_str_predictor(kind, data):
    seed = 1
    torch.manual_seed(seed)

    net1 = get_mock_net()
    wrapper1 = PCWrapper(net1, kind)
    ns1 = wrapper1.relax(data.x, data.y)

    torch.manual_seed(seed)

    net2 = get_mock_net()
    predictor = torch.nn.Linear(data.x.shape[1], data.y.shape[1])
    if kind == "linear-relu":
        predictor = torch.nn.Sequential(predictor, torch.nn.ReLU())
    elif kind == "linear-softmax":
        predictor = torch.nn.Sequential(predictor, torch.nn.Softmax())
    else:
        assert kind == "linear"
    wrapper2 = PCWrapper(net2, predictor)
    ns2 = wrapper2.relax(data.x, data.y)

    assert torch.allclose(ns1.y_pred, ns2.y_pred)


def test_wrapper_passes_along_pc_loss_value(mock_net, data):
    ret_val = torch.tensor(1.2)
    mock_net.pc_loss.return_value = ret_val
    wrapper = PCWrapper(mock_net, "linear")

    ns = wrapper.relax(data.x, data.y)
    loss = wrapper.pc_loss(ns.z)

    assert torch.allclose(loss, ret_val)


def test_relax_passes_additional_kwargs_to_pc_net(mock_net, data):
    wrapper = PCWrapper(mock_net, "linear")
    wrapper.relax(data.x, data.y, foo="bar")

    assert "foo" in mock_net.relax.call_args[1]
    assert mock_net.relax.call_args[1]["foo"] == "bar"


def test_calculate_weight_grad_passes_additional_kwargs_to_pc_net(mock_net, data):
    wrapper = PCWrapper(mock_net, "linear")
    ns = wrapper.relax(data.x, data.y)
    wrapper.calculate_weight_grad(ns, foo="bar")

    assert "foo" in mock_net.calculate_weight_grad.call_args[1]
    assert mock_net.calculate_weight_grad.call_args[1]["foo"] == "bar"


def test_to_calls_pc_net_to(mock_net):
    wrapper = PCWrapper(mock_net, "linear")
    wrapper.to("cpu")

    mock_net.to.assert_called_with("cpu")


def test_to_calls_predictor_to(mock_net):
    predictor = Mock()
    wrapper = PCWrapper(mock_net, predictor)
    wrapper.to("cuda")

    predictor.to.assert_called_with("cuda")


def test_to_returns_self():
    wrapper = get_wrapped()
    assert wrapper.to("cpu") is wrapper


def test_train_calls_pc_net_train(mock_net):
    wrapper = PCWrapper(mock_net, "linear")
    wrapper.train()

    mock_net.train.assert_called()


def test_eval_calls_pc_net_eval(mock_net):
    wrapper = PCWrapper(mock_net, "linear")
    wrapper.eval()

    mock_net.eval.assert_called()


def test_train_calls_predictor_train(mock_net):
    predictor = Mock()
    wrapper = PCWrapper(mock_net, predictor)
    wrapper.train()

    predictor.train.assert_called()


def test_eval_calls_predictor_train(mock_net):
    predictor = Mock()
    wrapper = PCWrapper(mock_net, predictor)
    wrapper.eval()

    predictor.eval.assert_called()
