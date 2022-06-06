import pytest

from cpcn.track import Tracker

import torch


@pytest.fixture
def tracker() -> Tracker:
    return Tracker()


def test_history_is_empty_if_nothing_is_added(tracker):
    tracker.finalize()
    assert len(tracker.history.__dict__) == 0


def test_report_adds_history_entry(tracker):
    tracker.test.report(0, "foo", torch.tensor(3.0))
    assert hasattr(tracker.history, "test")


def test_tensors_coalesced_after_finalize(tracker):
    a = 3.0
    b = 2.0
    tracker.test.report(0, "foo", torch.tensor(a))
    tracker.test.report(2, "foo", torch.tensor(b))
    tracker.finalize()

    assert "foo" in tracker.history.test
    assert torch.allclose(tracker.history.test["foo"], torch.FloatTensor([a, b]))


def test_reports_can_be_accessed_without_history(tracker):
    a = 3.0
    b = 2.0
    tracker.test.report(0, "foo", torch.tensor(a))
    tracker.test.report(2, "foo", torch.tensor(b))
    tracker.finalize()

    assert "foo" in tracker.test
    assert torch.allclose(tracker.test["foo"], torch.FloatTensor([a, b]))


def test_idx_field_automatically_generated(tracker):
    idxs = [1, 5, 2]
    for idx in idxs:
        tracker.test.report(idx, "bar", torch.tensor(0))
    tracker.finalize()

    assert "idx" in tracker.test
    assert torch.allclose(tracker.test["idx"], torch.LongTensor(idxs))


def test_report_scalar_nontensors(tracker):
    x = 3.0
    tracker.foo.report(0, "bar", x)
    tracker.finalize()

    assert torch.allclose(tracker.foo["bar"], torch.FloatTensor([x]))


def test_access_raises_if_called_for_inexistent_field(tracker):
    tracker.finalize()
    with pytest.raises(AttributeError):
        tracker.test["foo"]


def test_report_ints_leads_to_long_tensor(tracker):
    i = 3
    tracker.foo.report(0, "bar", i)
    tracker.finalize()

    assert torch.allclose(tracker.foo["bar"], torch.LongTensor([i]))


def test_report_list_makes_perlayer_entries(tracker):
    x = [torch.FloatTensor([1.0]), torch.FloatTensor([2.0, 3.0])]
    tracker.test.report(0, "x", x)
    tracker.finalize()

    for i in range(len(x)):
        assert f"x:{i}" in tracker.test
        assert torch.allclose(tracker.test[f"x:{i}"], x[i])


def test_report_adds_row_index_to_tensors(tracker):
    x = torch.FloatTensor([1.0, 3.0])
    tracker.test.report(0, "x", x)
    tracker.finalize()

    assert tracker.test["x"].shape == (1, len(x))


def test_report_stacks_tensors_properly(tracker):
    x = torch.FloatTensor([[1.0, 3.0], [4.0, 5.0]])
    for i, row in enumerate(x):
        tracker.test.report(i, "x", row)
    tracker.finalize()

    assert torch.allclose(tracker.test["x"], x)


def test_set_index_name(tracker):
    tracker.set_index_name("batch")
    tracker.test.report(0, "x", 0)
    tracker.finalize()

    assert "idx" not in tracker.test
    assert "batch" in tracker.test


def test_index_name_in_constructor():
    tracker = Tracker(index_name="batch")
    tracker.test.report(0, "x", 0)
    tracker.finalize()

    assert "idx" not in tracker.test
    assert "batch" in tracker.test


def test_str(tracker):
    tracker.test.report(0, "x", 0)
    s = str(tracker)

    assert s.startswith("Tracker(")
    assert s.endswith(")")


def test_repr(tracker):
    tracker.test.report(0, "x", 0)
    tracker.foo.report(0, "bar", 0)
    s = repr(tracker)

    assert s.startswith("Tracker(")
    assert s.endswith(")")


def test_report_detaches_tensor(tracker):
    x = torch.FloatTensor([1.0, 2.0]).requires_grad_()
    tracker.test.report(0, "x", x)
    tracker.finalize()

    assert tracker.test["x"].is_leaf


def test_report_clones_tensor(tracker):
    y = torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tracker.foo.report(0, "y", y)

    # change tensor after reporting, see if change persists
    y_orig = y.detach().clone()
    y[0, 1] = -2.0
    tracker.finalize()

    assert torch.allclose(tracker.foo["y"], y_orig)


def test_idx_is_not_duplicated_when_reporting_multiple_vars(tracker):
    tracker.test.report(0, "x", 1.0)
    tracker.test.report(0, "y", 2.0)
    tracker.finalize()

    assert len(tracker.test["idx"]) == 1


def test_report_higher_dim_tensor(tracker):
    x = torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.3], [0.5, 0.3, 0.4]])
    tracker.test.report(0, "foo", x)
    tracker.test.report(1, "foo", y)
    tracker.finalize()

    assert len(tracker.test["foo"]) == 2
    assert torch.allclose(tracker.test["foo"][0], x)
    assert torch.allclose(tracker.test["foo"][1], y)


def test_report_meld(tracker):
    tracker.test.report(0, "foo", torch.FloatTensor([1, 2, 3]), meld=True)
    tracker.test.report(
        0, "bar", torch.FloatTensor([[1, 2], [3, 4], [5, 6]]), meld=True
    )
    tracker.finalize()

    for var in ["idx", "foo", "bar"]:
        assert len(tracker.test[var]) == 3


def test_report_meld_repeats_idx_value(tracker):
    idx = 5
    tracker.test.report(
        idx, "bar", torch.FloatTensor([[1, 2], [3, 4], [5, 6]]), meld=True
    )
    tracker.finalize()

    assert torch.allclose(tracker.test["idx"], torch.tensor(idx))


def test_report_meld_raises_if_mismatched_sizes(tracker):
    tracker.test.report(0, "foo", torch.FloatTensor([1, 2, 3]), meld=True)
    with pytest.raises(IndexError):
        tracker.test.report(0, "bar", torch.FloatTensor([[1, 2], [3, 4]]), meld=True)


def test_report_raises_if_mixing_meld_with_non_meld(tracker):
    tracker.test.report(0, "foo", torch.FloatTensor([1, 2]), meld=True)
    with pytest.raises(IndexError):
        tracker.test.report(0, "bar", torch.FloatTensor([[1, 2], [3, 4]]))


def test_report_multiple_fields_at_once(tracker):
    x = torch.FloatTensor([1, 2, 3])
    y = torch.FloatTensor([[2, 3, 4], [-1, 0.5, 2]])
    tracker.test.report(0, {"x": x, "y": y})
    tracker.finalize()

    assert "x" in tracker.test
    assert "y" in tracker.test

    assert len(tracker.test["idx"]) == 1
    assert len(tracker.test["x"]) == 1
    assert len(tracker.test["y"]) == 1

    assert torch.allclose(tracker.test["x"][0], x)
    assert torch.allclose(tracker.test["y"][0], y)


def test_report_multiple_fields_works_with_meld(tracker):
    x = torch.FloatTensor([0.5, 1, 1.5])
    y = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    tracker.test.report(0, {"x": x, "y": y}, meld=True)
    tracker.finalize()

    assert len(tracker.test["idx"]) == 3
    assert torch.allclose(tracker.test["x"], x)
    assert torch.allclose(tracker.test["y"], y)


def test_report_multilayer_works_with_meld(tracker):
    w0 = torch.FloatTensor([[1, 2.5], [3, 2.2]])
    w1 = torch.FloatTensor([[-0.1, 1.5, 0.5], [0.2, 2.3, -1.2]])
    tracker.test.report(0, "w", [w0, w1], meld=True)
    tracker.finalize()

    assert "w:0" in tracker.test
    assert "w:1" in tracker.test

    assert len(tracker.test["idx"]) == 2
    assert torch.allclose(tracker.test["w:0"], w0)
    assert torch.allclose(tracker.test["w:1"], w1)


def test_finalize_does_not_average_over_entries_with_same_idx(tracker):
    x = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    i = torch.LongTensor([1, 1, 1])
    for crt_i, crt_x in zip(i, x):
        tracker.test.report(crt_i, "x", crt_x)
    tracker.finalize()

    assert len(tracker.test["x"]) == 3


def test_report_raises_if_mismatched_report_lengths(tracker):
    tracker.test.report(0, "x", 0.0)
    tracker.test.report(1, "x", -0.5)

    with pytest.raises(IndexError):
        tracker.test.report(1, "y", 1.0)


def test_report_raises_if_mismatched_report_indices(tracker):
    tracker.test.report(0, "x", 0.0)
    tracker.test.report(0, "y", 0.5)
    tracker.test.report(1, "x", -0.5)

    with pytest.raises(IndexError):
        tracker.test.report(0, "y", 1.0)


def test_accessing_dict_does_not_create_it(tracker):
    tracker.test

    assert not hasattr(tracker.history, "test")


def test_accumulate_followed_by_report_accumulated_averages(tracker):
    x = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for crt_x in x:
        tracker.test.accumulate("x", crt_x)
    tracker.test.report_accumulated(0)
    tracker.finalize()

    assert len(tracker.test["x"]) == 1
    assert torch.allclose(tracker.test["x"][-1], x.mean(dim=0))


def test_accumulate_followed_by_report_accumulated_uses_correct_idx(tracker):
    x = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for crt_x in x:
        tracker.test.accumulate("x", crt_x)

    idx = 15
    tracker.test.report_accumulated(idx)
    tracker.finalize()

    assert len(tracker.test["idx"]) == 1
    assert tracker.test["idx"][-1] == idx


def test_report_accumulated_resets_accumulator(tracker):
    xs = [1.5, 2.3]
    tracker.test.accumulate("x", xs[0])
    tracker.test.report_accumulated(0)

    tracker.test.accumulate("x", xs[1])
    tracker.test.report_accumulated(1)

    tracker.finalize()

    assert pytest.approx(tracker.test["x"][-1].item()) == xs[1]


def test_report_iterable_of_non_tensors(tracker):
    tracker.test.report(0, "x", [0, 0])
    tracker.finalize()

    assert len(tracker.test["x:0"]) == 1
    assert len(tracker.test["x:1"]) == 1


def test_report_dict_with_non_tensors(tracker):
    tracker.test.report(0, {"x": torch.tensor(1.0), "y": 2.0})
    tracker.finalize()

    assert len(tracker.test["x"]) == 1
    assert len(tracker.test["y"]) == 1


def test_calculate_accumulated(tracker):
    values = [5.0, 2.5, -0.3]
    for value in values:
        tracker.test.accumulate("x", value)
    mean = tracker.test.calculate_accumulated("x")

    assert pytest.approx(mean.item()) == sum(values) / len(values)


def test_calculate_accumulated_does_not_clear_accumulator(tracker):
    values = [0.5, -0.3]
    tracker.test.accumulate("x", values[0])
    assert pytest.approx(tracker.test.calculate_accumulated("x").item()) == values[0]

    tracker.test.accumulate("x", values[1])
    accum_val = tracker.test.calculate_accumulated("x").item()
    assert pytest.approx(accum_val) != values[1]
    assert pytest.approx(accum_val) == sum(values) / len(values)
