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
    tracker.report.test("foo", 0, torch.tensor(3.0))
    assert hasattr(tracker.history, "test")


def test_tensors_coalesced_after_finalize(tracker):
    a = 3.0
    b = 2.0
    tracker.report.test("foo", 0, torch.tensor(a))
    tracker.report.test("foo", 2, torch.tensor(b))
    tracker.finalize()

    assert "foo" in tracker.history.test
    assert torch.allclose(tracker.history.test["foo"], torch.FloatTensor([a, b]))


def test_idx_field_automatically_generated(tracker):
    idxs = [1, 5, 2]
    for idx in idxs:
        tracker.report.test("bar", idx, torch.tensor(0))
    tracker.finalize()

    assert "idx" in tracker.history.test
    assert torch.allclose(tracker.history.test["idx"], torch.LongTensor(idxs))


def test_report_scalar_nontensors(tracker):
    x = 3.0
    tracker.report.foo("bar", 0, x)
    tracker.finalize()

    assert torch.allclose(tracker.history.foo["bar"], torch.FloatTensor([x]))


def test_report_raises_if_called_after_finalize(tracker):
    tracker.finalize()
    with pytest.raises(ValueError):
        tracker.report.test("foo", 0, 0)


def test_report_ints_leads_to_long_tensor(tracker):
    i = 3
    tracker.report.foo("bar", 0, i)
    tracker.finalize()

    assert torch.allclose(tracker.history.foo["bar"], torch.LongTensor([i]))


def test_report_list_makes_perlayer_entries(tracker):
    x = [torch.FloatTensor([1.0]), torch.FloatTensor([2.0, 3.0])]
    tracker.report.test("x", 0, x)
    tracker.finalize()

    for i in range(len(x)):
        assert f"x:{i}" in tracker.history.test
        assert torch.allclose(tracker.history.test[f"x:{i}"], x[i])


def test_report_adds_row_index_to_tensors(tracker):
    x = torch.FloatTensor([1.0, 3.0])
    tracker.report.test("x", 0, x)
    tracker.finalize()

    assert tracker.history.test["x"].shape == (1, len(x))


def test_report_stacks_tensors_properly(tracker):
    x = torch.FloatTensor([[1.0, 3.0], [4.0, 5.0]])
    for i, row in enumerate(x):
        tracker.report.test("x", i, row)
    tracker.finalize()

    assert torch.allclose(tracker.history.test["x"], x)


def test_set_index_name(tracker):
    tracker.set_index_name("batch")
    tracker.report.test("x", 0, 0)

    assert "idx" not in tracker.history.test
    assert "batch" in tracker.history.test


def test_index_name_in_constructor():
    tracker = Tracker(index_name="batch")
    tracker.report.test("x", 0, 0)

    assert "idx" not in tracker.history.test
    assert "batch" in tracker.history.test


def test_finalize_averages_over_consecutive_entries_with_same_idx(tracker):
    x = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    i = torch.LongTensor([0, 1, 1])
    for crt_i, crt_x in zip(i, x):
        tracker.report.test("x", crt_i, crt_x)
    tracker.finalize()

    assert len(tracker.history.test["x"]) == 2
    assert torch.allclose(tracker.history.test["x"][-1], x[1:].mean(dim=0))


def test_str(tracker):
    tracker.report.test("x", 0, 0)
    s = str(tracker)

    assert s.startswith("Tracker(")
    assert s.endswith(")")


def test_repr(tracker):
    tracker.report.test("x", 0, 0)
    tracker.report.foo("bar", 0, 0)
    s = repr(tracker)

    assert s.startswith("Tracker(")
    assert s.endswith(")")


def test_report_detaches_tensor(tracker):
    x = torch.FloatTensor([1.0, 2.0]).requires_grad_()
    tracker.report.test("x", 0, x)
    tracker.finalize()

    assert tracker.history.test["x"].is_leaf


def test_report_clones_tensor(tracker):
    y = torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tracker.report.foo("y", 0, y)

    # change tensor after reporting, see if change persists
    y_orig = y.detach().clone()
    y[0, 1] = -2.0
    tracker.finalize()

    assert torch.allclose(tracker.history.foo["y"], y_orig)


def test_idx_is_not_duplicated_when_reporting_multiple_vars(tracker):
    tracker.report.test("x", 0, 1.0)
    tracker.report.test("y", 0, 2.0)
    tracker.finalize()

    assert len(tracker.history.test["idx"]) == 1


def test_report_higher_dim_tensor(tracker):
    x = torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.3], [0.5, 0.3, 0.4]])
    tracker.report.test("foo", 0, x)
    tracker.report.test("foo", 1, y)
    tracker.finalize()

    assert len(tracker.history.test["foo"]) == 2
    assert torch.allclose(tracker.history.test["foo"][0], x)
    assert torch.allclose(tracker.history.test["foo"][1], y)


def test_report_meld(tracker):
    tracker.report.test("foo", 0, torch.FloatTensor([1, 2, 3]), meld=True)
    tracker.report.test(
        "bar", 0, torch.FloatTensor([[1, 2], [3, 4], [5, 6]]), meld=True
    )
    tracker.finalize()

    for var in ["idx", "foo", "bar"]:
        assert len(tracker.history.test[var]) == 3


def test_report_raises_if_same_field_with_same_idx_and_meld(tracker):
    tracker.report.test("foo", 0, torch.FloatTensor([1, 2, 3]), meld=True)

    with pytest.raises(ValueError):
        tracker.report.test("foo", 0, torch.FloatTensor([1, 2, 3]), meld=True)


def test_report_meld_repeats_idx_value(tracker):
    idx = 5
    tracker.report.test(
        "bar", idx, torch.FloatTensor([[1, 2], [3, 4], [5, 6]]), meld=True
    )
    tracker.finalize()

    assert torch.allclose(tracker.history.test["idx"], torch.tensor(idx))


def test_report_meld_raises_if_mismatched_sizes(tracker):
    tracker.report.test("foo", 0, torch.FloatTensor([1, 2, 3]), meld=True)
    with pytest.raises(ValueError):
        tracker.report.test("bar", 0, torch.FloatTensor([[1, 2], [3, 4]]), meld=True)


def test_finalize_does_not_average_over_nonconsecutive_entries_with_same_idx(tracker):
    x = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    i = torch.LongTensor([1, 0, 1])
    for crt_i, crt_x in zip(i, x):
        tracker.report.test("x", crt_i, crt_x)
    tracker.finalize()

    assert len(tracker.history.test["x"]) == 3


def test_report_raises_if_mixing_meld_with_non_meld(tracker):
    tracker.report.test("foo", 0, torch.FloatTensor([1, 2]), meld=True)
    with pytest.raises(ValueError):
        tracker.report.test("bar", 0, torch.FloatTensor([[1, 2], [3, 4]]))


def test_report_keeps_last_entry_if_overwrite_true(tracker):
    x0 = torch.FloatTensor([1, 2, 3])
    x1 = torch.FloatTensor([2, 3, 4])
    tracker.report.test("foo", 0, x0)
    tracker.report.test("foo", 0, x1, overwrite=True)
    tracker.finalize()

    assert len(tracker.history.test["foo"]) == 1
    assert torch.allclose(tracker.history.test["foo"][0], x1)


def test_report_keeps_last_entry_if_overwrite_true_and_meld_true(tracker):
    x0 = torch.FloatTensor([1, 2, 3])
    x1 = torch.FloatTensor([2, 3, 4])
    tracker.report.test("foo", 0, x0)
    tracker.report.test("foo", 0, x1, overwrite=True)
    tracker.finalize()

    assert torch.allclose(tracker.history.test["foo"], x1)


def test_report_overwrite_true_does_nothing_if_first_entry(tracker):
    x0 = torch.FloatTensor([1, 2, 3])
    tracker.report.test("foo", 0, x0, overwrite=True)
    tracker.finalize()

    assert len(tracker.history.test["foo"]) == 1
    assert torch.allclose(tracker.history.test["foo"][0], x0)


def test_report_overwrite_true_does_nothing_if_first_entry_and_meld_true(tracker):
    x0 = torch.FloatTensor([1, 2, 3])
    tracker.report.test("foo", 0, x0, overwrite=True, meld=True)
    tracker.finalize()

    assert torch.allclose(tracker.history.test["foo"], x0)


def test_report_multiple_fields_at_once(tracker):
    x = torch.FloatTensor([1, 2, 3])
    y = torch.FloatTensor([[2, 3, 4], [-1, 0.5, 2]])
    tracker.report.test({"x": x, "y": y}, 0)
    tracker.finalize()

    assert "x" in tracker.history.test
    assert "y" in tracker.history.test

    assert len(tracker.history.test["idx"]) == 1
    assert len(tracker.history.test["x"]) == 1
    assert len(tracker.history.test["y"]) == 1

    assert torch.allclose(tracker.history.test["x"][0], x)
    assert torch.allclose(tracker.history.test["y"][0], y)


def test_report_multiple_fields_works_with_meld(tracker):
    x = torch.FloatTensor([0.5, 1, 1.5])
    y = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    tracker.report.test({"x": x, "y": y}, 0, meld=True)
    tracker.finalize()

    assert len(tracker.history.test["idx"]) == 3
    assert torch.allclose(tracker.history.test["x"], x)
    assert torch.allclose(tracker.history.test["y"], y)


def test_report_multiple_fields_works_with_overwrite(tracker):
    tracker.report.test("x", 0, torch.zeros(3))
    tracker.report.test("y", 0, torch.zeros(2, 2))

    x = torch.FloatTensor([0.5, 1, 1.5])
    y = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    tracker.report.test({"x": x, "y": y}, 0, overwrite=True)
    tracker.finalize()

    assert len(tracker.history.test["idx"]) == 1
    assert torch.allclose(tracker.history.test["x"][-1], x)
    assert torch.allclose(tracker.history.test["y"][-1], y)


def test_report_multilayer_works_with_meld(tracker):
    w0 = torch.FloatTensor([[1, 2.5], [3, 2.2]])
    w1 = torch.FloatTensor([[-0.1, 1.5, 0.5], [0.2, 2.3, -1.2]])
    tracker.report.test("w", 0, [w0, w1], meld=True)
    tracker.finalize()

    assert "w:0" in tracker.history.test
    assert "w:1" in tracker.history.test

    assert len(tracker.history.test["idx"]) == 2
    assert torch.allclose(tracker.history.test["w:0"], w0)
    assert torch.allclose(tracker.history.test["w:1"], w1)


def test_report_multilayer_works_with_overwrite(tracker):
    tracker.report.test("w", 0, [torch.zeros(2), torch.zeros(3)])

    w0 = torch.FloatTensor([1, 2.5])
    w1 = torch.FloatTensor([-0.1, 1.5, 0.5])
    tracker.report.test("w", 0, [w0, w1], overwrite=True)
    tracker.finalize()

    assert len(tracker.history.test["idx"]) == 1
    assert len(tracker.history.test["w:0"]) == 1
    assert len(tracker.history.test["w:1"]) == 1

    assert torch.allclose(tracker.history.test["w:0"][-1], w0)
    assert torch.allclose(tracker.history.test["w:1"][-1], w1)
