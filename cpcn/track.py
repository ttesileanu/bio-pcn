"""Define a class to track model (torch) tensors and/or (numpy) arrays."""


import torch
import numpy as np

from types import SimpleNamespace
from typing import Union, Iterable, Callable, Tuple


def _dispatch_values(
    reporter: Callable,
    field: Union[str, dict],
    value: Union[None, int, float, Iterable, torch.Tensor, np.ndarray] = None,
    **kwargs,
):
    """Pass values to a reporter, handling lists or tensors, as well as dicts for
    multi-parameter reports.
    """
    if not isinstance(field, str):
        if value is not None:
            raise ValueError("Tracker: value used with multi-parameter report")

        for crt_key, crt_value in field.items():
            _dispatch_values(reporter, crt_key, crt_value, **kwargs)
        return

    if not torch.is_tensor(value):
        if isinstance(value, np.ndarray):
            value = np.copy(value)
        elif hasattr(value, "__iter__"):
            for i, sub_value in enumerate(value):
                _dispatch_values(reporter, f"{field}:{i}", sub_value, **kwargs)
            return
        else:
            value = np.array(value)
    else:
        value = value.detach().cpu().clone().numpy()

    reporter(field, value, **kwargs)


class _Reporter:
    """Helper for Tracker, used to report new values."""

    def __init__(self, tracker: "Tracker", name: str):
        """Construct the reporter.
        
        :param tracker: associated tracker object
        :param name: dictionary that this refers to
        """
        self.tracker = tracker
        self.name = name

    def report(self, idx: Union[int, Tuple[int]], *args, **kwargs):
        """Report one or multiple values, layered or not, with melding or not."""
        # make a history field, if it does not exist
        if not hasattr(self.tracker.history, self.name):
            setattr(self.tracker.history, self.name, {})
        _dispatch_values(self._report, *args, idx=idx, **kwargs)

    def accumulate(self, *args, **kwargs):
        """Accumulate one or multiple values, layered or not."""
        # make an accumulator field, if it does not exist
        if not hasattr(self.tracker._accumulator, self.name):
            setattr(self.tracker._accumulator, self.name, {})
        _dispatch_values(self._accumulate, *args, **kwargs)

    def calculate_accumulated(self, field: str) -> np.ndarray:
        """Average all accumulated values for a given field."""
        # make an accumulator field, if it does not exist
        if not hasattr(self.tracker._accumulator, self.name):
            setattr(self.tracker._accumulator, self.name, {})
        accumulator = getattr(self.tracker._accumulator, self.name)
        if field in accumulator:
            values = accumulator[field]
            mean_value = np.mean(values, axis=0)
        else:
            mean_value = np.nan
        return mean_value

    def report_accumulated(self, idx: Union[int, Tuple[int]]):
        """Average all accumulated values, report them, and clear up accumulator."""
        if not hasattr(self.tracker.history, self.name):
            setattr(self.tracker.history, self.name, {})
        # make an accumulator field, if it does not exist
        if not hasattr(self.tracker._accumulator, self.name):
            setattr(self.tracker._accumulator, self.name, {})
        accumulator = getattr(self.tracker._accumulator, self.name)
        for field in accumulator:
            self._report(field, self.calculate_accumulated(field), idx=idx)

        accumulator.clear()

    def _report(
        self,
        field: Union[str, dict],
        value: Union[None, int, float, Iterable, torch.Tensor, np.ndarray] = None,
        idx: Union[int, Tuple[int]] = None,
        meld: bool = False,
    ):
        # `idx`` is only given a default value becaus `value` needs one, but it should
        # be non-None
        assert idx is not None
        if not meld:
            value = np.expand_dims(value, axis=0)

        # make sure all relevant dictionaries exist
        index_names = self.tracker.index_name
        if isinstance(index_names, str):
            # handle single- and multi-index
            index_names = [index_names]
            idx = [idx]
        else:
            index_names = list(index_names)

        target = getattr(self.tracker.history, self.name)
        for key in index_names + [field]:
            if key not in target:
                target[key] = []

        # record the indices -- unless they've been recorded already
        n_field = len(target[field])
        for crt_idx, index_name in zip(idx, index_names):
            recorded_idxs = target[index_name]
            n_idxs = len(recorded_idxs)

            # try to make sure we don't have mismatched entries
            if n_field not in [n_idxs, n_idxs - 1]:
                raise IndexError(
                    f"Tracker: mismatched number of reports in {self.name}"
                )

            # expand indices to array
            if np.size(crt_idx) == 1:
                crt_idx = np.repeat(crt_idx, len(value))
            else:
                crt_idx = np.copy(crt_idx)
            if n_field == n_idxs:
                # add new index entry
                recorded_idxs.append(crt_idx)
            else:
                # the index entry was already added
                # make sure the index is compatible with what we had before
                if not np.array_equal(crt_idx, recorded_idxs[-1]):
                    raise IndexError(f"Tracker: mismatched index values in {self.name}")

        # add new history entry
        target[field].append(value)

    def _accumulate(
        self,
        field: Union[str, dict],
        value: Union[None, int, float, Iterable, torch.Tensor, np.ndarray] = None,
    ):
        target = getattr(self.tracker._accumulator, self.name)
        if field not in target:
            target[field] = []

        target[field].append(value)


class Tracker:
    """Tracker for tensor/array and list-of-tensor/array values.
    
    Call as
        tracker.test.report(idx, "field", value)
    to report an entry in the `"field"` field of the `test` dictionary. The reported
    values can be accessed after `tracker.finalize()` is called, simply by indexing the
    appropriate namespace:
        tracker.test["field"]
    All values will be converted to Numpy arrays for storage. The same values are also
    available in the `self.history` namespace.

    The `report` function adds the `value` to the `"field"`, and makes sure a
    corresponding index `idx` is present in`tracker.test["idx"]`. If there is a value
    present in the `"idx"` field at the right location but it has the wrong value, an
    `IndexError` is raised.

    Note that multiple indices can be used; see `index_name` below.

    Dictionary names can be any valid Python variable name, except for those that are
    already attributes or methods of `Tracker`; see below.

    If `value` is a `np.ndarray`, a copy is added as-is; if it is a `torch.tensor`, it
    is converted to Numpy, then added. If it is an iterable other than tensor or array,
    it's considered a "layered" variable. An entry is recorded for each of its elements,
    with field name generated by adding `":{layer}"` to `"field"`. Note that therefore
    there is a big difference in behavior between reporting the 2d tensor
        torch.FloatTensor([[1.0, 2.0, 3.0]])
    and reporting the list with one 1d tensor entry
        [torch.FloatTensor([1.0, 2.0, 3.0])] .

    Multiple values can be recorded at once by using a `dict` for the first argument:
        tracker.test.report(1, {"foo": 2, "bar": 3})

    There is a way to submit several entries for the same index, which can be useful if
    we wish to, e.g., store a batch of results. This can be achieved by using the `meld`
    argument to `report`:
        tracker.test.report(idx, "field", value, meld=True)
    If `value` is a tensor or array, a new entry is generated for each row in `value`.
    If it is a different iterable, then the same is done *per layer*. This will generate
    as many entries as the length of the tensors/arrays. The constraint here is that, if
    we store more than one field per namespace, all have to have the same batch size.

    When using `meld`, the index (per each level; see below) can either be a scalar or
    a vector of the same sizeas the batch of values that is getting recorded. In the
    former case, the index will be repeated for each sample in the batch.

    Finally, you can accumulate values and then report a summary of the accumulated
    values using the following syntax:
        tracker.name.accumulate(field, value1)
        ...
        tracker.name.accumulate(field, valueN)
        tracker.name.report_accumulated(idx)
    The values `value1`, ..., `valueN` are averaged together and the mean is reported
    and associated to the given index, `idx`.

    You can also read out the accumulated value by using
        tracker.name.calculate_accumulated(field)

    Attributes:
    :param index_name: name or tuple of names used for the index field; if tuple, each
        `idx` argument to the `report` functions should be a tuple, giving the index for
        each level corresponding to the levels in `index_name`
    :param history: namespace holding reported values
    :param finalized: indicator whether `self.finalize()` was called
    """

    def __init__(self, index_name: Union[str, Tuple[str]] = "idx"):
        """Construct tracker.
        
        :param index_name: name of the index field
        """
        self.index_name = index_name
        self.finalized = False
        self.history = SimpleNamespace()
        self._accumulator = SimpleNamespace()

    def finalize(self):
        """Finalize recording, coalescing history into coherent arrays."""
        for field in self.history.__dict__:
            target = getattr(self.history, field)

            # coalesce
            for key in target:
                target[key] = np.concatenate(target[key])

        self.finalized = True

    def set_index_name(self, index_name: Union[str, Tuple[str]]):
        """Set the name(s) used as index."""
        self.index_name = index_name

    def __repr__(self) -> str:
        s = f"Tracker(index_name={self.index_name}, finalized={self.finalized}, "
        hist = "history=namespace("
        for i, name in enumerate(self.history.__dict__):
            if i > 0:
                hist += ", "
            sub_s = f"{name}={{" + ", ".join(
                f'"{_}"' for _ in getattr(self.history, name).keys()
            )
            hist += sub_s + "}"
        hist += ")"
        s += hist + ")"

        return s

    def __getattr__(self, name: str):
        if self.finalized:
            return getattr(self.history, name)
        else:
            return _Reporter(self, name)
