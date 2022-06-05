"""Define a class to track model tensors."""


import torch

from types import SimpleNamespace
from typing import Union, Iterable, Callable


def _dispatch_values(
    reporter: Callable,
    field: Union[str, dict],
    value: Union[None, int, float, Iterable, torch.Tensor] = None,
    **kwargs,
):
    """Pass values to a reporter, handling lists or tensors, as well as dicts for
    multi-parameter reports.
    """
    if not isinstance(field, str):
        if value is not None:
            raise ValueError("Tracker: value used with multi-parameter report")

        for crt_key, crt_value in field.items():
            reporter(crt_key, crt_value, **kwargs)
        return

    if not torch.is_tensor(value):
        if isinstance(value, int):
            value = torch.LongTensor([value])
        elif hasattr(value, "__iter__"):
            for i, sub_value in enumerate(value):
                reporter(f"{field}:{i}", sub_value, **kwargs)
            return
        else:
            value = torch.FloatTensor([value])
    else:
        value = value.detach().cpu().clone()

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

    def report(self, idx: int, *args, **kwargs):
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

    def report_accumulated(self, idx: int):
        """Average all accumulated values, report them, and clear up accumulator."""
        accumulator = getattr(self.tracker._accumulator, self.name)
        if not hasattr(self.tracker.history, self.name):
            setattr(self.tracker.history, self.name, {})
        for field, values in accumulator.items():
            mean_value = torch.cat(values).mean(dim=0)
            self._report(field, mean_value, idx=idx)

        accumulator.clear()

    def _report(
        self,
        field: Union[str, dict],
        value: Union[None, int, float, Iterable, torch.Tensor] = None,
        idx: int = None,
        meld: bool = False,
    ):
        assert idx is not None
        if not meld:
            value.unsqueeze_(0)

        index_name = self.tracker.index_name
        target = getattr(self.tracker.history, self.name)
        for key in [index_name, field]:
            if key not in target:
                target[key] = []

        idxs = target[index_name]

        # try to make sure we don't have mismatched entries
        if len(target[field]) not in [len(idxs), len(idxs) - 1]:
            raise IndexError(f"Tracker: mismatched number of reports in {self.name}")

        crt_idxs = torch.LongTensor(len(value) * [idx])
        if len(target[field]) == len(idxs):
            # add new index entry
            idxs.append(crt_idxs)
        else:
            # the index entry was already added
            # make sure the index is compatible with what we had before
            if len(crt_idxs) != len(idxs[-1]) or torch.any(crt_idxs != idxs[-1]):
                raise IndexError(f"Tracke: mismatched index values in {self.name}")

        # add new history entry
        target[field].append(value)

    def _accumulate(
        self,
        field: Union[str, dict],
        value: Union[None, int, float, Iterable, torch.Tensor] = None,
    ):
        target = getattr(self.tracker._accumulator, self.name)
        if field not in target:
            target[field] = []

        target[field].append(value.unsqueeze(0))


class Tracker:
    """Tracker for tensor and list-of-tensor values.
    
    Call as
        tracker.test.report("field", idx, value)
    to report an entry in the `"field"` field of the `test` dictionary. The reported
    values can be accessed after `tracker.finalize()` is called, simply by indexing the
    appropriate namespace:
        tracker.test["field"]
    The same values are also available in the `self.history` namespace.

    The `report` function adds the `value` to the `"field"`, and makes sure a
    corresponding index `idx` is present in`tracker.test["idx"]`. If there is a value
    present in the `"idx"` field at the right location but it has the wrong value, an
    `IndexError` is raised.

    Dictionary names can be any valid Python variable name, except for those that are
    already attributes or methods of `Tracker`; see below.

    If `value` is a `Tensor`, it is added as-is. If it is an iterable other than
    `Tensor`, it is considered a "layered" variable. An entry is recorded for each of
    its elements, with field name generated by adding `":{layer}"` to `"field"`. Note
    that therefore there is a big difference in behavior between reporting the 2d tensor
        torch.FloatTensor([[1.0, 2.0, 3.0]])
    and reporting the list with one tensor entry
        [torch.FloatTensor([1.0, 2.0, 3.0])] .

    Multiple values can be recorded at once by using a `dict` for the first argument:
        tracker.test.report({"foo": 2, "bar": 3}, idx=1)

    There is a way to submit several entries for the same index, which can be useful if
    we wish to, e.g., store a batch of results. This can be achieved by using the `meld`
    argument to `report`:
        tracker.test.report("field", idx, value, meld=True)
    If `value` is a tensor, a new entry is generated for each row in `value`. If it is a
    different iterable, then the same is done *per layer*. This will generate as many
    entries as the length of the tensors. The constraint here is that, if we store more
    than one field per namespace, all have to have the same batch size.

    Finally, you can accumulate values and then report a summary of the accumulated
    values using the following syntax:
        tracker.name.accumulate(field, value1)
        ...
        tracker.name.accumulate(field, valueN)
        tracker.name.report_accumulated(idx)
    The values `value1`, ..., `valueN` are averaged together and the mean is reported
    and associated to the given index, `idx`.

    Attributes:
    :param index_name: name used for the index field
    :param history: namespace holding reported values
    :param finalized: indicator whether `self.finalize()` was called
    """

    def __init__(self, index_name: str = "idx"):
        """Construct tracker.
        
        :param index_name: name of the index field
        """
        self.index_name = index_name
        self.finalized = False
        self.history = SimpleNamespace()
        self._accumulator = SimpleNamespace()

    def finalize(self):
        """Finalize recording, coalesce history into coherent tensors."""
        for field in self.history.__dict__:
            target = getattr(self.history, field)

            # coalesce
            for key in target:
                target[key] = torch.cat(target[key])

        self.finalized = True

    def set_index_name(self, index_name: str):
        """Set the name used as index."""
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
