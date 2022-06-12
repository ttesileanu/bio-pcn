import pytest

from cpcn.util import pretty_size


def test_pretty_size_small():
    assert pretty_size(23) == "23.0B"


def test_pretty_size_kb():
    assert pretty_size(1024) == "1.0KB"


def test_pretty_size_multiplier():
    assert pretty_size(16384, multiplier=1000) == "16.4KB"


def test_pretty_size_large():
    assert pretty_size(1_234_567_890_123_245, multiplier=1000) == "1234.6TB"
