import pytest
from biobench.utils import which


def test_which():
    assert which("python") is not None
    assert which("nonexistent") is None
