"""The Hypothesis profile ``tests/conftest.py`` loads for every run: a failing property's report carries the blob that
replays it, and nothing else changes from Hypothesis's defaults. The control: the same failing property under a
profile without the blob reports none.

MADDENING's ``verify_node`` battery (under every family case) catches its property's failure and keeps the drawn
counterexample, not the exception, so the blob reaches its report only through that counterexample."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def failure_notes(profile: settings) -> str:
    @settings(profile, database=None)
    @given(st.integers())
    def failing(x):
        assert x < 10

    with pytest.raises(AssertionError) as raised:
        failing()
    return "\n".join(getattr(raised.value, "__notes__", []))


def test_the_profile_prints_the_blob_and_changes_nothing_else():
    loaded, defaults = settings.default, settings.get_profile("default")
    assert loaded.print_blob is True
    for name in ("max_examples", "derandomize", "deadline", "phases", "suppress_health_check"):
        assert getattr(loaded, name) == getattr(defaults, name), name
    assert "@reproduce_failure(" in failure_notes(loaded)


def test_without_it_the_blob_is_not_printed():
    assert "@reproduce_failure(" not in failure_notes(settings(settings.get_profile("default"), print_blob=False))
