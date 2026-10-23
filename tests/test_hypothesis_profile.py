"""The Hypothesis profile ``tests/conftest.py`` loads for every run: a failing property's report carries the blob that
replays it, every run draws new examples (CI's included, where Hypothesis's own ``ci`` profile derandomises), and
nothing else changes from the profile Hypothesis would have loaded. The controls: the same failing property under a
profile without the blob reports none, and Hypothesis's ``ci`` profile fails the check that the profile draws anew.

MADDENING's ``verify_node`` battery (under every family case) catches its property's failure and keeps the drawn
counterexample, not the exception, so the blob reaches its report only through that counterexample."""

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

CONFTEST = Path(__file__).with_name("conftest.py")


def failure_notes(profile: settings) -> str:
    @settings(profile, database=None)
    @given(st.integers())
    def failing(x):
        assert x < 10

    with pytest.raises(AssertionError) as raised:
        failing()
    return "\n".join(getattr(raised.value, "__notes__", []))


def test_the_profile_prints_the_blob_draws_anew_and_changes_nothing_else():
    """Against what Hypothesis would have loaded here: its ``ci`` profile on CI, ``default`` elsewhere."""
    loaded, baseline = settings.default, settings.get_profile("mime-baseline")
    assert loaded.print_blob is True
    assert loaded.derandomize is False
    for name in ("max_examples", "deadline", "phases", "suppress_health_check", "database"):
        assert getattr(loaded, name) == getattr(baseline, name), name
    assert "@reproduce_failure(" in failure_notes(loaded)


def test_without_it_the_blob_is_not_printed():
    assert "@reproduce_failure(" not in failure_notes(settings(settings.get_profile("default"), print_blob=False))


_READ_PROFILES = f"""
import json, runpy
runpy.run_path({str(CONFTEST)!r})
from hypothesis import settings
def read(s):
    return {{"derandomize": s.derandomize, "print_blob": s.print_blob, "database": repr(s.database)}}
print(json.dumps({{"loaded": read(settings.default), "baseline": read(settings.get_profile("mime-baseline")),
                  "ci": read(settings.get_profile("ci"))}}))
"""


@pytest.fixture(scope="module")
def on_ci() -> dict[str, dict[str, object]]:
    """The profile the conftest loads, the one it starts from, and Hypothesis's own ``ci`` profile, as a fresh
    interpreter run with ``CI=true`` holds them after running the conftest."""
    run = subprocess.run([sys.executable, "-c", _READ_PROFILES], env={**os.environ, "CI": "true"},
                         capture_output=True, text=True, check=True)
    return json.loads(run.stdout.splitlines()[-1])


def assert_draws_anew_and_prints_the_blob(profile: Mapping[str, object]) -> None:
    assert profile["derandomize"] is False, "derandomised: every run draws the same examples"
    assert profile["print_blob"] is True, "a failing property would not print the blob that replays it"


def test_on_ci_the_profile_draws_anew_prints_the_blob_and_keeps_no_database(on_ci):
    assert on_ci["baseline"] == on_ci["ci"], "Hypothesis did not load its ci profile, so this tests nothing"
    assert_draws_anew_and_prints_the_blob(on_ci["loaded"])
    assert on_ci["loaded"]["database"] == "None"


def test_hypothesis_own_ci_profile_fails_that_check(on_ci):
    with pytest.raises(AssertionError, match="derandomised"):
        assert_draws_anew_and_prints_the_blob(on_ci["ci"])
