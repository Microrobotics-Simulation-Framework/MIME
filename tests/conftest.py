"""Each test's CPU time, written beside its wall time in the JUnit XML that CI's durations report reads
(``scripts/report_test_durations.py``), as OpenMaterica's ``tests/conftest.py`` records it.

CI runs the tests on pytest-xdist workers that share a runner's cores, so a test's wall time partly measures the
others; its CPU time much less. ``cpu_seconds`` is the CPU time of the test's own process (its xdist worker, or the
pytest process in a serial run), and ``children_cpu_seconds`` that of the processes it started and waited for, which
the first leaves out: a test that loads a saved experiment in a fresh process or runs pytest in a subprocess spends
its time there. Both span setup, call and teardown.
"""

import os
import sys
import time

#: One BLAS thread per process. JAX's CPU linear algebra (an LU, a Cholesky) runs through SciPy's OpenBLAS, whose
#: threads wait by spinning: with pytest-xdist's workers on every core, a test of many small solves ran seven times
#: slower than on one free core (the MRS's family suite, measured with three cores busy: 303 s, and 35 s with one
#: thread). Set before SciPy is first imported, which is when OpenBLAS reads it; a value already set is kept.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#: The suite runs MIME's errors in full: every frame and every chained error, so that a failing test's error shows
#: where it came from. The tests of the short form set it themselves, in the processes they start.
os.environ["MIME_TRACEBACK"] = "full"

import pytest  # noqa: E402
from hypothesis import settings  # noqa: E402

#: The Hypothesis profile every run uses, CI's included: the profile Hypothesis would load here (``default``, or on CI
#: its ``ci`` profile: no database, no deadline, no too-slow health check), kept as ``mime-baseline``, with two
#: changes. ``print_blob`` is set, so a failing property prints the ``@reproduce_failure`` blob that replays it. And
#: ``derandomize`` is off, which the ``ci`` profile turns on: a property test is worth running on CI for its new draws
#: on every run. No example count changes.
settings.register_profile("mime-baseline", parent=settings.default)
settings.register_profile("mime", parent=settings.get_profile("mime-baseline"), print_blob=True, derandomize=False)
settings.load_profile("mime")

if sys.platform != "win32":
    import resource

CPU_AT_SETUP = pytest.StashKey[tuple[float, float | None]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--mime-shard", metavar="i/N", help="keep CI shard i of N's test files (tests/_sharding.py)")
    parser.addoption("--mime-shard-record", metavar="FILE", help="write what the shard collected and kept, as JSON")


def pytest_configure(config: pytest.Config) -> None:
    """CI's plugins: ``--mime-shard i/N`` keeps one CI job's share of the test files (``tests/_sharding.py``), and
    ``MIME_TEST_JAX_TIMING=1`` records each test's JAX tracing, lowering, compiling and cache reads beside its times
    (``tests/_jax_timing.py``), which ``scripts/report_test_durations.py`` reads."""
    spec = config.getoption("--mime-shard")
    if spec:
        import _sharding

        _sharding.register(config, spec)
    if os.environ.get("MIME_TEST_JAX_TIMING") == "1":
        import _jax_timing

        _jax_timing.register(config)


def children_cpu() -> float | None:
    """The CPU time of the processes this one started that have ended and been waited for; ``None`` where Python
    cannot tell (Windows)."""
    if sys.platform == "win32":
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    item.stash[CPU_AT_SETUP] = (time.process_time(), children_cpu())


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    # What `record_property` does, at the last moment it can: the XML takes a test's properties from its teardown
    # report, which is made after the teardown and copies them when it is made.
    if call.when == "teardown" and CPU_AT_SETUP in item.stash:
        own, children = item.stash[CPU_AT_SETUP]
        item.user_properties.append(("cpu_seconds", f"{time.process_time() - own:.3f}"))
        children_now = children_cpu()
        if children is not None and children_now is not None:
            item.user_properties.append(("children_cpu_seconds", f"{children_now - children:.3f}"))
