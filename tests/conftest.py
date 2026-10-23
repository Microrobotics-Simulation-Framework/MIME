"""Each test's CPU time, written beside its wall time in the JUnit XML that CI's durations report reads
(``scripts/report_test_durations.py``), as OpenMaterica's ``tests/conftest.py`` records it.

CI runs the tests on pytest-xdist workers that share a runner's cores, so a test's wall time partly measures the
others; its CPU time much less. ``cpu_seconds`` is the CPU time of the test's own process (its xdist worker, or the
pytest process in a serial run), and ``children_cpu_seconds`` that of the processes it started and waited for, which
the first leaves out: a test that loads a saved experiment in a fresh process or runs pytest in a subprocess spends
its time there. Both span setup, call and teardown.
"""

import sys
import time

import pytest
from hypothesis import settings

#: The Hypothesis profile every run uses, CI's included: the profile Hypothesis would load here (its ``ci`` profile on
#: CI, which derandomises, and ``default`` elsewhere), kept as ``mime-baseline``, with ``print_blob`` set, so a failing
#: property prints the ``@reproduce_failure`` blob that replays it. It changes nothing else: no example count, no
#: derandomisation, no deadline.
settings.register_profile("mime-baseline", parent=settings.default)
settings.register_profile("mime", parent=settings.get_profile("mime-baseline"), print_blob=True)
settings.load_profile("mime")

if sys.platform != "win32":
    import resource

CPU_AT_SETUP = pytest.StashKey[tuple[float, float | None]]()


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
