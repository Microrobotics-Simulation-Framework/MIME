# A copy of MADDENING's tests/_jax_timing.py at its tag v0.4.2 (commit 8a6c9fa8066646ab57b55052a300e6badd85c658),
# Apache-2.0, beside the copy of scripts/report_test_durations.py that reads its properties. Changes to MADDENING's
# copy: this header; the variable that switches it on is MIME_TEST_JAX_TIMING, and the plugin registers as
# "mime-jax-timing"; the module is imported as `_jax_timing` (tests/ is not a package); the tests that guard it are
# in tests/test_report_test_durations.py.
"""Split each test's wall-clock into JAX tracing, lowering and compiling.

A test that is slow on CI is slow for one of three reasons, and each has a
different fix:

* **compiling** -- XLA backend compilation.  A persistent compilation
  cache removes it; so does not recompiling for every Hypothesis example.
* **tracing / lowering** -- Python-side program construction.  No cache
  removes it; rebuilding or retracing a large program per test does.
* **running** -- everything else: executing the program, Python overhead,
  I/O, subprocesses.  An un-jitted ``for`` loop over ``node.update`` lands
  here, thousands of op-by-op dispatches.

JAX already times the first two through :mod:`jax.monitoring`; this module
listens, sums per test (setup + call + teardown, the same span pytest's
JUnit ``time`` covers) and attaches the sums to the test as
``user_properties``, which ``--junitxml`` writes as ``<property>``
elements.  ``scripts/report_test_durations.py`` reads them.

It also counts the processes a test starts (``subprocesses``), from
Python's audit events (PEP 578: ``subprocess.Popen``, ``os.system``,
``os.fork``, ...).  A child's JAX events never reach this process, so a
test whose JAX work runs in a child records no compile here although a
warm cache -- whose directory the child inherits -- may still speed it
up; the report uses the count to say so instead of calling the test
uncacheable.  Not seen: ``multiprocessing`` children started with the
``spawn`` or ``forkserver`` method, which fork without an audit event
(the ``fork`` method raises ``os.fork`` and is counted).

Registered from ``tests/conftest.py`` only when
``MIME_TEST_JAX_TIMING=1`` (CI sets it), so local runs are untouched.
Each xdist worker is its own process and records its own tests.
"""

from __future__ import annotations

import collections
import sys

import pytest

#: jax.monitoring duration events -> the property they add to.  If JAX
#: renames one, ``tests/test_report_test_durations.py`` fails
#: rather than this module silently reporting zero.
DURATION_EVENTS = {
    "/jax/core/compile/jaxpr_trace_duration": "jax_trace_s",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "jax_lower_s",
    "/jax/core/compile/backend_compile_duration": "jax_compile_s",
    "/jax/compilation_cache/cache_retrieval_time_sec": "jax_cache_read_s",
}
#: jax.monitoring count events -> property.
COUNT_EVENTS = {
    "/jax/compilation_cache/cache_hits": "jax_cache_hits",
    "/jax/compilation_cache/cache_misses": "jax_cache_misses",
}
#: Audit events that start another process -> ``subprocesses``.  If
#: Python renames one, ``tests/test_report_test_durations.py``
#: fails rather than the count silently reading zero.
SPAWN_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.fork", "os.forkpty",
    "os.posix_spawn", "os.spawn", "os.exec",
})
PROPERTIES = (*DURATION_EVENTS.values(), *COUNT_EVENTS.values(), "subprocesses")


class JaxTiming:
    """Accumulates the JAX events seen since the last :meth:`reset`."""

    def __init__(self):
        self.totals = collections.Counter()

    def on_duration(self, event, duration_secs, **_):
        prop = DURATION_EVENTS.get(event)
        if prop is not None:
            self.totals[prop] += duration_secs

    def on_event(self, event, **_):
        prop = COUNT_EVENTS.get(event)
        if prop is not None:
            self.totals[prop] += 1

    def on_audit(self, event, args):
        # An audit hook runs inside the audited call and must never raise.
        if event in SPAWN_EVENTS:
            self.totals["subprocesses"] += 1

    def reset(self):
        self.totals.clear()

    def properties(self):
        """``(name, value)`` pairs for every property, zeros included."""
        return [(p, round(self.totals[p], 4)) for p in PROPERTIES]


class Plugin:
    """The pytest side: reset before each test, attach before teardown's report."""

    def __init__(self, timing: JaxTiming):
        self.timing = timing

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        # tryfirst: runs before the item's fixtures are set up, so a module fixture's
        # compile is charged to the first test that needs it -- which is
        # where pytest's own duration puts it too.
        self.timing.reset()

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        # JUnit XML takes a test's properties from its *teardown* report,
        # and a report copies ``item.user_properties`` when it is built, so
        # they must be attached before that.  Returning None lets pytest's
        # own implementation build the report.
        if call.when == "teardown":
            item.user_properties.extend(self.timing.properties())
        return None


def register(config):
    """Start listening and register the plugin; called from conftest."""
    from jax import monitoring

    timing = JaxTiming()
    monitoring.register_event_duration_secs_listener(timing.on_duration)
    monitoring.register_event_listener(timing.on_event)
    # Audit hooks cannot be removed; this one lives as long as the session.
    sys.addaudithook(timing.on_audit)
    config.pluginmanager.register(Plugin(timing), "mime-jax-timing")
    return timing
