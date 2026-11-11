"""Each test's CPU time, written beside its wall time in the JUnit XML that CI's durations report reads
(``scripts/report_test_durations.py``), as OpenMaterica's ``tests/conftest.py`` records it.

CI runs the tests on pytest-xdist workers that share a runner's cores, so a test's wall time partly measures the
others; its CPU time much less. ``cpu_seconds`` is the CPU time of the test's own process (its xdist worker, or the
pytest process in a serial run), and ``children_cpu_seconds`` that of the processes it started and waited for, which
the first leaves out: a test that loads a saved experiment in a fresh process or runs pytest in a subprocess spends
its time there. Both span setup, call and teardown.
"""

import contextlib
import ctypes
import gc
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

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


@pytest.fixture(autouse=True, scope="session")
def user_xla_cache_in_the_sessions_own_directory(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """``mime run`` and ``mime serve`` keep XLA's cache in the user's cache directory unless JAX's own setting is
    already made, as CI's is. Run here without it, the suite's commands would fill the developer's real cache; they
    fill the session's own directory instead. ``tests/env/test_user_xla_cache.py`` puts the real path back. Session
    scoped, so that no property test sees a function-scoped fixture."""
    with pytest.MonkeyPatch.context() as patch:
        root = tmp_path_factory.mktemp("user-xla-cache")
        patch.setattr("mime.env._xla_cache.user_xla_cache_root", lambda: root)
        yield


@pytest.fixture(autouse=True)
def licence_acceptances_in_the_tests_own_file(tmp_path_factory: pytest.TempPathFactory,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """A licence a test accepts at a prompt is written to the acceptances file (``mime.assets._acceptances``), which is the
    user's config directory's unless ``MIME_LICENCE_ACCEPTANCES`` names another. Each test has a file of its own, none yet
    written, so that no test is excused a question by another's yes, and the developer's own acceptances are never read
    or written."""
    monkeypatch.setenv("MIME_LICENCE_ACCEPTANCES", str(tmp_path_factory.mktemp("licences") / "licences.json"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--mime-shard", metavar="i/N", help="keep CI shard i of N's test files (tests/_sharding.py)")
    parser.addoption("--mime-shard-record", metavar="FILE", help="write what the shard collected and kept, as JSON")


def pytest_configure(config: pytest.Config) -> None:
    """CI's plugins: ``--mime-shard i/N`` keeps one CI job's share of the test files (``tests/_sharding.py``), and
    ``MIME_TEST_JAX_TIMING=1`` records each test's JAX tracing, lowering, compiling and cache reads beside its times
    (``tests/_jax_timing.py``), which ``scripts/report_test_durations.py`` reads. A marked test's child
    processes get a compilation cache of their own where CI sets a shared one (``tests/_child_cache.py``)."""
    spec = config.getoption("--mime-shard")
    if spec:
        import _sharding

        _sharding.register(config, spec)
    if os.environ.get("MIME_TEST_JAX_TIMING") == "1":
        import _jax_timing

        _jax_timing.register(config)
    import _child_cache
    import _collection_guard

    config.pluginmanager.register(_child_cache, "mime-child-cache")
    config.pluginmanager.register(_collection_guard, "mime-collection-guard")


MEMORY_HEAVY = "memory_heavy"


def memory_lock_path() -> Path | None:
    """The file the workers of one xdist run lock to take their ``memory_heavy`` tests one at a time; ``None`` in a
    run with one process, which runs its tests one at a time anyway. xdist gives every worker of a run the same
    ``PYTEST_XDIST_TESTRUNUID``, so two runs on one machine do not wait for each other."""
    if "PYTEST_XDIST_WORKER" not in os.environ:
        return None
    run = os.environ.get("PYTEST_XDIST_TESTRUNUID") or f"parent{os.getppid()}"
    return Path(tempfile.gettempdir()) / f"mime-memory-heavy-{run}.lock"


@contextlib.contextmanager
def held_exclusively(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created if absent) for the block, waiting for whoever holds it. The lock
    belongs to the open file, so a worker that dies releases it."""
    with open(path, "a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue  # LK_LOCK gives up after ten seconds; the wait goes on
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


#: The tests that compile a union of tubes (its seams' march, its corners), by their files and by the names a case
#: of another file gives them: each keeps its programs, some hundreds of MB, in its worker's caches for the rest of the
#: run unless they are released, and a worker that ran several held several GB (CI's runners were lost at the end of a
#: shard twice, runs 42964681870 and 42970695440, and in every stress run of 42979996849). Clearing all of JAX's caches
#: instead made a later test that reuses other programs rebuild them (the shape suite's share of probes, run
#: 42989831801).
TREE_FILES = frozenset({"tests/flow/test_fvm_general_wall.py", "tests/flow/test_fvm_tree_declaration.py",
                        "tests/scene/test_union_reverse.py", "tests/scene/test_union_tie.py",
                        "tests/scene/test_tube_tree.py", "tests/scene/test_exact_union.py",
                        "tests/scene/test_union_brute_force.py", "tests/scene/test_union_continuity.py",
                        "tests/two_builds/test_tree_walk.py"})
#: The shape suite's tube tree tests are not among them (its heavy ones are marked ``memory_heavy``).
TREE_NAMES = ("tree_flow", "geometry tube_tree")


def compiles_a_tree(item: pytest.Item) -> bool:
    return item.nodeid.split("::")[0] in TREE_FILES or any(name in item.nodeid for name in TREE_NAMES)


def release_programs() -> None:
    """Drop the union's own caches of compiled functions, then return the memory: JAX's caches hold a function's
    programs by the function, so they should go with it [inference: not measured alone]; the worker's other programs
    are kept, which the tests after it may reuse."""
    union = sys.modules.get("mime.scene._union")
    tree = sys.modules.get("mime.scene._tube_tree")
    for module, names in ((union, ("_compiled", "_checked")), (tree, ("_lumen", "_structural_lumen"))):
        for name in names:
            cached = getattr(module, name, None) if module is not None else None
            if cached is not None and hasattr(cached, "cache_clear"):
                cached.cache_clear()
    return_memory()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """A test marked ``memory_heavy`` runs, setup to teardown, while no other worker of the run is running one: CI's
    runner holds four workers in 16 GB, and the heavy tests peak at several GB each (``tests/memory_peaks.json``).
    Whatever xdist's ``--dist`` mode sends where, the marked tests take turns; the other tests go on beside them.

    The memory a heavy test frees stays with its worker unless it is handed back: the worker that ran the 8 GB arm
    battery held 8 GB for the rest of the run, 5 GB of it free and unreturned, which is as much to a runner as a second
    heavy test. So the test's garbage is collected and glibc's free memory returned (``malloc_trim``) before the lock
    is let go. A test that compiles a tube tree's union releases its programs after it (:func:`release_programs`)."""
    if item.get_closest_marker(MEMORY_HEAVY) is None:
        if not compiles_a_tree(item):
            return (yield)
        try:
            return (yield)
        finally:
            release_programs()
    path = memory_lock_path()
    with held_exclusively(path) if path is not None else contextlib.nullcontext():
        try:
            return (yield)
        finally:
            release_programs() if compiles_a_tree(item) else return_memory()


def return_memory() -> None:
    """Collect garbage and give glibc's free memory back to the system (a no-op where there is no ``malloc_trim``)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


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
