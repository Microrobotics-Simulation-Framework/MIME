"""Each test's child processes get a compilation cache of their own, never CI's shared one.

CI points every test process at one JAX compilation-cache directory per shard (``JAX_COMPILATION_CACHE_DIR``), and a
child a test starts (``mime run`` on a pseudo-terminal, ``mime serve``, a Python for an interrupt) inherits it with the
rest of the environment. JAX writes an entry in place, with no temporary file and no lock, and two processes that
compile the same program at the same moment both write it: the second, shorter write leaves a complete zstd frame
followed by the longer one's last bytes, which no process rewrites (``put`` leaves an existing file alone) and which
reads back as ``ZstdError: Unable to decompress Zstandard data: Unknown frame descriptor``. JAX only warns, and the
terminal tests require a child's standard error to be empty, so one such entry, read by a child, failed four tests of
one shard in CI. (The cause and its reproduction: ``tests/test_child_cache.py``.)

So that a test's children cannot read an entry another process damaged, nor damage one for the next test, a test marked
``own_child_cache`` gets a new empty directory in ``JAX_COMPILATION_CACHE_DIR`` for the length of the test, and the
shared one is put back afterwards. It acts only where the variable is set, which is CI's case: elsewhere a child keeps
the user's cache (``mime.env._xla_cache``). The test process itself is unaffected, since JAX read the variable when it
was imported; what changes is what ``os.environ`` holds, which a child copies. A hook, not a fixture, because a
function-scoped fixture would reach the property tests.

Only the tests that require a child's output to be exactly what it should be carry the mark (``tests/cli``: the terminal,
pipe and import-interrupt tests), since a damaged entry's warning on standard error is a failure only there. Every
child of every test was tried first and left a cold run slower: the ``mime serve`` children of ``tests/live`` had left
the programs of A0's apparatus in the shared cache for the in-process tests after them, which then compiled them, and
``test_a_stop_read_either_side_of_a_chunk_reads_alike_and_acts_where_its_reply_allows`` took 25 to 27 s on three cold
runs and failed CI's 20 s line (``tests/duration_allowlist.txt``), where warm it takes under 5 s.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator

import pytest

VARIABLE = "JAX_COMPILATION_CACHE_DIR"
MARK = "own_child_cache"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", f"{MARK}: the test's child processes get a compilation cache of their own "
                            "(tests/_child_cache.py)")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    shared = os.environ.get(VARIABLE)
    if shared is None or item.get_closest_marker(MARK) is None:
        return (yield)
    own = tempfile.mkdtemp(prefix="mime-child-cache-")
    os.environ[VARIABLE] = own
    try:
        return (yield)
    finally:
        os.environ[VARIABLE] = shared
        shutil.rmtree(own, ignore_errors=True)
