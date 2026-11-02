"""A test's child processes read no entry of CI's shared compilation cache (``tests/_child_cache.py``).

The damage this guards against, from the job log of the failed shard (CI run 41515058424, ``default`` on Python 3.14,
shard 7, four tests of ``tests/cli/test_terminal.py``): each child's standard error held ``UserWarning: Error reading
persistent compilation cache entry for 'jit_broadcast_in_dim': ZstdError: Unable to decompress Zstandard data: Unknown
frame descriptor``, and the terminal tests require that standard error to be empty. That job's cache was cold (nothing
restored), so the entry was written during the job. What the message says about it, measured below on the interpreter
that produced it: a partly written entry, or an empty one, reads back as "ended before the end-of-stream marker", not
as "Unknown frame descriptor"; that message is a **complete** zstd frame followed by other bytes. Two processes that
compile one program together both write its entry in place (JAX opens it for writing, truncating, with no lock and no
rename; ``put`` checks only that the file is absent first), and when their frames differ in length, which they can
since each holds that compile's own time, the file is the later writer's frame and the earlier's last bytes. Nothing
rewrites it. (Measured once, outside the suite: two processes calling JAX's own ``LRUCache.put`` on 2000 keys with
frames of 3000 and 3010 random bytes, started together, left 1178 files that the decompressor refuses with exactly the
log's message. A test of the race itself would be a coin; the signature below is not.)

What this does not show is that this race made the log's entry: CI's job did not keep it. The signature fits. So does
a second instance, run 41526160200 (job 125554638606, the same shard and Python, four of the same tests, the entry
``jit_add``): the job's cache mode was again "cold" (the pull request edits ``tests/duration_allowlist.txt``), no step
restored a compilation cache, and a pull request never saves one, so the damaged entry was written during the job and no
Actions cache holds it: a re-run starts empty again and meets the race again or does not, which is how the first
instance passed on its re-run. Only Python 3.14 reports it because JAX compresses with zstd there and with zlib before,
and a zlib stream followed by other bytes is read without complaint (``test_the_damage_is_a_complete_frame...``); the
damage may be as common on 3.12 and 3.13 and is not seen. (That the four terminal tests start their children together on
the shard's four workers is inferred from the tests, not read from the log.) The workers of a shard share the directory
in the same way; a worker only warns (``filterwarnings`` makes no warning an error), and the push run's pruning
(``scripts/prune_jax_cache.py``) drops unreadable entries before the cache is saved for a pull request to restore.
Two children started together inside one test would still share that test's directory, which this does not change (the
terminal tests start their children one after the other).

The controls: the planted entry is made so that it fails every run (a read of a complete frame and two more bytes is
refused by the interpreter's decompressor, not sometimes), the child reads it when it is given the shared directory, and
reads none of it under the plugin.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

try:
    from compression import zstd  # Python 3.14+, which JAX compresses with; before it, with zlib
except ImportError:
    zstd = None

TESTS = Path(__file__).resolve().parent
CHILD = "import jax, jax.numpy as jnp; jax.jit(lambda x: x + 1)(jnp.ones(3)).block_until_ready()"
WARNING = "Error reading persistent compilation cache entry"


def environment(**extra: str) -> dict[str, str]:
    held = {key: value for key, value in os.environ.items()
            if not key.startswith(("PYTEST_", "JAX_", "PYTHONWARNINGS", "MIME_TEST"))}
    return {**held, "JAX_PLATFORMS": "cpu", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1", **extra}


def damaged(entry: bytes) -> bytes:
    """An entry as the log's was where JAX compresses with zstd (a complete frame and more bytes), and cut in half where
    it compresses with zlib (Python before 3.14), since a zlib stream followed by other bytes is read without complaint."""
    return entry + b"\x07\x91" if zstd else entry[: len(entry) // 2]


def child(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=240,
                          check=False)


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> Path:
    """A shared cache directory in which every entry a child writes has been damaged the way the log's was: its frame
    and then more bytes."""
    shared = tmp_path_factory.mktemp("shared-jax-cache")
    warm = child(environment(JAX_COMPILATION_CACHE_DIR=str(shared)))
    assert warm.returncode == 0 and WARNING not in warm.stderr, warm.stderr
    entries = [path for path in shared.iterdir() if path.name.endswith("-cache")]
    assert entries, "the warming child wrote no entry; the plant has nothing to damage"
    for path in entries:
        path.write_bytes(damaged(path.read_bytes()))
    return shared


def test_control_a_child_given_the_planted_shared_directory_reads_the_damage(planted, tmp_path):
    """The plant works, on every run: without the plugin the child warns, in the log's words."""
    copy = tmp_path / "copy"
    copy.mkdir()
    for path in planted.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    done = child(environment(JAX_COMPILATION_CACHE_DIR=str(copy)))
    assert WARNING in done.stderr and (zstd is None or "Unknown frame descriptor" in done.stderr), done.stderr


INNER = """
import os, subprocess, sys
from pathlib import Path
import pytest
pytestmark = pytest.mark.own_child_cache

def test_the_child_reads_none_of_the_shared_directory():
    shared = Path(os.environ["SHARED"])
    before = {path.name: path.read_bytes() for path in shared.iterdir()}
    done = subprocess.run([sys.executable, "-c", __CHILD__], capture_output=True, text=True, timeout=240, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stderr == "", done.stderr
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] != str(shared)
    assert {path.name: path.read_bytes() for path in shared.iterdir()} == before
"""


def test_a_child_of_a_test_run_with_the_plugin_reads_and_writes_none_of_the_shared_directory(planted, tmp_path):
    done = run_inner(tmp_path, INNER, plugin=True, shared=str(planted))
    assert done.returncode == 0, done.stdout + done.stderr


def test_control_the_same_test_without_the_plugin_reads_the_damage(planted, tmp_path):
    """The test above can fail: with the plugin left out, on a copy of the planted directory, it does."""
    copy = tmp_path / "copy"
    copy.mkdir()
    for path in planted.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    done = run_inner(tmp_path, INNER, plugin=False, shared=str(copy))
    assert done.returncode != 0 and WARNING in done.stdout, done.stdout


INNER_PER_TEST = """
import os
from pathlib import Path
import pytest
pytestmark = pytest.mark.own_child_cache

def test_a():
    Path(os.environ["RECORD"], "a").write_text(os.environ["JAX_COMPILATION_CACHE_DIR"])
    assert os.path.isdir(os.environ["JAX_COMPILATION_CACHE_DIR"])

def test_b():
    a = Path(os.environ["RECORD"], "a").read_text()
    now = os.environ["JAX_COMPILATION_CACHE_DIR"]
    assert now not in (a, os.environ["SHARED"]) and a != os.environ["SHARED"]
    assert not os.path.exists(a), "the first test's directory is removed when it ends"
"""
INNER_CONFTEST = """
import os
from pathlib import Path

def pytest_sessionfinish(session):
    Path(os.environ["RECORD"], "after").write_text(os.environ.get("JAX_COMPILATION_CACHE_DIR", "<unset>"))
"""


def run_inner(tmp_path: Path, source: str, *, plugin: bool, shared: str | None) -> subprocess.CompletedProcess[str]:
    (tmp_path / "test_inner.py").write_text(textwrap.dedent(source).replace("__CHILD__", repr(CHILD)), encoding="utf-8")
    (tmp_path / "conftest.py").write_text(textwrap.dedent(INNER_CONFTEST), encoding="utf-8")
    (tmp_path / "record").mkdir(exist_ok=True)
    env = environment(RECORD=str(tmp_path / "record"), SHARED=str(shared),
                      PYTHONPATH=os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]),
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if shared is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = shared
    plugins = ["-p", "_child_cache"] if plugin else []
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *plugins,
                           str(tmp_path / "test_inner.py")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240, check=False)


def test_each_test_has_a_directory_of_its_own_and_the_shared_one_is_back_after_the_run(tmp_path):
    shared = str(tmp_path / "shared")
    done = run_inner(tmp_path, INNER_PER_TEST, plugin=True, shared=shared)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "record" / "after").read_text() == shared


def test_control_without_the_plugin_the_same_tests_see_the_shared_directory(tmp_path):
    """The two tests above can fail: with the plugin left out they do, in the test that names the directory."""
    done = run_inner(tmp_path, INNER_PER_TEST, plugin=False, shared=str(tmp_path / "shared"))
    assert done.returncode != 0 and "AssertionError" in done.stdout, done.stdout


def test_a_test_without_the_mark_keeps_the_shared_directory(tmp_path):
    """The mark is the whole reach: the other tests' children keep sharing the worker's cache, which a cold run's
    in-process tests are quicker for (``tests/_child_cache.py``)."""
    shared = str(tmp_path / "shared")
    done = run_inner(tmp_path, f"""
        import os
        def test_unmarked():
            assert os.environ["JAX_COMPILATION_CACHE_DIR"] == {shared!r}
        """, plugin=True, shared=shared)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_tests_that_require_a_clean_child_are_marked():
    """The terminal tests failed on a damaged entry's warning, and the import-interrupt tests require the same standard
    error."""
    import re
    for name in ("tests/cli/test_terminal.py", "tests/cli/test_import_interrupt.py"):
        assert re.search(r"^pytestmark = .*pytest\.mark\.own_child_cache", (TESTS.parent / name).read_text(
            encoding="utf-8"), re.MULTILINE), name


def test_where_ci_sets_no_shared_directory_the_plugin_sets_none(tmp_path):
    done = run_inner(tmp_path, """
        import os
        import pytest
        pytestmark = pytest.mark.own_child_cache
        def test_unset():
            assert "JAX_COMPILATION_CACHE_DIR" not in os.environ
        """, plugin=True, shared=None)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "record" / "after").read_text() == "<unset>"


def test_the_damage_is_a_complete_frame_and_more_bytes_not_a_partial_write():
    """The message of the log is the decompressor's for a complete zstd frame followed by other bytes. A partly written
    entry and an empty one are refused with another message, so neither made the log's entry. (Before Python 3.14 JAX
    compresses with zlib, which reads a stream followed by other bytes without complaint: there only the cut entry is
    refused, and that is all this can show.)"""
    import zlib

    if zstd is None:
        whole = zlib.compress(b"x" * 3000)
        zlib.decompress(whole + b"\x07\x91")
        with pytest.raises(zlib.error):
            zlib.decompress(whole[: len(whole) // 2])
        return
    frame = zstd.compress(b"\x00\x00\x00\x01" + os.urandom(64) + b"x" * 3000)
    with pytest.raises(zstd.ZstdError, match="Unknown frame descriptor"):
        zstd.decompress(frame + b"\x07\x91")
    for cut in (b"", frame[:4], frame[: len(frame) // 2]):
        with pytest.raises(zstd.ZstdError, match="before the end-of-stream marker"):
            zstd.decompress(cut)
