"""Two commands writing one program into the user's XLA cache at once do not leave a damaged entry, because MIME's cache
goes through JAX's own locked path.

JAX's ``LRUCache.put`` writes an entry in place, with no lock, unless a size limit is set (``jax_compilation_cache_max_size``):
then every read and write of an entry takes a file lock (``<directory>/.lockfile``, through the ``filelock`` package) and
the least recently used entry goes when one is added. Without the lock, two processes that compile one program at once
both write its entry, and when their frames differ in length the file is the shorter frame followed by the longer one's tail,
which JAX never rewrites and which reads back as ``Unknown frame descriptor`` (Python 3.14, zstd) on every later run.
``user_xla_cache`` therefore gives JAX its cap as that limit.

The gate, and its control, are deterministic: a child compiles one program in ``user_xla_cache`` while the test holds the
directory's lock. With MIME's configuration the child's first read waits for the lock (it has not finished after the test has
held it for a time, and finishes once it is released); with MIME's setting reverted (JAX's limit unset, the child's own
arm ``base``) the child compiles and finishes while the lock is held. A change that stops MIME's cache going through
JAX's locked path fails the first, and the second shows the first can fail.

The race test is a statistical check, not a gate: two children, started at one moment through MIME's own
``user_xla_cache``, compile 150 programs into one fresh directory, held in step by a file barrier before each program so that
their writes overlap, and the damage is counted by a decompressor of the test's own. Measured, with JAX's limit unset (the
base before this change): Python 3.14, 13 trials of 152 entries, 121 damaged entries, damage in 13 of 13 trials; Python 3.12
(zlib, where JAX reads a stream followed by other bytes without complaint and the test counts one as damage), 16 damaged
entries, damage in 9 of 13 trials; and on one CI runner on Python 3.12 six trials of the same harness found none (the race
was not reached there). With the limit set, as MIME sets it: none in 13 of 13 trials on each of the two Pythons. So the race
test can pass where the race is not reached; it fails if the lock stops protecting what it protected when measured."""

import os
import subprocess
import sys
import textwrap
import time
import zlib
from pathlib import Path

import filelock
import jax
import pytest

from mime.env import _xla_cache
from mime.env._xla_cache import DIRECTORY_LIMIT_BYTES, USER_CACHE_CAP_BYTES, point_xla_cache, prune_to_cap, user_xla_cache

try:
    from compression import zstd  # Python 3.14+, which JAX compresses with
except ImportError:
    zstd = None

PROGRAMS = 150
DEPTH = 8

CHILD = textwrap.dedent('''
    import sys, time
    from pathlib import Path
    root, start_at, programs, depth, me, gate = sys.argv[1:]
    other = "b" if me == "a" else "a"
    import jax, jax.numpy as jnp
    from mime.env import _xla_cache

    def program(i):
        def chain(x):
            for j in range(int(depth)):
                x = jnp.sin(x @ x.T + (i + 1) * 0.37 + j) * 0.5
            return x
        return jax.jit(chain)

    with _xla_cache.user_xla_cache(root=Path(root)) as held:
        assert held is not None
        while time.time() < float(start_at):
            pass
        x = jnp.ones((8, 8))
        for i in range(int(programs)):
            (Path(gate) / f"{me}{i}").touch()
            while not (Path(gate) / f"{other}{i}").exists():
                pass
            program(i)(x).block_until_ready()
''')

#: One compile under the user's cache that says where its directory is, waits to be told to go, says it is about to
#: compile, and says it is done. ``base`` reverts MIME's limit as the first thing in the block.
LOCKED_CHILD = textwrap.dedent('''
    import sys, time
    from pathlib import Path
    root, arm, folder = sys.argv[1:]
    folder = Path(folder)
    import jax, jax.numpy as jnp
    from mime.env import _xla_cache

    with _xla_cache.user_xla_cache(root=Path(root)) as held:
        assert held is not None
        if arm == "base":
            jax.config.update("jax_compilation_cache_max_size", -1)
            _xla_cache.point_xla_cache(str(held))
        (folder / "directory").write_text(str(held))
        while not (folder / "go").exists():
            time.sleep(0.01)
        (folder / "reached").touch()
        jax.jit(lambda x: jnp.sin(x) + 3.5)(jnp.ones(5)).block_until_ready()
        (folder / "done").touch()
''')


def child_environment() -> dict[str, str]:
    held = {key: value for key, value in os.environ.items()
            if not key.startswith(("JAX_", "PYTEST_", "XDG_CACHE", "XLA_FLAGS", "MIME_TEST"))}
    return {**held, "JAX_PLATFORMS": "cpu"}


def entries_under(root: Path) -> list[Path]:
    return [path for path in root.rglob("*-cache") if path.is_file()]


def not_one_stream(data: bytes) -> bool:
    """Whether a file is not exactly one compressed stream, by the test's own call: the interpreter's zstd decompressor
    (which refuses a frame followed by other bytes, as JAX's reader does) or, where JAX compresses with zlib, a zlib
    object that must reach its end with nothing left over (JAX's own read takes the leftover and says nothing)."""
    try:
        if zstd is not None:
            zstd.decompress(data)
        else:
            reader = zlib.decompressobj()
            reader.decompress(data)
            return not reader.eof or bool(reader.unused_data)
    except Exception:  # noqa: BLE001 - any refusal
        return True
    return False


def damaged_entries(root: Path) -> int:
    return sum(not_one_stream(path.read_bytes()) for path in entries_under(root))


@pytest.fixture
def jax_cache_unset():
    """JAX's own cache settings off for the test (CI sets them for its lanes, and ``user_xla_cache`` leaves a set directory
    alone), and put back."""
    names = ("jax_persistent_cache_min_compile_time_secs", "jax_persistent_cache_min_entry_size_bytes",
             "jax_compilation_cache_max_size")
    held_dir, held = jax.config.jax_compilation_cache_dir, {name: getattr(jax.config, name) for name in names}
    point_xla_cache(None)
    yield
    for name, value in held.items():
        jax.config.update(name, value)
    point_xla_cache(held_dir)


def wait_for(path: Path, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return path.exists()


def locked_child(tmp_path: Path, arm: str) -> tuple[subprocess.Popen[str], Path, Path]:
    folder = tmp_path / f"signals-{arm}"
    folder.mkdir()
    script = tmp_path / "locked_child.py"
    script.write_text(LOCKED_CHILD)
    child = subprocess.Popen([sys.executable, str(script), str(tmp_path / f"root-{arm}"), arm, str(folder)],
                             env=child_environment(), stderr=subprocess.PIPE, text=True)
    assert wait_for(folder / "directory", 240), "the child did not reach its cache directory"
    return child, folder, Path((folder / "directory").read_text())


def test_a_write_under_the_users_cache_waits_for_jaxs_lock_on_the_directory(tmp_path):
    """The gate: held from outside, the directory's lock stops the child's first read until it is let go."""
    child, folder, directory = locked_child(tmp_path, "mime")
    with filelock.FileLock(directory / ".lockfile"):
        (folder / "go").touch()
        assert wait_for(folder / "reached", 60)
        time.sleep(3)  # JAX waits up to 10 s for the lock; a compile of this program takes well under one second
        assert not (folder / "done").exists(), "the child compiled and wrote while the lock was held"
    assert wait_for(folder / "done", 120), "the child did not finish once the lock was released"
    assert child.wait(timeout=60) == 0, child.stderr.read() if child.stderr else ""
    assert entries_under(directory), "the child wrote no entry"


def test_control_with_the_limit_reverted_the_child_writes_while_the_lock_is_held(tmp_path):
    """The gate above can fail, on every run: with JAX's limit unset, as before this change, the same child does not look at
    the lock at all and finishes while the test holds it."""
    child, folder, directory = locked_child(tmp_path, "base")
    with filelock.FileLock(directory / ".lockfile"):
        (folder / "go").touch()
        assert wait_for(folder / "done", 120), "with the limit unset the child still waited for the lock"
    assert child.wait(timeout=60) == 0, child.stderr.read() if child.stderr else ""


def test_the_users_cache_is_given_to_jax_as_its_size_limit_and_the_setting_comes_back(jax_cache_unset, tmp_path):
    before = jax.config.jax_compilation_cache_max_size
    with user_xla_cache(root=tmp_path / "root", say=lambda _: None) as held:
        assert held is not None
        assert jax.config.jax_compilation_cache_max_size == DIRECTORY_LIMIT_BYTES == 64 * 1024**2
    assert jax.config.jax_compilation_cache_max_size == before
    with user_xla_cache(root=tmp_path / "small", cap=1000, say=lambda _: None):
        assert jax.config.jax_compilation_cache_max_size == 1000, "never more than the cap"
    assert USER_CACHE_CAP_BYTES == 2 * 1024**3


def test_two_commands_writing_one_program_into_the_users_cache_leave_no_damaged_entry(tmp_path):
    """The statistical check (see the module's text for its measured rates): it can pass where the race is not reached, and
    its control is the lock test above."""
    root, gate = tmp_path / "root", tmp_path / "gate"
    gate.mkdir()
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    begin = str(time.time() + 8)
    children = [subprocess.Popen([sys.executable, str(script), str(root), begin, str(PROGRAMS), str(DEPTH), who, str(gate)],
                                 env=child_environment(), stderr=subprocess.PIPE, text=True) for who in "ab"]
    for child in children:
        _, error = child.communicate(timeout=600)
        assert child.returncode == 0, error
    kept = entries_under(root)
    print(f"entries {len(kept)} of {PROGRAMS + 2}, damaged {damaged_entries(root)}")
    assert damaged_entries(root) == 0
    assert len(kept) >= PROGRAMS, "an entry went missing"


def test_pruning_with_a_directory_to_keep_counts_and_deletes_only_the_others(tmp_path):
    current = tmp_path / "jaxlib-1" / "cpu" / "now"
    other = tmp_path / "jaxlib-0" / "cpu" / "then"
    current.mkdir(parents=True)
    other.mkdir(parents=True)
    kept = [current / "a-cache", current / "b-cache"]
    for index, path in enumerate(kept):
        path.write_bytes(b"x" * 900)
        os.utime(path, (1_000_000 + index, 1_000_000 + index))
    old, new = other / "c-cache", other / "d-cache"
    for index, path in enumerate((old, new)):
        path.write_bytes(b"x" * 400)
        os.utime(path, (2_000_000 + index, 2_000_000 + index))
    assert prune_to_cap(tmp_path, 500, keep=current) == 400
    assert not old.exists() and new.exists() and all(path.exists() for path in kept), \
        "the kept directory's entries are JAX's to evict, however old"
    assert prune_to_cap(tmp_path, 500) > 0 and not any(path.exists() for path in kept), \
        "without keep every directory is pruned together, as before"


def test_a_command_prunes_other_directories_to_the_cap_and_leaves_its_own_to_jax(jax_cache_unset, tmp_path):
    other = tmp_path / "root" / "jaxlib-0" / "cpu" / "elsewhere"
    other.mkdir(parents=True)
    for index in range(3):
        path = other / f"k{index}-cache"
        path.write_bytes(b"x" * 400)
        os.utime(path, (1_000_000 + index, 1_000_000 + index))
    # JAX's directory takes the limit out of the cap, so the other directories are held to what is left: 700 bytes
    with user_xla_cache(root=tmp_path / "root", cap=DIRECTORY_LIMIT_BYTES + 700, say=lambda _: None) as held:
        assert held is not None and not held.is_relative_to(other)
        survivors = sorted(path.name for path in other.iterdir())
    assert survivors == ["k2-cache"], "the oldest files of the other directories go until they fit what the cap leaves"


def test_two_commands_starting_at_once_do_not_share_the_file_that_proves_the_directory_writable(
        jax_cache_unset, monkeypatch, tmp_path):
    """Both wrote and deleted one probe file, so the second delete failed and the command compiled without its cache
    (found by the race above on a fresh directory). The probe a process writes carries its id: with two ids it is two
    files, and with the id left out (the control: one id for both) it is one."""
    written: list[str] = []
    original = Path.write_bytes

    def record(self, data):
        if self.name.startswith(".mime-write-probe"):
            written.append(self.name)
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", record)
    for pid in (101, 102, 101):
        monkeypatch.setattr(_xla_cache.os, "getpid", lambda pid=pid: pid)
        with user_xla_cache(root=tmp_path / "root", say=lambda _: None):
            pass
    assert len(set(written)) == 2 and written[0] == written[2] != written[1]


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs a file this user cannot write")
def test_a_lock_file_the_user_cannot_write_is_the_cache_cannot_be_written_note_not_a_warning_per_program(
        jax_cache_unset, tmp_path):
    """JAX opens ``.lockfile`` for every read and write and warns on each program when it cannot; the probe opens it too, so
    the note MIME already gives for a cache it cannot write is what the user sees, once, and the block runs without the cache.
    The control: the same directory with a lock file the user can write gives no note."""
    messages: list[str] = []
    with user_xla_cache(root=tmp_path / "root", say=messages.append) as held:
        assert held is not None
        lock = held / ".lockfile"
    assert messages == [] and lock.exists()
    lock.chmod(0o400)
    try:
        with user_xla_cache(root=tmp_path / "root", say=messages.append) as chosen:
            assert chosen is None and jax.config.jax_compilation_cache_dir is None
    finally:
        lock.chmod(0o600)
    assert len(messages) == 1 and "cannot be written" in messages[0] and "--no-xla-cache" in messages[0]
