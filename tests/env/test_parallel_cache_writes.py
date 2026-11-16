"""Two commands writing one program into the user's XLA cache at once leave no entry that JAX cannot read.

JAX's ``LRUCache.put`` (eviction off, as MIME keeps it) checks that the entry is absent and then writes it in place, with no
lock, so two processes that compile one program together both write it, and when their frames differ in length the file is
the shorter frame followed by the longer one's tail: a complete zstd frame and more bytes, which JAX refuses on Python 3.14
(``Unknown frame descriptor``) and never rewrites, so every later run that reads it warns. ``user_xla_cache`` now deletes, at
the end of its block, the entries written during it that JAX could not read back (``remove_damaged_entries``).

The race: two children, started at one moment through MIME's own ``user_xla_cache``, compile the same programs into one
fresh directory, held in step by a file barrier before each program so that their writes overlap. A program is a chain of
small matrix products: its executable's compressed length differs from one compile to the next (a tiny program's
rarely does, and equal lengths leave no damage). The damage is read by a decompressor of the test's own, not by the code
under test.

The control is the same two children with the repair turned off (what the code did before it): it must leave damage, and
it is tried up to ``CONTROL_TRIES`` times because a trial left 3 to 13 damaged entries of 152 in 12 of 13 measured (one left
none, so a single trial would fail the control about one run in thirteen; six in a row would at about one in 10^6). The fixed arm is one trial and must leave none: the process that ends last looks
at every entry after both are done writing.

On a Python that compresses with zlib (before 3.14, without ``zstandard``) JAX reads a stream followed by other bytes
without complaint, so the damage costs a run nothing there; the repair still deletes such a file (it is not what either
writer wrote), and the test counts it with a zlib object that must reach its end with nothing left over. Both lanes run
the race, since the default lane allows no skipped test."""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import jax
import pytest

from mime.env import _xla_cache
from mime.env._xla_cache import point_xla_cache, remove_damaged_entries, user_xla_cache

import zlib

try:
    from compression import zstd  # Python 3.14+, which JAX compresses with
except ImportError:
    zstd = None

PROGRAMS = 150
DEPTH = 8
CONTROL_TRIES = 6

CHILD = textwrap.dedent('''
    import sys, time
    from pathlib import Path
    root, start_at, programs, depth, me, gate, repair = sys.argv[1:]
    other = "b" if me == "a" else "a"
    import jax, jax.numpy as jnp
    from mime.env import _xla_cache
    if repair == "off":
        _xla_cache.remove_damaged_entries = lambda *_: 0

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


def race(tmp_path: Path, name: str, repair: str) -> Path:
    root, gate = tmp_path / name, tmp_path / f"{name}-gate"
    gate.mkdir()
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("JAX_", "PYTEST_", "XDG_CACHE", "XLA_FLAGS", "MIME_TEST"))}
    environment["JAX_PLATFORMS"] = "cpu"
    begin = str(time.time() + 8)
    children = [subprocess.Popen([sys.executable, str(script), str(root), begin, str(PROGRAMS), str(DEPTH), who, str(gate),
                                  repair], env=environment, stderr=subprocess.PIPE, text=True) for who in "ab"]
    for child in children:
        _, error = child.communicate(timeout=600)
        assert child.returncode == 0, error
    return root


def test_control_without_the_repair_two_commands_writing_one_program_leave_a_damaged_entry(tmp_path):
    """The harness can fail: it left damage in 12 of the 13 trials measured, 3 to 13 entries of 152."""
    found = []
    for attempt in range(CONTROL_TRIES):
        found.append(damaged_entries(race(tmp_path, f"base{attempt}", "off")))
        if found[-1]:
            break
    print(f"damaged entries by trial, repair off: {found} of {len(entries_under(tmp_path / f'base{len(found) - 1}'))}")
    assert found[-1] > 0, f"no damage in {CONTROL_TRIES} trials ({found}): the race is not being reached"


def test_two_commands_writing_one_program_into_the_users_cache_leave_no_damaged_entry(tmp_path):
    root = race(tmp_path, "fixed", "on")
    kept = entries_under(root)
    print(f"entries kept {len(kept)} of {PROGRAMS + 2}")
    assert damaged_entries(root) == 0
    assert len(kept) >= PROGRAMS // 2, "the repair deleted what it should have left"


@pytest.fixture
def jax_cache_unset():
    """JAX's own cache setting off for the test (CI sets it for its lanes, and ``user_xla_cache`` leaves a set one alone),
    and put back."""
    names = ("jax_persistent_cache_min_compile_time_secs", "jax_persistent_cache_min_entry_size_bytes")
    held_dir, held = jax.config.jax_compilation_cache_dir, {name: getattr(jax.config, name) for name in names}
    point_xla_cache(None)
    yield
    for name, value in held.items():
        jax.config.update(name, value)
    point_xla_cache(held_dir)


def whole_and_damaged(root: Path) -> dict[str, bytes]:
    """Compressed entries as JAX writes them, by the decompressor of this interpreter, and the damage of each kind."""
    payload = bytes(range(256)) * 40
    frame = zstd.compress(payload) if zstd else zlib.compress(payload)
    return {"whole": frame, "cut": frame[: len(frame) // 2], "empty": b"", "tail": frame + b"\x07\x91"}


def test_remove_damaged_entries_deletes_what_jax_cannot_read_and_only_what_was_written_since(tmp_path):
    entries = whole_and_damaged(tmp_path)
    since = time.time()
    for name, content in entries.items():
        (tmp_path / f"new_{name}-cache").write_bytes(content)
        (tmp_path / f"old_{name}-cache").write_bytes(content)
        os.utime(tmp_path / f"old_{name}-cache", (since - 1000, since - 1000))
    (tmp_path / "keep.txt").write_bytes(b"not an entry")
    unreadable = {"cut", "empty", "tail"}
    removed = remove_damaged_entries(tmp_path, since - 10)
    left = {path.name for path in tmp_path.iterdir()}
    assert removed == len(unreadable)
    assert {f"new_{name}-cache" for name in entries} - left == {f"new_{name}-cache" for name in unreadable}
    assert {f"old_{name}-cache" for name in entries} <= left, "an entry from before the block is not this block's to judge"
    assert "keep.txt" in left


def test_a_block_that_leaves_a_damaged_entry_removes_it_and_says_so_once(jax_cache_unset, tmp_path):
    messages: list[str] = []
    entries = whole_and_damaged(tmp_path)
    with user_xla_cache(root=tmp_path / "root", say=messages.append) as held:
        assert held is not None
        (held / "a-cache").write_bytes(entries["cut"])
        (held / "b-cache").write_bytes(entries["empty"])
        (held / "c-cache").write_bytes(entries["whole"])
    assert sorted(path.name for path in held.iterdir() if path.name.endswith("-cache")) == ["c-cache"]
    assert len(messages) == 1 and "removed 2 damaged entries" in messages[0]


def test_a_block_with_nothing_damaged_says_nothing(jax_cache_unset, tmp_path):
    messages: list[str] = []
    with user_xla_cache(root=tmp_path / "root", say=messages.append) as held:
        assert held is not None
        (held / "c-cache").write_bytes(whole_and_damaged(tmp_path)["whole"])
    assert messages == [] and (held / "c-cache").exists()


def test_the_repair_is_what_user_xla_cache_calls(jax_cache_unset, monkeypatch, tmp_path):
    """The control for the two tests above: where the block's end does not call it, a damaged entry stays."""
    monkeypatch.setattr(_xla_cache, "remove_damaged_entries", lambda *_: 0)
    with user_xla_cache(root=tmp_path / "root", say=lambda _: None) as held:
        assert held is not None
        (held / "a-cache").write_bytes(b"")
    assert (held / "a-cache").exists()


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
