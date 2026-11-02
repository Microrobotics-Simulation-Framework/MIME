"""XLA's persistent compilation cache for ``mime run`` and ``mime serve``: on by default in the user's cache directory,
off for ``mime verify`` and under ``--no-xla-cache``, capped in size, never a reason for a run to stop, and never a
change to a number. The in-process tests turn JAX's own cache setting off first (CI sets it for its lanes), and the
bit-for-bit test runs the command in fresh processes, which is the only way a warm cache is read at all."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import platformdirs
import pytest

from mime.cli import main as mime_main
from mime.env import USER_CACHE_CAP_BYTES, clear_user_xla_cache, prune_to_cap, user_xla_cache, user_xla_cache_root
from mime.env import _xla_cache
from mime.env._xla_cache import point_xla_cache
from mime.saved import read_folder
from mime.testing.toys import coupled_toy, full_toy

REAL_ROOT = user_xla_cache_root
SETTINGS = ("jax_persistent_cache_min_compile_time_secs", "jax_persistent_cache_min_entry_size_bytes")


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    folder = tmp_path_factory.mktemp("toy")
    full_toy().save(folder)
    jax.config.update("jax_enable_x64", held)
    return folder


@pytest.fixture(scope="module")
def changed(tmp_path_factory) -> Path:
    """The coupled toy: another graph than the full toy, so other programs."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    folder = tmp_path_factory.mktemp("coupled")
    coupled_toy().save(folder)
    jax.config.update("jax_enable_x64", held)
    return folder


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A home that is not the user's, with JAX's own cache setting off, as both were found, and the real path of the
    user's cache (the suite's fixture moves it into each test's directory)."""
    monkeypatch.setattr(_xla_cache, "user_xla_cache_root", REAL_ROOT)
    folder = tmp_path / "home"
    folder.mkdir()
    for name in ("HOME", "USERPROFILE", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(folder))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    held_dir = jax.config.jax_compilation_cache_dir
    held_x64 = jax.config.jax_enable_x64
    held = {name: getattr(jax.config, name) for name in SETTINGS}
    point_xla_cache(None)
    yield folder
    jax.config.update("jax_enable_x64", held_x64)
    for name, value in held.items():
        jax.config.update(name, value)
    point_xla_cache(held_dir)


def entries(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()] if root.exists() else []


def duration_of(folder: Path, steps: int) -> str:
    return repr(steps * read_folder(folder).lock["resolved"]["run"]["dt"])


def test_the_default_cache_is_under_the_users_cache_directory_and_is_restored_afterwards(saved, home, tmp_path):
    root = user_xla_cache_root()
    assert root == Path(platformdirs.user_cache_dir("mime")) / "xla" and root.is_relative_to(home)
    assert mime_main(["run", str(saved), "--duration", duration_of(saved, 2), "--out", str(tmp_path / "run")]) == 0
    assert entries(root) and all(path.is_relative_to(root) for path in entries(root))
    assert jax.config.jax_compilation_cache_dir is None


def test_the_cache_is_written_with_every_program_stored_and_the_settings_come_back(home, tmp_path):
    held = {name: getattr(jax.config, name) for name in SETTINGS}
    with user_xla_cache(root=tmp_path / "xla") as chosen:
        assert chosen is not None and jax.config.jax_compilation_cache_dir == str(chosen)
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == -1
    assert jax.config.jax_compilation_cache_dir is None
    assert {name: getattr(jax.config, name) for name in SETTINGS} == held


def test_no_xla_cache_and_mime_verify_leave_the_cache_unused(saved, home, tmp_path, capsys):
    run = tmp_path / "run"
    assert mime_main(["run", str(saved), "--duration", duration_of(saved, 2), "--out", str(run), "--no-xla-cache"]) == 0
    assert not user_xla_cache_root().exists()
    assert mime_main(["verify", str(run / "experiment")]) == 0
    assert not user_xla_cache_root().exists() and "moved" not in capsys.readouterr().out
    assert mime_main(["run", str(saved), "--duration", duration_of(saved, 2), "--out", str(tmp_path / "again")]) == 0
    assert entries(user_xla_cache_root())  # the control: the same run without the flag does fill it


def test_verify_says_in_its_help_that_it_keeps_no_cache(capsys):
    with pytest.raises(SystemExit):
        mime_main(["verify", "--help"])
    assert "compilation cache" in capsys.readouterr().out.replace("\n", " ")


def test_a_cache_already_set_for_jax_is_left_as_it_is(home, tmp_path):
    point_xla_cache(str(tmp_path / "theirs"))
    try:
        with user_xla_cache(root=tmp_path / "xla") as chosen:
            assert chosen is None and jax.config.jax_compilation_cache_dir == str(tmp_path / "theirs")
        assert not (tmp_path / "xla").exists() and jax.config.jax_compilation_cache_dir == str(tmp_path / "theirs")
    finally:
        point_xla_cache(None)


def write(path: Path, size: int, age: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (1_000_000 + age, 1_000_000 + age))
    return path


def test_pruning_deletes_the_oldest_entries_first_down_to_the_cap(tmp_path):
    oldest = write(tmp_path / "a" / "k1-cache", 400, 1)
    middle = write(tmp_path / "b" / "k2-cache", 400, 2)
    newest = write(tmp_path / "b" / "k3-cache", 400, 3)
    assert prune_to_cap(tmp_path, 800) == 400
    assert not oldest.exists() and middle.exists() and newest.exists()
    assert prune_to_cap(tmp_path, 800) == 0  # within the cap: nothing goes
    assert prune_to_cap(tmp_path, 0) == 800 and not entries(tmp_path)


def test_the_cap_is_two_gigabytes_and_is_applied_at_the_start(home, tmp_path):
    assert USER_CACHE_CAP_BYTES == 2 * 1024**3
    old = write(tmp_path / "xla" / "jaxlib-0" / "cpu" / "x" / "old-cache", 600, 1)
    new = write(tmp_path / "xla" / "jaxlib-0" / "cpu" / "x" / "new-cache", 600, 2)
    with user_xla_cache(root=tmp_path / "xla", cap=700):
        assert not old.exists() and new.exists()


def test_a_cache_that_cannot_be_written_does_not_stop_the_block_and_is_said_once(home, tmp_path):
    if sys.platform == "win32" or os.geteuid() == 0:
        pytest.skip("needs a directory this user cannot write: a read-only mode does not stop Windows or root")
    shut = tmp_path / "shut"
    shut.mkdir()
    shut.chmod(0o500)
    said: list[str] = []
    try:
        with user_xla_cache(root=shut / "xla", say=said.append) as chosen:
            assert chosen is None and jax.config.jax_compilation_cache_dir is None
            assert float(jax.numpy.sum(jax.numpy.ones(3))) == 3.0
    finally:
        shut.chmod(0o700)
    assert len(said) == 1 and "cannot be written" in said[0] and "--no-xla-cache" in said[0]


def test_a_command_with_an_unwritable_home_runs_and_says_so_once(saved, home, tmp_path, capsys):
    if sys.platform == "win32" or os.geteuid() == 0:
        pytest.skip("needs a directory this user cannot write: a read-only mode does not stop Windows or root")
    home.chmod(0o500)
    try:
        assert mime_main(["run", str(saved), "--duration", duration_of(saved, 2), "--out", str(tmp_path / "run")]) == 0
    finally:
        home.chmod(0o700)
    assert capsys.readouterr().err.count("cannot be written") == 1 and (tmp_path / "run" / "outputs.npz").is_file()


def test_cache_clear_deletes_the_cache_and_says_how_much(home, capsys):
    assert mime_main(["cache", "clear"]) == 0
    assert "no XLA compilation cache to remove" in capsys.readouterr().out
    write(user_xla_cache_root() / "jaxlib-0" / "cpu" / "x" / "k-cache", 300, 1)
    assert mime_main(["cache", "clear"]) == 0
    assert "removed 300 bytes" in capsys.readouterr().out and not user_xla_cache_root().exists()
    assert clear_user_xla_cache(user_xla_cache_root()) == 0


def mime_run(folder: Path, out: Path, steps: int, home: Path, *flags: str) -> subprocess.Popen[bytes]:
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("JAX_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE", "XDG_CACHE", "XLA_FLAGS"))}
    environment.update(HOME=str(home), USERPROFILE=str(home), LOCALAPPDATA=str(home), JAX_PLATFORMS="cpu")
    return subprocess.Popen([sys.executable, "-m", "mime.cli", "run", str(folder), "--duration",
                             duration_of(folder, steps), "--out", str(out), "--history", *flags], env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def digest(out: Path) -> str:
    held = hashlib.sha256()
    with np.load(out / "outputs.npz", allow_pickle=False) as outputs:
        for key in sorted(outputs.files):
            held.update(key.encode())
            held.update(np.ascontiguousarray(outputs[key]).tobytes())
    return held.hexdigest()


def finished(process: subprocess.Popen[bytes]) -> None:
    text, _ = process.communicate(timeout=600)
    assert process.returncode == 0, text.decode(errors="replace")


def test_a_cold_and_a_warm_cache_give_the_bits_of_no_cache_and_a_changed_program_is_a_miss_not_a_stale_hit(
        saved, changed, tmp_path):
    """The control is the same command on another experiment, whose programs differ: it must add entries (a miss) and
    equal its own uncached run (not the first run's result served from an entry, were the key too coarse). (A longer
    run of the same experiment is no such control: its scan length is not part of the compiled program, and it adds
    no entry.)"""
    first = planted = 4
    home = tmp_path / "home"
    home.mkdir()
    bare_first = mime_run(saved, tmp_path / "off", first, tmp_path / "unused-1", "--no-xla-cache")
    bare_planted = mime_run(changed, tmp_path / "planted-off", planted, tmp_path / "unused-2", "--no-xla-cache")
    finished(mime_run(saved, tmp_path / "cold", first, home))
    after_cold = len(entries(home))
    finished(mime_run(saved, tmp_path / "warm", first, home))
    after_warm = len(entries(home))
    finished(mime_run(changed, tmp_path / "planted", planted, home))
    after_planted = len(entries(home))
    finished(bare_first)
    finished(bare_planted)
    assert not entries(tmp_path / "unused-1") and not entries(tmp_path / "unused-2")
    assert digest(tmp_path / "off") == digest(tmp_path / "cold") == digest(tmp_path / "warm")
    assert after_cold > 0 and after_warm == after_cold  # the warm run found every program it needed
    assert after_planted > after_warm  # the control: the changed program was compiled, not found
    assert digest(tmp_path / "planted") == digest(tmp_path / "planted-off") != digest(tmp_path / "off")
