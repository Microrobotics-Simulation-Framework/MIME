# A copy of MADDENING's tests/compliance/test_prune_jax_cache.py at its tag v0.4.2 (commit
# 8a6c9fa8066646ab57b55052a300e6badd85c658), Apache-2.0, beside the copy of the script it tests. Changes: this
# header; the file sits in tests/, so REPO_ROOT is one level up, not two; the codecs test makes its zstd frame with
# Python's compression.zstd where there is one (3.14), as JAX and the script then do; with no codec at all it checks the
# script's refusal of a zstd frame instead of skipping, so that no test of the suite is skipped in CI.
"""``scripts/prune_jax_cache.py`` keeps what JAX can read back and deletes the rest.

CI runs it on a push run's compilation cache before saving it (see the
"Drop unreadable compilation-cache entries" step in ``ci.yml``): JAX
writes entries in place, and a truncated one, once saved, raises in every
pull request that restores it.  The entries here are built with the
installed JAX's own ``combine_executable_and_time`` and
``compress_executable`` -- the functions its cache writes with -- so if JAX
changes its entry format this test fails instead of the prune step
quietly deleting every entry, or none.
"""

import importlib.util
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prune_jax_cache.py"


@pytest.fixture(scope="module")
def prune():
    spec = importlib.util.spec_from_file_location("_prune_jax_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jax_entry(payload: bytes) -> bytes:
    from jax._src import compilation_cache as cc

    return cc.compress_executable(cc.combine_executable_and_time(payload, 1234))


def test_an_entry_jax_wrote_is_kept_and_a_truncated_one_is_deleted(prune, tmp_path, capsys):
    entry = _jax_entry(b"serialized executable " * 200)
    cache = tmp_path / "jax-cache"
    cache.mkdir()
    (cache / "jit_f-aaaa-cache").write_bytes(entry)
    (cache / "jit_g-bbbb-cache").write_bytes(entry[: len(entry) // 2])      # killed mid-write
    (cache / "jit_h-cccc-cache").write_bytes(b"")                           # killed before any byte
    (cache / "jit_i-dddd-cache").write_bytes(b"not compressed at all")
    (cache / "jit_f-aaaa-atime").write_bytes(b"\x01\x02")                   # not an entry
    (cache / ".lockfile").write_bytes(b"")                                   # not an entry
    assert prune.main([str(cache)]) == 0
    out = capsys.readouterr().out
    assert sorted(p.name for p in cache.iterdir()) == [
        ".lockfile", "jit_f-aaaa-atime", "jit_f-aaaa-cache"]
    for name in ("jit_g-bbbb-cache", "jit_h-cccc-cache", "jit_i-dddd-cache"):
        assert f"::warning title=Compilation cache::deleted unreadable entry {name}" in out
    assert "1 entries readable, 3 deleted" in out


def test_both_codecs_jax_may_write_with_are_read_whole(prune):
    # JAX compresses with zstandard when it is installed and zlib otherwise.
    body = (1234).to_bytes(4, "big") + b"executable" * 100
    assert prune.unreadable(zlib.compress(body)) is None
    assert prune.unreadable(zlib.compress(body)[:-3]) is not None
    # A whole stream holding no executable after the 4-byte compile time.
    assert prune.unreadable(zlib.compress(b"\x00\x00\x04\xd2")) is not None
    # Which branch runs is the interpreter's: no test here skips, since CI's lane check counts a skipped test against
    # the lane (scripts/check_shards.py). With a codec (compression.zstd on 3.14, which JAX uses first; zstandard
    # where it is installed) a frame is made and read whole. Without one (3.12 and 3.13 in CI, which do not install
    # zstandard) JAX writes zlib entries, and a zstd frame can only be refused by name, which is what the script does
    # there.
    if prune.stdlib_zstd is not None:
        frame = prune.stdlib_zstd.compress(body)
    elif prune.zstandard is not None:
        frame = prune.zstandard.ZstdCompressor().compress(body)
    else:
        assert "neither compression.zstd nor zstandard" in prune.unreadable(prune.ZSTD_MAGIC + body)
        return
    assert prune.unreadable(frame) is None
    assert "zstd" in prune.unreadable(frame[:-3])


def test_with_no_zstd_codec_a_zstd_entry_is_refused_by_name_on_every_python(prune, monkeypatch):
    monkeypatch.setattr(prune, "stdlib_zstd", None)
    monkeypatch.setattr(prune, "zstandard", None)
    assert "neither compression.zstd nor zstandard" in prune.unreadable(prune.ZSTD_MAGIC + b"\x00" * 16)


def test_a_missing_cache_directory_is_nothing_to_check(prune, tmp_path, capsys):
    assert prune.main([str(tmp_path / "never-created")]) == 0
    assert "nothing to check" in capsys.readouterr().out
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    assert prune.main([str(not_a_dir)]) == 2
