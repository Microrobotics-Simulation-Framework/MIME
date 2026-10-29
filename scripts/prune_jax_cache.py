#!/usr/bin/env python
# A copy of MADDENING's scripts/prune_jax_cache.py at its tag v0.4.2 (commit
# 8a6c9fa8066646ab57b55052a300e6badd85c658), Apache-2.0, for the compilation cache CI saves per shard. The docstring
# is MADDENING's; in MIME pytest does not turn warnings into errors, so an unreadable entry would cost a recompile
# and a warning on every run that restores it rather than a test error. Changes: this header; a zstd entry is read
# with Python's own compression.zstd where it has one (3.14), which JAX prefers to zstandard there, so that entries
# written on Python 3.14 without zstandard are kept.
"""Delete persistent XLA compilation-cache entries that JAX could not read back.

CI saves each shard's compilation cache after a push run, and every later
pull request restores it.  JAX writes an entry in place -- no temporary
file and rename -- and never rewrites an entry whose file exists, and the
test lanes' subprocesses inherit ``JAX_COMPILATION_CACHE_DIR``: a child
killed mid-write (a timeout, a crash) leaves a truncated entry under its
final name, for good.  Reading one makes JAX warn "Error reading
persistent compilation cache entry", and the suite's
``filterwarnings = ["error"]`` turns that into a test error on every run
that restores the cache.  So ``ci.yml`` runs this before saving.

An entry is JAX's ``compress_executable`` of a 4-byte big-endian compile
time followed by the serialized executable, compressed with zstandard when
it is installed and zlib otherwise.  An entry is kept only if it
decompresses completely and holds more than the 4-byte header; anything
else is deleted and named.  (The executable itself cannot be checked
without the backend that deserialises it; truncation, the failure mode
here, is caught by decompression: a zstd frame records its content size,
and a zlib stream its end.)

Only regular files in the directory itself are checked; ``*-atime`` files
and ``.lockfile`` (written only when a size cap is set, which CI does not
set) and subdirectories are left alone.  Deleting an entry costs one
recompile, so an unreadable file is always deleted.

Usage::

    python scripts/prune_jax_cache.py <cache dir>

Exit 0 after a scan, whatever it deleted (a GitHub warning annotation
names each deleted entry); 2 if the path exists and is not a directory.
A missing directory is nothing to check (exit 0).
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

try:
    from compression import zstd as stdlib_zstd  # Python 3.14+, which JAX uses first
except ImportError:
    stdlib_zstd = None
try:
    import zstandard
except ImportError:  # JAX then writes zlib entries, and so could not read a zstd one
    zstandard = None

#: The first four bytes of every zstd frame.
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
#: ``jax._src.compilation_cache._TIME_BYTES``: the compile-time header.
TIME_BYTES = 4
#: Files JAX's cache writes that are not entries.
NOT_ENTRIES = (".lockfile",)
NOT_ENTRY_SUFFIXES = ("-atime",)


def unreadable(data: bytes) -> str | None:
    """Why JAX could not read this entry back, or ``None`` if it could."""
    if not data:
        return "empty file"
    if data.startswith(ZSTD_MAGIC):
        if stdlib_zstd is not None:
            try:
                payload = stdlib_zstd.decompress(data)
            except stdlib_zstd.ZstdError as exc:
                return f"zstd: {exc}"
        elif zstandard is None:
            return "zstd entry, but neither compression.zstd nor zstandard is installed to read it"
        else:
            try:
                payload = zstandard.ZstdDecompressor().decompress(data)
            except zstandard.ZstdError as exc:
                return f"zstd: {exc}"
    else:
        try:
            payload = zlib.decompress(data)
        except zlib.error as exc:
            return f"zlib: {exc}"
    if len(payload) <= TIME_BYTES:
        return f"{len(payload)} bytes after decompression: no executable after the header"
    return None


def _escape(text: str) -> str:
    """Escape for a GitHub workflow-command message."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def prune(cache_dir: Path) -> tuple[int, list[tuple[str, str]]]:
    """Delete the unreadable entries; return ``(kept, [(name, why), ...])``."""
    kept, removed = 0, []
    for path in sorted(cache_dir.iterdir()):
        if not path.is_file() or path.name in NOT_ENTRIES or path.name.endswith(NOT_ENTRY_SUFFIXES):
            continue
        why = unreadable(path.read_bytes())
        if why is None:
            kept += 1
            continue
        path.unlink()
        removed.append((path.name, why))
    return kept, removed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cache_dir", type=Path)
    args = p.parse_args(argv)
    if not args.cache_dir.exists():
        print(f"prune_jax_cache: {args.cache_dir} does not exist; nothing to check")
        return 0
    if not args.cache_dir.is_dir():
        print(f"::error title=Compilation cache::{args.cache_dir} is not a directory")
        return 2
    kept, removed = prune(args.cache_dir)
    for name, why in removed:
        print("::warning title=Compilation cache::" + _escape(f"deleted unreadable entry {name} ({why})"))
    print(f"prune_jax_cache: {kept} entries readable, {len(removed)} deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
