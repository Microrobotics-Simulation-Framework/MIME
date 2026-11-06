"""A folder held as it is: every file by its size, modification time and sha256, and every folder by its modification
time, which a file added, removed or renamed in it changes. ``untouched`` fails a block that wrote anything below the
folder, the same bytes written again among it (a write is a write)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

type FolderState = dict[str, tuple[int] | tuple[int, int, str]]


def state_of(folder: Path) -> FolderState:
    held: FolderState = {".": (folder.stat().st_mtime_ns,)}
    for path in sorted(folder.rglob("*")):
        seen = path.stat()
        name = path.relative_to(folder).as_posix()
        held[name] = ((seen.st_size, seen.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) if path.is_file()
                      else (seen.st_mtime_ns,))
    return held


@contextmanager
def untouched(folder: Path) -> Iterator[None]:
    """Raises ``AssertionError`` where anything below ``folder`` was written while the block ran, naming each path."""
    before = state_of(folder)
    yield
    after = state_of(folder)
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    assert not changed, f"the folder {folder.name} was written: {changed}"
