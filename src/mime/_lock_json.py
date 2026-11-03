"""The one reader of a saved folder's lock (``mime.lock``) as JSON: every module that reads a lock reads it here.

A lock is integrity data, the pins and hashes a load holds a folder to. Python's ``json`` keeps the last of a repeated
key without a word, and a reader that keeps the first (some tools do) would read another lock from the same bytes, so a
mapping that holds one key twice is refused here; ``save`` never writes one. The refusal is a ``ValueError``
(:class:`RepeatedKey`), as ``json``'s own is for text that is not JSON, so each caller refuses both by its own code: a
load as ``load.lock``, ``mime assets fetch`` as ``asset.lock``, a run's or a Modal script's environment as
``env.lock``."""

from __future__ import annotations

import json
from pathlib import Path


class RepeatedKey(ValueError):
    """A mapping of the lock holds one key twice."""


def _once(pairs: list[tuple[str, object]]) -> dict[str, object]:
    held: dict[str, object] = {}
    for key, value in pairs:
        if key in held:
            raise RepeatedKey(f"the key {key!r} appears twice in one mapping; a lock holds each key once")
        held[key] = value
    return held


def lock_data(text: str | bytes) -> object:
    """The lock ``text`` holds, as ``json`` reads it; raises ``ValueError`` on text that is not JSON and on a mapping
    that holds one key twice."""
    return json.loads(text, object_pairs_hook=_once)


def read_lock(path: Path) -> object:
    """The lock in the file ``path``; raises ``OSError`` where it cannot be read, and ``ValueError`` as
    :func:`lock_data` does."""
    return lock_data(path.read_text(encoding="utf-8"))


__all__ = ["RepeatedKey", "lock_data", "read_lock"]
