"""The shards' table's ``split``, read with the standard library only: ``tests/_sharding.py`` (which imports pytest) and
``scripts/stress_select.py`` (which runs on a bare runner Python) read it the same way, through this."""

from __future__ import annotations

import json
from pathlib import Path


class TableError(ValueError):
    """A table that cannot be read, or whose ``split`` is not ``{test file: {test: whole seconds}}``."""


def read_split(path: Path) -> dict[str, dict[str, int]]:
    """``{test file: {test: seconds}}`` for the files the table deals by test (its optional ``split``): each test is named
    by its node id without the file."""
    try:
        split = json.loads(path.read_text(encoding="utf-8")).get("split", {})
    except (OSError, ValueError, AttributeError) as error:
        raise TableError(f"{path}: cannot read the shards' table ({error!r})") from None
    ok = isinstance(split, dict) and all(
        isinstance(file, str) and isinstance(tests, dict) and tests
        and all(isinstance(part, str) and type(sec) is int and sec > 0 for part, sec in tests.items())
        for file, tests in split.items())
    if not ok:
        raise TableError(f"{path}: 'split' must be a {{test file: {{test: whole seconds}}}} table, no file empty")
    return split
