"""Test-only switches for MIME's own tests of the stage writer. Not public API: nothing outside ``tests/`` imports
this.

``without_default_prim()`` makes the writer leave ``defaultPrim`` unset, so a test can show that its read-back check
and usd-core's validators both fail on such a stage.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

_omit_default_prim = False


@contextmanager
def without_default_prim() -> Generator[None]:
    global _omit_default_prim
    held, _omit_default_prim = _omit_default_prim, True
    try:
        yield
    finally:
        _omit_default_prim = held


def omitting_default_prim() -> bool:
    return _omit_default_prim
