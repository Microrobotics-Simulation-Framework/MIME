"""Test-only switches for MIME's own tests of the stage writer. Not public API: nothing outside ``tests/`` imports
this.

``without_default_prim()`` makes the writer leave ``defaultPrim`` unset, so a test can show that its read-back check
and usd-core's validators both fail on such a stage. ``capsule_height_with_caps()`` writes a native capsule's
``height`` as its whole length, caps included, and ``native_lengths_in_metres()`` writes a native primitive's lengths
in metres whatever the stage's unit, so a test can show that the round trip and USD's own definition both catch each.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

_omit_default_prim = False
_height_with_caps = False
_lengths_in_metres = False


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


@contextmanager
def capsule_height_with_caps() -> Generator[None]:
    global _height_with_caps
    held, _height_with_caps = _height_with_caps, True
    try:
        yield
    finally:
        _height_with_caps = held


def writing_capsule_height_with_caps() -> bool:
    return _height_with_caps


@contextmanager
def native_lengths_in_metres() -> Generator[None]:
    global _lengths_in_metres
    held, _lengths_in_metres = _lengths_in_metres, True
    try:
        yield
    finally:
        _lengths_in_metres = held


def native_scale(scale: float) -> float:
    """The metres per unit a native primitive's lengths are divided by: the stage's, or 1 when planted."""
    return 1.0 if _lengths_in_metres else scale
