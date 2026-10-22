"""Test-only switches for MIME's own tests of the build. Not public API: nothing outside ``tests/`` imports this.

``declaration_order()`` makes phase 2 add nodes and edges in the order the script declared them, skipping the
canonical order, so a test can show that order-invariance fails without it. ``after_construction(hook)`` calls
``hook(graph, plans)`` after phase 2 has constructed the graph and before the post-build checks, so a test can plant
the defect each check exists to catch.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

_skip_canonical_order = False
_hook: Callable[[Any, Any], None] | None = None


@contextmanager
def declaration_order() -> Generator[None]:
    global _skip_canonical_order
    held, _skip_canonical_order = _skip_canonical_order, True
    try:
        yield
    finally:
        _skip_canonical_order = held


@contextmanager
def after_construction(hook: Callable[[Any, Any], None]) -> Generator[None]:
    global _hook
    held, _hook = _hook, hook
    try:
        yield
    finally:
        _hook = held


def skipping_canonical_order() -> bool:
    return _skip_canonical_order


def run_hook(graph: Any, plans: Any) -> None:
    if _hook is not None:
        _hook(graph, plans)
