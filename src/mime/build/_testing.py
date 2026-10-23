"""Test-only switches for MIME's own tests of the build. Not public API: nothing outside ``tests/`` imports this.

``declaration_order()`` makes phase 2 add nodes and edges in the order the script declared them, skipping the
canonical order, so a test can show that order-invariance fails without it. Declared in that order, a feedback loop is
closed wherever add order puts its back edge, often a delivered load read a step late, which the build refuses as
unplanned; so the switch lifts that refusal too, unless ``refuse_unplanned=True`` asks to see it.

``after_construction(hook)`` calls ``hook(graph, plans)`` after phase 2 has constructed the graph and before the
post-build checks, so a test can plant the defect each check exists to catch.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

_skip_canonical_order = False
_lift_unplanned = False
_hook: Callable[[Any, Any], None] | None = None


@contextmanager
def declaration_order(*, refuse_unplanned: bool = False) -> Generator[None]:
    global _skip_canonical_order, _lift_unplanned
    held = (_skip_canonical_order, _lift_unplanned)
    _skip_canonical_order, _lift_unplanned = True, not refuse_unplanned
    try:
        yield
    finally:
        _skip_canonical_order, _lift_unplanned = held


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


def refusing_unplanned_loops() -> bool:
    return not _lift_unplanned


def run_hook(graph: Any, plans: Any) -> None:
    if _hook is not None:
        _hook(graph, plans)
