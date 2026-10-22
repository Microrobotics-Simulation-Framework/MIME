"""Running a built graph: the build's one call into MADDENING after construction."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(state: dict) -> dict[str, dict[str, np.ndarray]]:
    return {node: {name: np.asarray(value) for name, value in fields.items()} for node, fields in state.items()}


def run_steps(graph: Any, steps: int, *, history: bool) -> tuple[dict, dict | None]:
    """``steps`` base steps from where the graph stands; the final state by node and field, and with ``history`` every
    field stacked over the steps."""
    if history:
        final, stacked = graph.run_scan_with_history(steps)
        return _to_numpy(final), _to_numpy(stacked)
    return _to_numpy(graph.run_scan(steps)), None
