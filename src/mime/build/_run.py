"""Running a built graph: the build's one call into MADDENING after construction."""

from __future__ import annotations

import numpy as np
from maddening import GraphManager

from mime.build._maddening import NodeFields, run_scan, run_scan_with_history

type History = dict[str, dict[str, np.ndarray]]


def _to_numpy(state: NodeFields) -> History:
    return {node: {name: np.asarray(value) for name, value in fields.items()} for node, fields in state.items()}


def run_steps(graph: GraphManager, steps: int, *, history: bool) -> tuple[History, History | None]:
    """``steps`` base steps from where the graph stands, as one compiled scan; the final state by node and field,
    and with ``history`` every field stacked over the steps."""
    if history:
        final, stacked = run_scan_with_history(graph, steps)
        return _to_numpy(final), _to_numpy(stacked)
    return _to_numpy(run_scan(graph, steps)), None
