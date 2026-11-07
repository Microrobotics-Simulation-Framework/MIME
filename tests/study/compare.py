"""Members against their own runs, in eps of each field's scale: the comparison gate 1 and gate 8 hold the sweep to, with
the tolerance committed in ``predictions.json`` before any gate case ran."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mime.results import RunResult

PREDICTIONS = json.loads((Path(__file__).parent / "predictions.json").read_text(encoding="utf-8"))
TOLERANCE_EPS: Mapping[str, float] = PREDICTIONS["gate_1"]["tolerance_eps"]


def miss_eps(a: Any, b: Any, over: Any = None) -> float:
    """The largest |a - b| in eps of ``a``'s dtype, over the largest magnitude of ``over`` (the field over the run; ``a``
    when not given); exact (0 or inf) where that scale is zero or the arrays are not floating."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.dtype.kind not in "fc":
        return 0.0 if np.array_equal(a, b) else float("inf")
    held = np.asarray(a if over is None else over)
    scale = float(np.max(np.abs(held))) if held.size else 0.0
    delta = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    if scale == 0.0:
        return 0.0 if delta == 0.0 else float("inf")
    return delta / (np.finfo(a.dtype).eps * scale)


def misses(reference: RunResult, member: RunResult, series: Sequence[str] = ()) -> dict[str, float]:
    """Every field of the final state and every series in ``series``, the member against its reference, in eps of the
    reference field's largest magnitude over the run (``reference`` keeps its history: amendment 1)."""
    history = reference.history
    assert history is not None, "the reference keeps every step, for each field's scale over the run"
    found = {f"final {node}.{name}": miss_eps(value, member.final[node][name], history[node][name])
             for node, fields in reference.final.items() for name, value in fields.items()}
    for path in series:
        observation, _, output = path.rpartition(".")
        found[f"series {path}"] = miss_eps(reference[observation][output].raw, member[observation][output].raw)
    return found


def worst(found: Mapping[str, float]) -> tuple[str, float]:
    key = max(found, key=lambda k: found[k])
    return key, found[key]
