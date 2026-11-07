"""One case's reference outputs as a save takes them from the run's own graph, against those of a graph built afresh
(what a save took before saved form 8, and the program ``verify`` runs for a lock that names none), compared array by
array. Run in a fresh process by ``test_save_reference.py``, so the build is the process's first, in the precision the
environment gives (``JAX_ENABLE_X64``).

The run before the save is chunked by one step and keeps its steps' state, as the tube demo's ``run_to`` with
``stage_every`` is: the one-step history program is compiled, and the save takes the reference with it. A case with a
controller called between chunks runs in its control periods (a hook cannot set them), and takes the one-step program
where its writes' latency split the periods into single steps; otherwise the history scan, on the run's graph.

    python reference_programs.py <case> <steps>

prints one JSON line: the case, the precision, the path the save took (``one-step``, ``history`` or ``fresh``), the
arrays compared, and, each with its largest difference and the first step it differs at, the arrays of the save's
reference that differ from: a fresh graph's history scan (``moved``: the save before form 8); a fresh graph run by the
same program (``verify_moved``: what ``verify`` reruns, which must be none); and the run's own first steps
(``own_moved``: the reference is those steps, run again from the initial state, which must be none); and those in which
the history scan on the run's graph differs from a fresh graph's (``history_moved``: what a run not chunked by one step
saves, which must be none). One fresh graph serves the rerun and, reset to its initial state, the history scan (a reset
graph holds a fresh one's state, bit for bit). ``table_row`` is also called in a pytest worker, for the float64 rows."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from mime.build import construct  # noqa: E402
from mime.build._maddening import state_put_back, step_reads  # noqa: E402
from mime.experiment import Experiment  # noqa: E402
from mime.units import s  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def experiment_of(name: str) -> Experiment:
    """A case of ``tests/saved/cases.py`` by name, or an example: ``sphere_settling`` (PR #104's MRS sphere) and
    ``sphere_settling_tube`` (the tube demo), each as its ``main`` builds it."""
    if name in ("sphere_settling", "sphere_settling_tube"):
        sys.path.insert(0, str(ROOT / "examples"))
        module = importlib.import_module(name)
        return getattr(module, name)()
    from cases import CASES

    return next(case.make for case in CASES if case.name == name)()


def table_row(name: str, steps: int) -> dict[str, Any]:
    built = experiment_of(name).build()
    dt = float(built.report.dt)  # type: ignore[arg-type]
    period = built._runs.chunk or 1  # noqa: SLF001
    whole = -(-steps // period) * period
    if built._runs.chunk is None:  # noqa: SLF001
        ran = built.run(s(whole * dt), history=True, hook=lambda chunk: None, chunk=s(dt))
    else:
        ran = built.run(s(whole * dt), history=True)
    assert ran.history is not None
    own = {node: {field: value[:steps] for field, value in fields.items()} for node, fields in ran.history.items()}
    reused = step_reads(built.graph).unchanged_since(built._built_reads)  # noqa: SLF001
    taken, program = built._reference(steps)  # noqa: SLF001
    fresh = construct(built._plans, built._law)[0]
    rerun = built._history_on(fresh, steps, program)  # noqa: SLF001
    fresh.reset_state()
    before = built._history_on(fresh, steps, None)  # noqa: SLF001
    with state_put_back(built.graph):  # the history scan on the run's graph: what a run not chunked by one step saves
        built.graph.reset_state()
        history = built._history_on(built.graph, steps, None)  # noqa: SLF001
    return {"case": name, "x64": bool(jax.config.jax_enable_x64), "steps": steps,
            "path": program or ("history" if reused else "fresh"), "compared": sum(map(len, before.values())),
            "moved": moved(taken, before), "verify_moved": moved(taken, rerun),
            "history_moved": moved(history, before), "own_moved": moved(taken, own),
            "usable_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()}


def moved(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, list[float | int]]:
    """Each array of ``b`` that ``a`` does not hold bit for bit: its largest difference and the first step it differs at."""
    found: dict[str, list[float | int]] = {}
    for node, fields in b.items():
        for field, value in fields.items():
            x, y = np.asarray(a[node][field]), np.asarray(value)
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=True):
                delta = np.abs(x.astype(np.float64) - y.astype(np.float64)) if x.shape == y.shape else np.full(1, np.inf)
                rows = delta.reshape(delta.shape[0], -1) if delta.ndim > 0 else delta.reshape(1, 1)
                found[f"{node}.{field}"] = [float(np.max(delta)), int(np.argmax(np.any(rows > 0, axis=1)))]
    return found


if __name__ == "__main__":
    print(json.dumps(table_row(sys.argv[1], int(sys.argv[2]))), flush=True)
