"""The dtype audit: the wide operations in a float32 run's step, with JAX's 64-bit types enabled.

A float32 run holds float32 and int32 values. Each equation of a traced step (and of every program it holds: a ``jit``,
a ``cond`` branch, a loop's body) whose output is float64, int64, uint64 or complex128 is **wide**; a wide equation with
no wide input is an **origin**, where the widening starts (a promotion, a new constant), named by the first MIME or
MADDENING line of its traceback. A state field whose dtype a step changes is **drifted**: the node's step returned it
wider than it came in (``MimeNode.update`` narrows it back; the audit reads the step's own output, before that).

Report-only until float64 islands exist (precision's P2): nothing here fails a run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import numpy as np

WIDE = frozenset({"float64", "int64", "uint64", "complex128"})
#: The frames an equation's traceback keeps while the audit traces: JAX's default (10) stops inside ``jax.numpy``.
TRACEBACK_FRAMES = 60


@dataclass(frozen=True)
class Audit:
    """One traced step's wide equations: ``wide`` by ``(primitive, dtypes)``, ``origins`` by ``(primitive, dtypes,
    line)``, of ``total`` equations; ``drifted`` the state fields its step returned wider, as ``node.field: in->out``."""

    name: str
    total: int
    wide: Mapping[tuple[str, str], int] = field(default_factory=dict[tuple[str, str], int])
    origins: Mapping[tuple[str, str, str], int] = field(default_factory=dict[tuple[str, str, str], int])
    drifted: tuple[str, ...] = ()

    @property
    def wide_count(self) -> int:
        return sum(self.wide.values())

    def by_dtype(self) -> dict[str, int]:
        found: Counter[str] = Counter()
        for (_, dtypes), count in self.wide.items():
            for dtype in dtypes.split(","):
                found[dtype] += count
        return dict(sorted(found.items()))

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "equations": self.total, "wide": self.wide_count, "by_dtype": self.by_dtype(),
                "origins": [{"primitive": p, "dtypes": d, "line": line, "count": n}
                            for (p, d, line), n in sorted(self.origins.items(), key=lambda kv: (-kv[1], kv[0]))],
                "drifted": list(self.drifted)}

    def text(self) -> str:
        lines = [f"{self.name}: {self.wide_count} wide of {self.total} equations {self.by_dtype()}; "
                 f"{len(self.drifted)} state fields drifted"]
        lines += [f"  {n:4d} {p} {d} {line}" for (p, d, line), n in sorted(self.origins.items(),
                                                                         key=lambda kv: (-kv[1], kv[0]))]
        lines += [f"  drifted: {drift}" for drift in self.drifted]
        return "\n".join(lines)


def _line(eqn: Any) -> str:
    traceback = eqn.source_info.traceback
    if traceback is None:
        return "?"
    for frame in traceback.frames:
        name = str(frame.file_name)
        if ("/mime/" in name or "/maddening/" in name) and "/jax/" not in name:
            short = name.split("site-packages/")[-1].split("/src/")[-1]
            return f"{short}:{frame.line_num}"
    return "?"


def _equations(jaxpr: Any, held: str = "?") -> Iterator[tuple[Any, str]]:
    """Every equation of ``jaxpr`` and of the programs it holds, each with its line: its own, or where its own
    traceback reaches no MIME line (a ``jit`` inside ``jax.numpy`` is traced apart), the line of the equation holding
    it."""
    for eqn in jaxpr.eqns:
        own = _line(eqn)
        line = own if own != "?" else held
        yield eqn, line
        for param in cast(dict[str, Any], eqn.params).values():
            subs = cast(tuple[Any, ...], tuple(cast(Any, param)) if isinstance(param, list | tuple) else (param,))
            for sub in subs:
                inner: Any = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _equations(inner, line)


def _dtype(var: Any) -> str:
    return str(getattr(getattr(var, "aval", None), "dtype", ""))


def audit_jaxpr(name: str, closed: Any, drifted: tuple[str, ...] = ()) -> Audit:
    """The wide equations and origins of a traced program ``closed`` (``jax.make_jaxpr``'s)."""
    from jax.extend.core import Literal

    total = 0
    wide: Counter[tuple[str, str]] = Counter()
    origins: Counter[tuple[str, str, str]] = Counter()
    made: set[Any] = set()  # the variables a wide equation wrote
    for eqn, line in _equations(closed.jaxpr):
        total += 1
        out = sorted({_dtype(v) for v in eqn.outvars} & WIDE)
        if not out:
            continue
        key = (str(eqn.primitive.name), ",".join(out))
        wide[key] += 1
        made.update(eqn.outvars)
        if not any(not isinstance(v, Literal) and v in made for v in eqn.invars):
            held = [v for v in eqn.invars if not isinstance(v, Literal) and _dtype(v) in WIDE]
            origins[(*key, line + (" (a wide argument or constant)" if held else ""))] += 1
    return Audit(name, total, dict(wide), dict(origins), drifted)


def drifted_fields(state: Mapping[str, Mapping[str, Any]], ended: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """The fields ``node.field: in->out`` whose dtype differs between a step's state and its output."""
    found: list[str] = []
    for node, fields in ended.items():
        for key, value in fields.items():
            given = state.get(node, {}).get(key)
            if given is not None and hasattr(value, "dtype") and hasattr(given, "dtype") and value.dtype != given.dtype:
                found.append(f"{node}.{key}: {given.dtype}->{value.dtype}")
    return tuple(sorted(found))


def audit_graph(name: str, graph: Any, fed: Mapping[str, Mapping[str, Any]] | None = None) -> Audit:
    """The audit of a built graph's step as MIME's chunked run executes it (``chunk_step``), from its initial state
    (``reset_state``, made in each node's precision), fed ``fed`` (a zero for each external input where not given), with
    the whole parameter tree as an argument. The drift is read from the nodes' own steps, before ``MimeNode.update``
    narrows them back. Leaves the graph reset."""
    from mime.build._maddening import chunk_step, external_inputs, fed_inputs, full_state, graph_params, inputs
    from mime.core._node import steps_as_traced

    if fed is None:
        zeros: dict[str, dict[str, Any]] = {}
        for node, input_name in external_inputs(graph):
            spec = cast(Any, inputs(graph.get_node(node))[input_name])
            zeros.setdefault(node, {})[input_name] = np.zeros(tuple(spec.shape), dtype=np.dtype(spec.dtype or "float32"))
        fed = zeros
    step = cast(Callable[..., Any], chunk_step(graph))
    graph.reset_state()
    state = {node: dict(fields) for node, fields in full_state(graph).items()}
    arguments = (state, fed_inputs(graph, fed), graph_params(graph))
    config = cast(Any, jax.config)
    limit = config.jax_traceback_in_locations_limit
    config.update("jax_traceback_in_locations_limit", TRACEBACK_FRAMES)  # deep enough to reach MIME's line
    try:
        with steps_as_traced():
            closed = cast(Any, jax).make_jaxpr(step)(*arguments)
            ended = cast(Any, jax).eval_shape(step, *arguments)
    finally:
        config.update("jax_traceback_in_locations_limit", limit)
    graph.reset_state()
    return audit_jaxpr(name, closed, drifted_fields({n: f for n, f in state.items() if n != "_meta"},
                                                     {n: f for n, f in ended.items() if n != "_meta"}))


__all__ = ["WIDE", "Audit", "audit_graph", "audit_jaxpr", "drifted_fields"]
