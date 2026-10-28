"""A controller inside the compiled step (its first mode): its node and delay line, built from its law, and the checks
that the law's step can run there and that the node runs where the latency rule needs it.

**The trace check.** The step is traced once with the shapes and dtypes of what it reads. A step that cannot be traced
(Python on a traced value: ``int()``, ``if``, NumPy), that calls the host (a callback, ``jax.debug.print``), or whose
state or writes come back in other fields, shapes or dtypes than it starts with, is refused for this mode: it runs as
a callback, between compiled chunks, where it may be any Python.

**The order.** A tick at step t reads the state as it stands at t, and its writes act from step t + latency
(``LATENCY_RULE``). Inside the compiled step that holds when the controller's node runs before every node it reads and
every node its writes reach: each read edge then reads the previous step (the state at the start of this one), and
each write edge this step. Its nodes are added first in the canonical order for that; a node it reads that its
writes do not reach runs before it whatever the order, since the schedule follows the edges, and is refused."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, cast

from maddening.core.node import SimulationNode

from mime.build._builders import COMMAND, STATE, EdgeDecl
from mime.build._control import DELAY, ControllerPlan, Mode
from mime.build._lags import EdgeLag, Lag, scheduled
from mime.build._maddening import initial_fields, inputs, make_jaxpr_with_shapes
from mime.build._probes import host_callbacks
from mime.build._refusals import MIME_DEFECT, Refusal, RefusalClass
from mime.core import jnp
from mime.nodes.control import ControllerNode, DelayNode, Step


def inlined(step: Any) -> Step:
    """The step as it is traced into the compiled step: a step its law compiled with ``jax.jit`` (for the callback's
    speed) through the function it compiles, which XLA inlines either way, so that a nested ``jit`` hides none of its
    switches from the switch scan; any other step as it is."""
    wrapped = inspect.unwrap(step) if hasattr(step, "lower") else step
    return cast(Step, wrapped)


def trace_problems(step: Step, state: Mapping[str, Any], reads: Mapping[str, Any],
                   writes: Mapping[str, Any]) -> list[str]:
    """Why ``step`` cannot run inside the compiled step, traced from ``state`` and ``reads``: it cannot be traced,
    it calls the host, or its state or writes do not keep the fields, shapes and dtypes it starts with (``writes``,
    the writes it holds, in the dtypes of their commands, whose shapes its writes keep)."""
    try:
        jaxpr, shapes = make_jaxpr_with_shapes(step, dict(state), dict(reads))
    except Exception as error:  # noqa: BLE001 - whatever stops the trace is the reason, named
        first = str(error).strip().splitlines()[0] if str(error).strip() else ""
        return [f"its step cannot be traced: {type(error).__name__}: {first}"]
    found: list[str] = []
    calls = host_callbacks(jaxpr)
    if calls:
        found.append(f"its step calls the host ({', '.join(sorted(calls))})")
    after, written = cast(tuple[Any, Any], shapes)
    found += _kept(after, state, "state", "its state comes back as", keep_dtype=True)
    found += _kept(written, writes, "write", "its writes come back as", keep_dtype=False)
    return found


def _kept(given: Any, start: Mapping[str, Any], what: str, said: str, *, keep_dtype: bool) -> list[str]:
    if not isinstance(given, Mapping) or set(cast(Mapping[str, Any], given)) != set(start):
        keys = sorted(cast(Mapping[str, Any], given)) if isinstance(given, Mapping) else given
        return [f"{said} {keys!r}, and it holds {sorted(start)!r}"]
    found: list[str] = []
    for key, held in sorted(cast(Mapping[str, Any], given).items()):
        want = jnp.asarray(start[key])
        if tuple(held.shape) != tuple(want.shape) or (keep_dtype and held.dtype != want.dtype):
            found.append(f"{said} {key!r} {held.dtype}{list(held.shape)}, and the {what} it starts from is "
                         f"{want.dtype}{list(want.shape)}")
    return found


def controller_nodes(plan: ControllerPlan, law: Any, dt: float,
                     built: Mapping[str, SimulationNode]) -> tuple[list[SimulationNode], list[EdgeDecl], list[Refusal]]:
    """The controller's nodes and edges inside the compiled step, from its law (``law``, a ``ControlLaw``) and the
    nodes built for the rest of the graph, whose state it reads and whose commands it writes; or the refusal of its
    step for this mode."""
    owner = plan.owner
    reads = {read.label: jnp.asarray(initial_fields(built[read.node])[read.field]) for read in plan.reads}
    dtypes = {write.label: inputs(built[write.node])[write.field].dtype for write in plan.writes}
    held = {label: jnp.asarray(value, dtype=dtypes[label]) for label, value in law.held.items()}
    start = {key: jnp.asarray(value) for key, value in law.state.items()}
    step = inlined(law.step)
    problems = trace_problems(step, start, {label: jnp.zeros_like(value) for label, value in reads.items()}, held)
    if problems:
        label = plan.registered.cls.label()
        return [], [], [Refusal(
            "controller.untraceable", RefusalClass.USER,
            f"the controller {label} cannot run inside the compiled step (mode {Mode.GRAPH.value}): "
            + "; ".join(problems),
            f"run it as a callback, mode={Mode.CALLBACK.word!r}, where its step may be any Python and no derivative "
            f"passes through it; or write its step in jax.numpy, with a state and writes of fixed shapes")]
    node = ControllerNode(owner, dt * plan.period, step=step, controller=plan.registered.cls.label(),
                          reads={read.label: (read.units, tuple(reads[read.label].shape)) for read in plan.reads},
                          initial_state=start, initial_writes=held)
    edges = [EdgeDecl(owner, (read.node, read.field), (owner, f"read_{read.label}"), False, read.units, seq, STATE)
             for seq, read in enumerate(plan.reads)]
    nodes: list[SimulationNode] = [node]
    sources = {write.label: (owner, f"write_{write.label}") for write in plan.writes}
    if plan.latency:
        delay = f"{owner}.{DELAY}"
        nodes.append(DelayNode(delay, dt, latency=plan.latency, units={w.label: w.units for w in plan.writes},
                               initial_writes=held))
        edges += [EdgeDecl(owner, sources[write.label], (delay, write.label), False, write.units, len(edges) + seq,
                           COMMAND) for seq, write in enumerate(plan.writes)]
        sources = {write.label: (delay, write.label) for write in plan.writes}
    edges += [EdgeDecl(owner, sources[write.label], (write.node, write.field), False, write.units, len(edges) + seq,
                       COMMAND) for seq, write in enumerate(plan.writes)]
    return nodes, edges, []


def in_graph_order(controller: Sequence[str], physics: Sequence[str], edges: Sequence[EdgeDecl],
                   groups: Sequence[Sequence[str]]) -> list[str]:
    """The order the nodes are added in with a controller inside the compiled step: its nodes first, so that it reads
    the state at the start of each step and its writes act within it; then the rest (``physics``, in the canonical
    order, with their ``edges`` and coupling ``groups``) in the order MADDENING runs them without the controller's
    edges, as a callback's graph has none. Each of their edges then reads the step it reads with the controller called
    between chunks; the canonical order alone would put a commanded segment after the fields it carries, which would
    read its pose a step late."""
    return [*controller, *scheduled(physics, edges, groups)]


def order_refusals(plan: ControllerPlan | None, lags: Sequence[EdgeLag],
                   without: Sequence[EdgeLag] = ()) -> list[Refusal]:
    """Where the controller's node does not run before every node it reads and every node its writes reach: a read
    of a node its writes do not reach, which the schedule runs before it; a read or write edge MIME built the wrong
    way round; or an edge of the rest of the graph reading another step than it reads ``without`` the controller's
    node (as a callback's graph): the last two defects of MIME's order."""
    if plan is None or plan.mode is not Mode.GRAPH:
        return []
    before = {(edge.source, edge.target): edge.lag for edge in without}
    owner, label = plan.owner, plan.registered.cls.label()
    mine = {owner, f"{owner}.{DELAY}"}
    reached = _reached(owner, lags)
    refusals: list[Refusal] = []
    for edge in lags:
        if edge.target[0] == owner and edge.lag is not Lag.PREVIOUS_STEP and edge.source[0] not in reached:
            refusals.append(Refusal(
                "controller.unreached-read", RefusalClass.USER,
                f"the controller {label} reads {edge.source[0]}.{edge.source[1]}, which nothing it writes reaches: "
                f"inside the compiled step that node runs before the controller in every step, so the controller "
                f"would read its state at the end of the step rather than at its start, against the latency rule",
                f"read only what the controller's writes reach in mode {Mode.GRAPH.value}, or run it as a callback, "
                f"mode={Mode.CALLBACK.word!r}, which reads any state at the start of the period"))
        elif (edge.target[0] == owner and edge.lag is not Lag.PREVIOUS_STEP) or \
                (edge.source[0] in mine and edge.lag is not Lag.THIS_STEP):
            refusals.append(Refusal(
                "post-build.controller-order", RefusalClass.INTERNAL,
                f"the edge {edge.text} of the controller {label} reads {edge.lag.value}: inside the compiled step the "
                f"controller's node runs before every node it reads (each read edge reads the previous step) and every "
                f"node its writes reach (each write edge this step), or its latency is not the one declared",
                MIME_DEFECT))
        elif before.get((edge.source, edge.target), edge.lag) is not edge.lag:
            refusals.append(Refusal(
                "post-build.controller-order", RefusalClass.INTERNAL,
                f"the edge {edge.text} reads {edge.lag.value} with the controller {label} inside the compiled step, "
                f"and {before[(edge.source, edge.target)].value} with it called between chunks: swapping the "
                f"controller's mode changes no physics", MIME_DEFECT))
    return refusals


def _reached(owner: str, lags: Sequence[EdgeLag]) -> set[str]:
    """The nodes the controller's writes reach along the graph's edges, its own included."""
    successors: dict[str, set[str]] = {}
    for edge in lags:
        successors.setdefault(edge.source[0], set()).add(edge.target[0])
    reached, pending = {owner}, [owner]
    while pending:
        for nxt in successors.get(pending.pop(), set()) - reached:
            reached.add(nxt)
            pending.append(nxt)
    return reached


__all__ = ["controller_nodes", "in_graph_order", "inlined", "order_refusals", "trace_problems"]

