"""A study's derivative (``Study.gradient``): the objective's derivative in each axis, at each point, through the
study's own path: every point planned and assembled by ``build_point``, grouped by the compiled program it needs, and
each group's members run together, vmapped, by a differentiated chunked run (``mime.build._maddening.differentiated_run``)
beside the batched scan every forward run takes, which it does not touch.

**What is differentiated.** An axis writes leaves of the nodes' parameter trees (``mime.study._override`` lists them
for a run's record). The derivative is taken in the trainable leaves it writes, each a linear function of the axis's SI
value (the identity, or a unit's factor), which a probe point finds: the point with the axis moved by a relative
:data:`PROBE_MOVE`, planned and assembled, and compared leaf by leaf. A non-trainable leaf it writes is allowed only
where the node makes it from the others (``MimeNode.made_again``: an MRS drag's held matrix, whose derivative is the
made value's); any other, an affine or nonlinear map, a moved initial state, or a move that changes the compiled
program is refused with its route.

**Modes.** Forward mode, one tangent a scalar, up to :data:`FORWARD_UP_TO` scalars (``mode="auto"``); reverse mode
beyond, each chunk a ``jax.checkpoint``.

**What the result carries** (``mime.results.GradientResult``): the flags the differentiated run raised, a node's stop
read from its final state (a guard's flag form keeps it finite), the material reads the member's plan extrapolated,
the path's status composed from the entries the derivative passes through (a ``cut`` or ``unverified`` entry refused
unless acknowledged), and each iterative solve's report forward and in the derivative solves, with the warning of a
solve that stopped short and the recheck at a tighter tolerance."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
from jax.flatten_util import ravel_pytree

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._checks import stopped_refusals
from mime.build._control import Mode
from mime.build._maddening import META, FlagTest, differentiated_run, full_state, graph_edges, parameter_specs, \
    initial_fields, live_inputs, state_put_back, unreadable_leaves
from mime.build._refusals import ArgumentRefused
from mime.control._run import ChunkedRun
from mime.gradients import ACKNOWLEDGED_ONLY, GradientStatus, composed, unacknowledged
from mime.core import MimeNode
from mime.nodes.control import ControllerNode, DelayNode
from mime.registry import Role, placed_gradient, rung_gradient
from mime.results._gradient import GradientResult, MemberGradient, RaisedFlag, Recheck, SolveReport
from mime.saved._pins import keeping_pins
from mime.study._axes import STUDY_REASON, Axis, AxisKind, base_value, outside_range, value_text
from mime.study._point import Point, build_point, precision
from mime.study._study import stacked_state, stacked_tree
from mime.units import Quantity, unit_symbol

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment, Experiment
    from mime.study._study import Group, StudyPlan

_jax: Any = jax
_jnp: Any = jax.numpy

#: Up to how many scalars ``mode="auto"`` takes the derivative in forward mode (one tangent each, the primal run with
#: each); reverse mode beyond, its cost about a few runs whatever their number. Provisional: from the gate kit's
#: measurement of the two on the study's consumers (forward mode about 1.5 runs a scalar, reverse 2 to 4 runs).
FORWARD_UP_TO = 2
MODES = ("auto", "forward", "reverse")
#: The relative move of a probe point, which finds the leaves an axis writes: a power of two, so that the move of a
#: value is exact, and small enough to stay inside a declared range and on one side of every value-keyed choice the
#: study's points are on.
PROBE_MOVE = 2.0 ** -20
#: How much a recheck tightens a flagged solve's tolerance.
RECHECK_FACTOR = 1e-2
#: A node's iterative solves report through these: a ``params`` leaf of zeros whose gradient is the transpose solves'
#: reports, and a state field holding the primal reports' running sum (``mime.nodes.flow.fvm``'s design, which every
#: iterative solve in MIME follows); and the leaf a recheck tightens.
REPORT_LEAF, REPORT_SUM, REPORT_STEP, TOLERANCE = "solve_report", "solve_report_sum", "solve_report", "tolerance"
#: How far a written leaf's ratio to its axis's value may differ between the probe and a member for the map from the
#: axis to the leaf to be taken as linear: rounding's, many times over, and far below any affine or nonlinear map's.
LINEAR = 1e-9


@dataclass(frozen=True)
class Written:
    """An axis and what it writes: the trainable leaves (``(node, leaf)``) and the initial state's fields (``(node,
    field)``) the derivative is taken through, each with its factor from the axis's SI value (``factors``, keyed
    :func:`key`) and whether that factor is the ratio of the two (``through_zero``: a linear map, the factor read at
    each member exactly) or the probes' slope (an affine map, a node's start made from the value and a constant); and
    the leaves their nodes make from the trainable ones (``made``)."""

    axis: Axis
    leaves: tuple[tuple[str, str], ...]
    states: tuple[tuple[str, str], ...]
    made: tuple[tuple[str, str], ...]
    factors: Mapping[str, np.ndarray]
    through_zero: Mapping[str, bool]


def key(kind: str, node: str, name: str) -> str:
    """The key of a written leaf (``kind`` ``leaf``) or initial field (``state``) among a :class:`Written`'s."""
    return f"{kind}:{node}.{name}"


def magnitude(value: object) -> np.ndarray | None:
    """An axis value's SI magnitude as a float64 array, or ``None`` for a value with none (a count, a drive, a name)."""
    if isinstance(value, Quantity) and not isinstance(value.value, str):
        held = np.asarray(value.value, dtype=np.float64)
        return held if held.dtype.kind == "f" else None
    if isinstance(value, float):
        return np.asarray(value, dtype=np.float64)
    return None


def _moved(value: object, sign: float) -> object:
    """``value`` moved by a relative :data:`PROBE_MOVE` (a component at zero by that much of the largest component's
    size, or of one SI unit where every component is zero)."""
    held = magnitude(value)
    assert held is not None
    size = float(np.max(np.abs(held))) if held.size else 0.0
    step = np.where(held != 0.0, held * PROBE_MOVE, (size or 1.0) * PROBE_MOVE) * sign
    moved = held + step
    if isinstance(value, Quantity):
        return dataclasses.replace(value, value=float(moved) if moved.ndim == 0 else tuple(float(c) for c in moved))
    return float(moved)


def _variant(axis: Axis, built: BuiltExperiment) -> str | None:
    """What a variant axis is, which no derivative is taken in: the run's step, the precision, a rung, a numerical
    setting (its profile's role)."""
    if axis.kind in (AxisKind.DT, AxisKind.PRECISION):
        return "the run's step" if axis.kind is AxisKind.DT else "the precision"
    if axis.kind is AxisKind.SETTING and axis.name.split(".")[-1] == "rung":
        return "the rung"
    site = f"{axis.owner}.{axis.name}" if axis.kind is AxisKind.COMMAND else axis.path
    role = built.plans.profile.roles.get(site)
    return "a numerical setting" if role is not None and role.role is Role.NUMERICAL else None


def _value_of(axis: Axis, point: Point, settings: Any, declarations: Sequence[Any]) -> object | None:
    return point.values[axis] if axis in point.values else base_value(axis, settings, declarations)


def written(axis: Axis, point: Point, settings: Any, declarations: Sequence[Any], built: BuiltExperiment,
            reference: Callable[[str], Point | None], reasons: Mapping[str, str] | None, protocol: Any) -> Written:
    """What ``axis`` writes at ``point``, found from two probe points (``point`` with the axis moved by one and two
    relative :data:`PROBE_MOVE`, toward the inside of its declared range): each leaf and initial field that moves, with
    its factor (:func:`factor_of`); or the refusal of an axis no derivative can be taken in through them."""
    value = _value_of(axis, point, settings, declarations)
    if value is None or magnitude(value) is None:
        held = "no value the experiment declares" if value is None else f"{value_text(value)}, which is no number"
        raise BuildRefused([Refusal(
            "gradient.not-a-number", RefusalClass.USER,
            f"the derivative in {axis.path} needs a value of numbers, and it has {held}",
            f"declare its value, `exp.declare({axis.path!r}, <value>, reason=...)`, or name another axis in "
            f"`study.gradient(of, wrt=[...])`",
            why="the derivative is taken in the value an axis declares, by moving it")])
    sign = 1.0 if not outside_range(axis, _moved(value, 2.0), declarations) else -1.0
    probes: list[Point] = []
    for times in (1.0, 2.0):
        moved = _moved(value, sign * times)
        with precision(point.declared.x64):
            probe = build_point(-1, settings, declarations, {**point.values, axis: moved}, STUDY_REASON, reference,
                                reasons, protocol)
        if probe.refused:
            raise BuildRefused([Refusal(
                "gradient.probe-refused", RefusalClass.USER,
                f"the experiment with {axis.path} moved from {value_text(value)} to {value_text(moved)} is refused, so "
                f"no derivative is taken there: " + "; ".join(f"[{r.code}] {r.message}" for r in probe.refusals),
                "take the derivative at a value whose neighbourhood the experiment allows, a point of "
                "`Study(exp).sweep(...)` further from the line that refuses it")])
        if probe.key != point.key:
            raise BuildRefused([Refusal(
                "gradient.value-keyed", RefusalClass.USER,
                f"{axis.path} moved from {value_text(value)} to {value_text(moved)} needs another compiled program: an "
                f"entry makes a choice from its value when it is built (a count, a spacing, a scheme), so the "
                f"objective is not differentiable there in the program's parameters",
                f"declare a range of it, exp.declare_range({axis.path!r}, low, high, reason=...), so that every value "
                f"inside builds one program, and take the derivative inside it",
                why="a derivative through the parameters of one program says nothing of a choice made when it is "
                    "built")])
        probes.append(probe)
    values = [magnitude(each) for each in (value, _moved(value, sign), _moved(value, 2.0 * sign))]
    specs = parameter_specs(built.graph)
    changed = sorted({(node, leaf) for probe in probes for node, leaves in probe.tree.items()
                      for leaf, held in leaves.items() if not _same(held, point.tree[node][leaf])})
    moved_state = sorted({(node, name) for probe in probes for node, fields in probe.state.items()
                          for name, held in fields.items() if not _same(held, point.state[node][name])})
    unread = unreadable_leaves(built.graph)
    baked = [(node, leaf) for node, leaf in changed if (node, leaf) in unread and not leaf.startswith("initial_")]
    if baked:
        raise BuildRefused([Refusal(
            "gradient.leaf-unread", RefusalClass.USER,
            f"{axis.path} writes " + "; ".join(f"params['nodes'][{node!r}][{leaf!r}], which changes nothing in the "
                                               f"compiled step: {unread[(node, leaf)]}" for node, leaf in baked) +
            ": the node consumes the value when it is built, so its derivative there would read zero",
            "name an axis the step reads in `study.gradient(of, wrt=[...])`; a node holds every value it consumes in "
            "a leaf its step reads, or a declared static (report the entry)",
            why="a value derived from a declared leaf is computed in the program, never at build")])
    # A node's start held as a parameter (``initial_*``) that its step does not read reaches the run through the
    # initial state, which is differentiated where it moves.
    changed = [(node, leaf) for node, leaf in changed if (node, leaf) not in unread]
    # A leaf its node gives no spec is trainable, as MADDENING's default spec is.
    trainable = tuple((node, leaf) for node, leaf in changed if _trainable(specs, node, leaf))
    rest = [(node, leaf) for node, leaf in changed if not _trainable(specs, node, leaf)]
    made: list[tuple[str, str]] = []
    for node in sorted({node for node, _ in rest}):
        held_node = built.graph.get_node(node)
        makes: set[str] = set(held_node.made_again(probes[0].tree[node])) if isinstance(held_node, MimeNode) else set()
        made += [(n, leaf) for n, leaf in rest if n == node and leaf in makes]
    held_out = [f"{node}.{leaf}" for node, leaf in rest if (node, leaf) not in made]
    if held_out:
        raise BuildRefused([Refusal(
            "gradient.held-leaf", RefusalClass.USER,
            f"{axis.path} writes {', '.join(held_out)}, which its node holds as a parameter no derivative is taken in "
            f"(trainable=False), made at build from the value: the derivative through it would be dropped",
            "name another axis in `study.gradient(of, wrt=[...])`; the entry is to make the value from its leaves in "
            "the program, as a run-constant (`MimeNode.run_constants` and `made_again`, as the MRS drag makes its "
            "matrix): report the entry",
            why="a value derived from a declared leaf is computed in the program, never at build")])
    if not trainable and not moved_state:
        raise BuildRefused([Refusal(
            "gradient.no-leaf", RefusalClass.USER,
            f"{axis.path} writes no parameter a derivative is taken in and no initial state: its value reaches the "
            f"run only through what the nodes are built with",
            "name an axis a node holds as a trainable parameter in `study.gradient(of, wrt=[...])`")])
    factors: dict[str, np.ndarray] = {}
    through_zero: dict[str, bool] = {}
    for kind, items, trees in (("leaf", trainable, [point.tree, *(p.tree for p in probes)]),
                               ("state", tuple(moved_state), [point.state, *(p.state for p in probes)])):
        for node, name in items:
            fields = [np.asarray(tree[node][name]) for tree in trees]
            shape = np.shape(cast(list[np.ndarray], values)[0])
            if int(np.prod(shape)) != 1 and shape != fields[0].shape:
                raise BuildRefused([Refusal(
                    "gradient.shape", RefusalClass.USER,
                    f"{axis.path} (of shape {shape}) moves {node}.{name}, of shape {fields[0].shape}: a part of a "
                    f"field or a field made of several values, which a derivative here is not read through",
                    "name an axis the node holds whole, of its own shape, in `study.gradient(of, wrt=[...])`",
                    why="the derivative in an axis is read through what it writes element by element")])
            found = factor_of(fields, cast(list[np.ndarray], values))
            if found is None:
                what = "parameter" if kind == "leaf" else "initial state's field"
                raise BuildRefused([Refusal(
                    "gradient.derived-leaf", RefusalClass.USER,
                    f"{axis.path} writes the {what} {node}.{name} by a map that is not affine in its value (a power, "
                    f"a product with other values), made when the node is built: its derivative is not read there",
                    "name the value the node holds, declared, in `study.gradient(of, wrt=[...])`, or report the "
                    "entry",
                    why="a value derived from a declared value is computed in the program, never at build")])
            factors[key(kind, node, name)], through_zero[key(kind, node, name)] = found
    return Written(axis, trainable, tuple(moved_state), tuple(made), factors, through_zero)


def _trainable(specs: Mapping[str, Mapping[str, Any]], node: str, leaf: str) -> bool:
    spec = specs.get(node, {}).get(leaf)
    return True if spec is None else bool(spec.trainable)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


def _broadcast(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | None:
    return np.broadcast_to(value, shape) if value.size == 1 else (value if value.shape == shape else None)


def factor_of(fields: Sequence[np.ndarray], values: Sequence[np.ndarray]) -> tuple[np.ndarray, bool] | None:
    """The factor from an axis's value to what it writes, from three points (the member's and two probes',
    ``fields`` and ``values``, each in that order): the field over the value where the map is linear (through zero),
    exact; the first probe's slope where it is affine (its rounding about ``eps / PROBE_MOVE`` of it); ``None`` where
    the two slopes differ beyond rounding (a map that is not affine), or the field's shape is not the value's (a
    scalar value moves a field of any shape)."""
    f0, f1, f2 = (np.asarray(f, dtype=np.float64) for f in fields)
    held = [_broadcast(np.asarray(v, dtype=np.float64), f0.shape) for v in values]
    if any(v is None for v in held):
        return None
    v0, v1, v2 = cast(list[np.ndarray], held)
    eps = float(np.finfo(np.asarray(fields[0]).dtype if np.asarray(fields[0]).dtype.kind == "f" else np.float64).eps)
    step = v1 - v0
    tiny = np.finfo(np.float64).tiny
    rounding = 8 * eps * np.maximum.reduce([np.abs(f0), np.abs(f1), np.abs(f2)]) / np.maximum(np.abs(step), tiny)
    slope = np.divide(f1 - f0, step, out=np.zeros_like(f0), where=step != 0)
    further = np.divide(f2 - f1, v2 - v1, out=np.zeros_like(f0), where=(v2 - v1) != 0)
    if np.any(np.abs(slope - further) > LINEAR * np.abs(slope) + rounding):
        return None
    ratio = np.divide(f0, v0, out=slope.copy(), where=v0 != 0)
    linear = bool(np.all(np.abs(ratio - slope) <= LINEAR * np.abs(slope) + rounding) and
                  np.all(np.where(v0 == 0, f0 == 0, True)))
    return (ratio if linear else slope), linear


def member_factor(field_: np.ndarray, value: np.ndarray, factor: np.ndarray, through_zero: bool,
                  leader: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    """A member's factor from its value to what the axis writes: through zero, its own ratio, checked against the
    probes' to :data:`LINEAR`; affine, the probes' slope, checked by the member's own field against the leader's
    (``leader``: its field and value) moved by it. ``None`` where the check fails: the map is not the probes' across
    the study's points."""
    held = np.asarray(field_, dtype=np.float64)
    value_ = _broadcast(np.asarray(value, dtype=np.float64), held.shape)
    if value_ is None:
        return None
    scale = LINEAR * np.maximum(np.abs(factor), np.finfo(np.float64).tiny)
    eps = float(np.finfo(np.asarray(field_).dtype if np.asarray(field_).dtype.kind == "f" else np.float64).eps)
    if through_zero:
        ratio = np.divide(held, value_, out=np.asarray(factor, dtype=np.float64).copy(), where=value_ != 0)
        rounding = 4 * eps * np.abs(ratio)
        return ratio if not np.any(np.abs(ratio - factor) > scale + rounding) else None
    field0, value0 = leader
    base = _broadcast(np.asarray(value0, dtype=np.float64), held.shape)
    assert base is not None
    expected = np.asarray(field0, dtype=np.float64) + factor * (value_ - base)
    rounding = 8 * eps * np.maximum(np.abs(held), np.abs(expected))
    spread = np.abs(factor * (value_ - base))
    return np.asarray(factor, dtype=np.float64) if not np.any(np.abs(held - expected) > LINEAR * spread + rounding) \
        else None


def solving_nodes(graph: Any) -> tuple[str, ...]:
    """The nodes that solve iteratively and report it (:data:`REPORT_LEAF` among their leaves, :data:`REPORT_SUM` in
    their state)."""
    found: list[str] = []
    for name in graph.node_names:
        node = graph.get_node(name)
        if isinstance(node, MimeNode) and REPORT_LEAF in node.constants() and REPORT_SUM in initial_fields(node):
            found.append(name)
    return tuple(found)


def objective_nodes(of: Callable[[Any], Any], final: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """The nodes whose final state the objective reads, from its trace: every argument its program uses."""
    flat, tree = _jax.tree_util.tree_flatten_with_path(final)
    paths = [str(getattr(path[0], "key", path[0])) for path, _ in flat]

    def given(*leaves: Any) -> Any:
        return of(_jax.tree_util.tree_unflatten(tree, list(leaves)))

    live = live_inputs(_jax.make_jaxpr(given)(*[leaf for _, leaf in flat]).jaxpr)
    return tuple(sorted({paths[i] for i, read in enumerate(live) if read}))


def path_nodes(graph: Any, sources: Sequence[str], targets: Sequence[str],
               through: Sequence[tuple[str, str]] = ()) -> set[str]:
    """The nodes on a path from a node of ``sources`` to one of ``targets`` along the graph's edges and ``through``
    (a controller called on the host, from each node it reads to each it writes; a step reads its sources' state of
    the step before, so a path may cross steps): reachable from the sources and reaching the targets."""
    edges = [(source, target) for source, _, target, _ in graph_edges(graph)] + list(through)

    def reach(start: Sequence[str], forward: bool) -> set[str]:
        seen = set(start)
        frontier = list(start)
        while frontier:
            here = frontier.pop()
            for a, b in edges:
                there = b if forward and a == here else a if not forward and b == here else None
                if there is not None and there not in seen:
                    seen.add(there)
                    frontier.append(there)
        return seen

    return reach(sources, True) & reach(targets, False)


def path_statuses(built: BuiltExperiment, nodes: set[str], callback: bool) -> dict[str, GradientStatus]:
    """The status of each entry whose nodes lie on the path (by entry id): its attached rung's, the law's for a
    controller inside the compiled step, and the binding's ``cut`` for a controller called on the host, on the path
    wherever it reads a node on it and writes one."""
    plans = built.plans
    found: dict[str, GradientStatus] = {}
    for entry in plans.entries:
        if any(node in nodes for node in entry.node_names()):
            cls = entry.registered.cls
            found[entry.owner] = rung_gradient(cls, getattr(entry.entry, "rung", None)).differentiability
    controller = plans.controller
    if controller is not None:
        cls = controller.registered.cls
        if callback:
            reads = {read.node for read in controller.reads}
            writes = {write.node for write in controller.writes}
            if reads & nodes and writes & nodes:
                found[controller.owner] = placed_gradient(cls, callback=True).differentiability
        elif any(isinstance(built.graph.get_node(node), (ControllerNode, DelayNode)) for node in nodes):
            found[controller.owner] = placed_gradient(cls).differentiability
    return found


@dataclass(frozen=True)
class _Prepared:
    """A group's differentiated program and what its members feed it."""

    function: Callable[..., Any]
    batched: bool
    solving: tuple[str, ...]
    tests: tuple[FlagTest, ...]
    meanings: tuple[tuple[str, str], ...]
    #: The member's objective, ``(theta, theta0, factors, tree, state, taps) -> (value, aux)``, undifferentiated: what
    #: a fit's Fisher information is taken of.
    objective: Callable[..., Any]


def final_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {node: fields for node, fields in state.items() if node != META}


def _moved_by(held: Any, factor: Any, delta: Any) -> Any:
    """``held + factor delta``, in ``held``'s dtype and shape: the member's own value where ``delta`` is zero."""
    value = _jnp.asarray(held)
    return (value + _jnp.asarray(factor) * delta).astype(value.dtype).reshape(_jnp.shape(value))


def _replaced(tree: Mapping[str, Any], state: Mapping[str, Any], theta: Mapping[str, Any],
              theta0: Mapping[str, Any], factors: Mapping[str, Any], writes: Sequence[Written],
              taps: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The tree and the initial state with what each axis writes at ``held + factor (theta - theta0)``: the member's
    own where ``theta`` is its own value, and its derivative in ``theta`` the factor; and each solving node's report
    leaf ``taps``."""
    nodes = {node: dict(leaves) for node, leaves in cast(Mapping[str, Mapping[str, Any]], tree["nodes"]).items()}
    start = {node: dict(fields) for node, fields in state.items()}
    for write in writes:
        delta = _jnp.asarray(theta[write.axis.path]) - _jnp.asarray(theta0[write.axis.path])
        for node, leaf in write.leaves:
            nodes[node][leaf] = _moved_by(nodes[node][leaf], factors[key("leaf", node, leaf)], delta)
        for node, name in write.states:
            start[node][name] = _moved_by(start[node][name], factors[key("state", node, name)], delta)
    for node, tap in taps.items():
        nodes[node][REPORT_LEAF] = tap
    return {**tree, "nodes": nodes}, start


def prepare(built: BuiltExperiment, writes: Sequence[Written], of: Callable[[Any], Any], mode: str, params: Any,
             axes: tuple[tuple[str, str, int | None], ...], batched: bool,
             fed: Sequence[tuple[int, Any]] | None, steps: int) -> _Prepared:
    graph = built.graph
    tests = tuple(built.flag_tests())
    meanings = tuple((f"{flag.node}.{flag.name or flag.field}", flag.meaning) for flag in built.layout.flags)
    solving = solving_nodes(graph)
    kept = tuple((node, REPORT_STEP) for node in solving)
    run = differentiated_run(graph, steps, tests, kept, checkpoint=mode == "reverse", fed=fed)

    def objective(theta: Any, theta0: Any, factors: Any, tree: Any, state: Any, taps: Any) -> tuple[Any, Any]:
        params, start = _replaced(tree, state, theta, theta0, factors, writes, taps)
        final, tally, out = run(start, params)
        value = of(final_state(final))
        return _jnp.asarray(value), (tally, final, out)

    if mode == "reverse":
        def member(theta: Any, theta0: Any, factors: Any, tree: Any, state: Any, taps: Any) -> Any:
            (value, aux), (grad, tapped) = _jax.value_and_grad(objective, argnums=(0, 5), has_aux=True)(
                theta, theta0, factors, tree, state, taps)
            return value, grad, tapped, aux
    else:
        def member(theta: Any, theta0: Any, factors: Any, tree: Any, state: Any, taps: Any) -> Any:
            flat, unravel = cast(Any, ravel_pytree)(theta)

            def along(direction: Any) -> Any:
                def at(point: Any) -> Any:
                    value, (tally, final, out) = objective(unravel(point), theta0, factors, tree, state, taps)
                    sums = {node: final[node][REPORT_SUM] for node in solving}
                    return value, sums, (tally, final, out)

                (value, _, aux), (tangent, tangent_sums, _) = _jax.jvp(at, (flat,), (direction,))
                return value, tangent, tangent_sums, aux

            basis = _jnp.eye(flat.size, dtype=flat.dtype)
            values, tangents, tangent_sums, aux = _jax.vmap(along)(basis)
            first = _jax.tree_util.tree_map(_first, aux)
            backward = {node: _jnp.sum(tangent_sums[node], axis=0) for node in solving}
            return values[0], columns(theta, tangents), backward, first

    if not batched:
        return _Prepared(_jax.jit(member), False, solving, tests, meanings, objective)
    marked = {(node, leaf): axis for node, leaf, axis in axes}
    tree_axes = {"nodes": {node: {leaf: marked.get((node, leaf)) for leaf in leaves}
                           for node, leaves in params["nodes"].items()}, "mappings": {}}
    vmapped = _jax.vmap(member, in_axes=(0, 0, 0, tree_axes, 0, None))
    return _Prepared(_jax.jit(vmapped), True, solving, tests, meanings, objective)


def stacked_members(points: Sequence[Point], members: Sequence[int], writes: Sequence[Written], settings: Any,
             declarations: Sequence[Any], dtype: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The members' own values of each axis (``theta0``, by path, stacked) and each factor of what it writes (by
    :func:`key`, stacked), each member's checked against the probes' (:func:`member_factor`)."""
    theta0: dict[str, list[np.ndarray]] = {}
    factors: dict[str, list[np.ndarray]] = {}
    leader = points[members[0]]
    for index in members:
        point = points[index]
        for write in writes:
            value = _value_of(write.axis, point, settings, declarations)
            held = magnitude(value)
            assert held is not None
            theta0.setdefault(write.axis.path, []).append(held.astype(dtype))
            first = magnitude(_value_of(write.axis, leader, settings, declarations))
            assert first is not None
            for kind, items, own, lead in (("leaf", write.leaves, point.tree, leader.tree),
                                           ("state", write.states, point.state, leader.state)):
                for node, name in items:
                    at = key(kind, node, name)
                    factor = member_factor(own[node][name], held, write.factors[at], write.through_zero[at],
                                           (lead[node][name], first))
                    if factor is None:
                        raise BuildRefused([Refusal(
                            "gradient.derived-leaf", RefusalClass.USER,
                            f"at point {index}, {write.axis.path} = {value_text(value)} writes {node}.{name} by a map "
                            f"that is not the one its probes found: it is not affine across the study's points",
                            "name the value the node holds, declared, in `study.gradient(of, wrt=[...])`, or report "
                            "the entry")])
                    factors.setdefault(at, []).append(factor.astype(np.asarray(own[node][name]).dtype))
    return {path: np.stack(held) for path, held in theta0.items()}, \
        {at: np.stack(held) for at, held in factors.items()}


def _report(solving: Sequence[str], final: Mapping[str, Any], out: Mapping[str, Any], backward: Mapping[str, Any],
            mode: str) -> tuple[SolveReport, ...]:
    return tuple(SolveReport(node, np.asarray(final[node][REPORT_SUM], dtype=np.float64),
                             np.asarray(out[node][REPORT_STEP], dtype=np.float64) if node in out else None,
                             np.asarray(backward[node], dtype=np.float64), mode) for node in solving)


def _warnings(reports: Sequence[SolveReport]) -> tuple[str, ...]:
    found: list[str] = []
    for report in reports:
        if report.forward_short:
            found.append(f"{report.forward_short} of node {report.node}'s primal solves stopped short of their "
                         f"tolerance (at the cap, stagnated or not finite): the run's values there hold to the "
                         f"residual they stopped at, not to the tolerance")
        if report.backward_short:
            which = "transpose" if report.mode == "reverse" else "tangent"
            found.append(f"{report.backward_short} of node {report.node}'s {which} solves stopped short of their "
                         f"tolerance: the derivative holds to the residual they stopped at, not to the tolerance")
    return tuple(found)


@dataclass
class _Member:
    value: float
    gradient: dict[str, np.ndarray]
    tally: tuple[np.ndarray, np.ndarray, np.ndarray]
    final: dict[str, dict[str, np.ndarray]]
    out: dict[str, dict[str, np.ndarray]]
    backward: dict[str, np.ndarray]


def columns(theta: Mapping[str, Any], tangents: Any) -> dict[str, Any]:
    """The tangents of forward mode's basis directions (one row a number of ``theta``, in ``ravel_pytree``'s order:
    the paths sorted) as each axis's derivative, of the axis's shape followed by the objective's."""
    found: dict[str, Any] = {}
    offset = 0
    for path in sorted(theta):
        shape = tuple(_jnp.shape(theta[path]))
        size = int(np.prod(shape, dtype=np.int64))
        block = tangents[offset:offset + size]
        found[path] = _jnp.reshape(block, (*shape, *_jnp.shape(block)[1:]))
        offset += size
    return found


def _numpy(tree: Any) -> Any:
    return _jax.tree_util.tree_map(np.asarray, tree)


def _first(held: Any) -> Any:
    return held[0]


def _one(held: Any) -> Any:
    """An unbatched member's array as a batch of one."""
    return np.asarray(held)[None]


def _call(prepared: _Prepared, theta0: Mapping[str, np.ndarray], factors: Mapping[str, np.ndarray], params: Any,
          state: Any, taps: Mapping[str, Any]) -> list[_Member]:
    value, grad, backward, aux = prepared.function(dict(theta0), dict(theta0), dict(factors), params, state,
                                                   dict(taps))
    tally, final, out = aux
    value, grad, backward, tally, final, out = (_numpy(each) for each in (value, grad, backward, tally, final, out))
    if not prepared.batched:
        value, grad, backward, tally, final, out = (_jax.tree_util.tree_map(_one, each)
                                                    for each in (value, grad, backward, tally, final, out))
    found: list[_Member] = []
    for i in range(int(np.asarray(value).shape[0])):
        found.append(_Member(float(value[i]), {path: np.asarray(held[i]) for path, held in grad.items()},
                             (tally[0][i], tally[1][i], tally[2][i]),
                             {node: {name: held[i] for name, held in fields.items()} for node, fields in final.items()},
                             {node: {name: held[i] for name, held in fields.items()} for node, fields in out.items()},
                             {node: np.asarray(held[i]) for node, held in backward.items()}))
    return found


def _flags(prepared: _Prepared, tally: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[RaisedFlag, ...]:
    counted, first, last = tally
    meanings = dict(prepared.meanings)
    return tuple(RaisedFlag(path := f"{test.node}.{test.field}", meanings.get(path, ""), int(counted[i]),
                            int(first[i]), int(last[i]))
                 for i, test in enumerate(prepared.tests) if int(counted[i]) > 0)


def _extrapolations(point: Point) -> tuple[str, ...]:
    plans = point.plans
    if plans is None:
        return ()
    return tuple(f"{read.material}.{read.field}: {read.use}" for read in plans.materials.reads if read.extrapolated)


def _group_members(plan: StudyPlan, group: Group) -> list[int]:
    return [index for index in group.members if not plan.points[index].refused]


def gradient(experiment: Experiment, plan: StudyPlan, of: Callable[[Any], Any], wrt: Sequence[object] | None,
             mode: str, acknowledged: Mapping[str, str] | None, reasons: Mapping[str, str],
             protocol: Any) -> GradientResult:
    """``Study.gradient``: the derivative of ``of`` (a JAX function of a member's final state, by node and field, to a
    scalar) in each axis of ``wrt`` (the sweep's axes where not given), at each of the plan's points."""
    from mime.study._axes import resolve

    if mode not in MODES:
        raise ArgumentRefused([Refusal("gradient.mode", RefusalClass.USER, f"mode is one of {MODES}, not {mode!r}",
                                       "write mode='auto', 'forward' or 'reverse'")])
    if not callable(of):
        raise ArgumentRefused([Refusal("call.wrong-type", RefusalClass.USER,
                                       f"the objective is a function of the final state, not {type(of).__name__}",
                                       "write of=lambda final: final['ball']['pose'][0]")])
    if plan.refused and not plan.drop_refused:
        raise BuildRefused([Refusal(
            "study.refused-points", RefusalClass.USER,
            f"{len(plan.refused)} of the study's {len(plan.points)} points are refused, so no derivative is taken there",
            "change the axes' values so that every point is one the experiment allows, or sweep(..., "
            "drop_refused=True)")])
    settings, declarations, saved = experiment.declared_as()
    axes = tuple(resolve(path, declarations) for path in wrt) if wrt is not None else plan.axes
    if not axes:
        raise ArgumentRefused([Refusal("gradient.no-axis", RefusalClass.USER, "the derivative is taken in no axis",
                                       "name the axes: study.gradient(of, wrt=['umr.mass'])")])
    acknowledged = dict(acknowledged or {})
    built0 = next(group.built for group in plan.groups if group.built is not None)
    variants = [(axis, what) for axis in axes if (what := _variant(axis, built0)) is not None]
    if variants:
        raise BuildRefused([Refusal(
            "gradient.variant", RefusalClass.USER,
            f"{axis.path} is {what}: it changes how the experiment is computed, not what it is, and no derivative is "
            f"taken in it", "compare it as a variant: `Study(exp).sweep(grid({...}), ...)` over it, each value its "
                            "own group",
            why="a variant is structural, and a derivative is taken in the values one program reads")
            for axis, what in variants])
    members: list[MemberGradient | None] = [None] * len(plan.points)
    statuses: dict[str, GradientStatus] = {}
    chosen = mode
    first: dict[str, Point] = {}
    for point in plan.points:
        if point.key is not None and not point.refused:
            first.setdefault(point.key, point)
    with keeping_pins(saved):
        for group in plan.groups:
            built = group.built
            assert built is not None
            indices = _group_members(plan, group)
            if not indices:
                continue
            leader = plan.points[indices[0]]
            writes = [written(axis, leader, settings, declarations, built, first.get, reasons, protocol)
                      for axis in axes]
            scalars = sum(int(np.size(cast(np.ndarray, magnitude(_value_of(write.axis, leader, settings,
                                                                            declarations))))) for write in writes)
            chosen = mode if mode != "auto" else ("forward" if scalars <= FORWARD_UP_TO else "reverse")
            callback = built.plans.controller is not None and built.plans.controller.mode is Mode.CALLBACK
            with precision(leader.declared.x64):
                found = _group(plan, group, indices, built, writes, of, chosen, callback, settings, declarations,
                               acknowledged)
            for index, member, path in found:
                members[index] = member
                statuses.update(path)
    return GradientResult(tuple(axis.path for axis in axes), chosen, composed(statuses.values()), statuses,
                          tuple(members), {owner: reason for owner, reason in acknowledged.items()
                                           if statuses.get(owner) in ACKNOWLEDGED_ONLY})


def _group(plan: StudyPlan, group: Group, indices: Sequence[int], built: BuiltExperiment, writes: Sequence[Written],
           of: Callable[[Any], Any], mode: str, callback: bool, settings: Any, declarations: Sequence[Any],
           acknowledged: Mapping[str, str]) -> list[tuple[int, MemberGradient, dict[str, GradientStatus]]]:
    graph = built.graph
    points = plan.points
    with state_put_back(graph):
        graph.reset_state()
        meta = dict(full_state(graph).get(META) or {})
    template = {node: dict(fields) for node, fields in final_state(stacked_state(points, indices[:1], meta)).items()}
    template = {node: {name: value[0] for name, value in fields.items()} for node, fields in template.items()}
    shape = tuple(_jax.eval_shape(of, template).shape)
    if shape != ():
        raise BuildRefused([Refusal(
            "gradient.objective", RefusalClass.USER,
            f"the objective gives an array of shape {shape}, and a study's derivative is of one number",
            "return one number from the objective, `of=lambda final: jnp.sum(...)`; a fit takes a vector of "
            "residuals, `Study(exp).fit(residual, ...)`",
            why="the derivative of a scalar in each axis is what the gradient path computes")])
    sources = sorted({node for write in writes for node, _ in (*write.leaves, *write.states)})
    controller = built.plans.controller
    through = [(read.node, write.node) for read in controller.reads for write in controller.writes] \
        if callback and controller is not None else []
    nodes = path_nodes(graph, sources, objective_nodes(of, template), through)
    path = path_statuses(built, nodes, callback)
    refused = unacknowledged(path, acknowledged)
    if refused:
        raise BuildRefused([Refusal(
            "gradient.path-status", RefusalClass.USER,
            "the derivative passes through " + "; ".join(
                f"{owner}, whose status is {path[owner].value}" for owner in refused) +
            ": no derivative passes there (cut) or none is checked (unverified), so the result would not be the "
            "objective's derivative",
            "acknowledge each with its reason, study.gradient(of, ..., acknowledged={" +
            ", ".join(f"{owner!r}: '<why a derivative that stops or is unchecked there is what you need>'"
                      for owner in refused) + "}), or take the derivative in a path that avoids it",
            why="a cut or unverified path is refused unless acknowledged (mime.gradients)")])
    real = np.float64 if _jnp.zeros(()).dtype == np.float64 else np.float32
    found: list[tuple[int, MemberGradient, dict[str, GradientStatus]]] = []
    batched = group.batched and not callback
    runs: list[tuple[list[int], Any]] = []
    if batched:
        runs.append((list(indices), None))
    else:
        for index in indices:
            runs.append(([index], _stretches(built, points[index], group.steps) if callback else None))
    for chunk, fed in runs:
        params, axes = stacked_tree(points, chunk, group.varying)
        prepared = prepare(built, writes, of, mode, params, axes, batched, fed, group.steps)
        theta0, factors = stacked_members(points, chunk, writes, settings, declarations, real)
        state = stacked_state(points, chunk, meta)
        if not batched:
            theta0 = {path: held[0] for path, held in theta0.items()}
            factors = {key: held[0] for key, held in factors.items()}
            state = {node: {name: value[0] for name, value in fields.items()} for node, fields in state.items()}
            params = {"nodes": {node: {leaf: (value[0] if (node, leaf) in set(group.varying) else value)
                                       for leaf, value in leaves.items()} for node, leaves in params["nodes"].items()},
                      "mappings": {}}
        taps = {node: params["nodes"][node][REPORT_LEAF] for node in prepared.solving}
        results = _call(prepared, theta0, factors, params, state, taps)
        reports = [_report(prepared.solving, held.final, held.out, held.backward, mode) for held in results]
        flagged = sorted({report.node for held in reports for report in held
                          if report.forward_short or report.backward_short})
        rechecks = _recheck(prepared, flagged, results, theta0, factors, params, state, taps, mode) if flagged \
            else [None] * len(results)
        for index, held, own, recheck in zip(chunk, results, reports, rechecks, strict=True):
            point = points[index]
            warnings = _warnings(own)
            member = MemberGradient(
                index, {axis.path: value_text(point.values[axis]) for axis in plan.axes}, held.value,
                {write.axis.path: np.asarray(held.gradient[write.axis.path], dtype=np.float64) for write in writes},
                {write.axis.path: _unit(write.axis, point, settings, declarations) for write in writes},
                _flags(prepared, held.tally), tuple(stopped_refusals(graph, held.final)), _extrapolations(point),
                own, recheck if warnings else None, warnings)
            found.append((index, member, path))
    return found


def _unit(axis: Axis, point: Point, settings: Any, declarations: Sequence[Any]) -> str:
    value = point.values[axis] if axis in point.values else base_value(axis, settings, declarations)
    unit = unit_symbol(value.dimension) if isinstance(value, Quantity) else "1"
    return f"the objective's units per {unit or '1'}"


def _stretches(built: BuiltExperiment, point: Point, steps: int) -> list[tuple[int, Any]]:
    """A member's run with its controller called on the host, run forward on the group's compiled step from its own
    start, with each stretch of steps through which the same writes act recorded: what its differentiated run takes as
    its external inputs, held (a derivative through the controller is cut)."""
    from mime.build._maddening import set_node_state

    graph = built.graph
    assert point.plans is not None
    with state_put_back(graph):
        graph.reset_state()
        for name, fields in point.state.items():
            set_node_state(graph, name, fields)
        runs = ChunkedRun(graph, point.plans, point.law)
        runs.stretches = []
        runs.run(steps, history=False, params={"nodes": point.tree, "mappings": {}})
        return list(runs.stretches)


def _recheck(prepared: _Prepared, flagged: Sequence[str], results: Sequence[_Member], theta0: Mapping[str, Any],
             factors: Mapping[str, Any], params: Any, state: Any, taps: Mapping[str, Any],
             mode: str) -> list[Recheck | None]:
    """Each member's derivative again with the tolerance of every node whose solve stopped short (a parameter of the
    program) tightened by :data:`RECHECK_FACTOR`, on the same compiled program, the batch at once: the change in the
    objective and the derivative. A float32 run's floor leaves nothing to tighten, so there the recheck says so and is
    not made here."""
    if _jnp.zeros(()).dtype != np.float64:
        return [Recheck("not made: float32's rounding floor leaves nothing to tighten; run the member in float64 (a "
                        "PRECISION variant) to recheck it", None, held.value, held.gradient, float("nan"),
                        {path: float("nan") for path in held.gradient}, ()) for held in results]
    nodes = {node: dict(leaves) for node, leaves in params["nodes"].items()}
    tolerance = None
    for node in flagged:
        current = np.asarray(nodes[node][TOLERANCE])
        nodes[node][TOLERANCE] = current * RECHECK_FACTOR
        tolerance = float(np.min(current)) * RECHECK_FACTOR
    again = _call(prepared, theta0, factors, {**params, "nodes": nodes}, state, taps)
    found: list[Recheck | None] = []
    for held, other in zip(results, again, strict=True):
        solves = _report(prepared.solving, other.final, other.out, other.backward, mode)
        scale = abs(held.value) or 1.0
        found.append(Recheck(
            f"the same compiled program with the tolerance of {', '.join(flagged)} times {RECHECK_FACTOR:g}", tolerance,
            other.value, other.gradient, abs(other.value - held.value) / scale,
            {path: float(np.max(np.abs(other.gradient[path] - held.gradient[path])) /
                         (float(np.max(np.abs(held.gradient[path]))) or 1.0)) for path in held.gradient}, solves))
    return found


__all__ = ["FORWARD_UP_TO", "MODES", "PROBE_MOVE", "RECHECK_FACTOR", "Written", "factor_of", "gradient", "key",
           "final_state", "magnitude", "member_factor", "prepare", "stacked_members", "objective_nodes", "path_nodes", "path_statuses", "solving_nodes", "written"]
