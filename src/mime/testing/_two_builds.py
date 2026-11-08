"""The two-builds check: whether a compiled program holds a value made from a declared value as a constant of its own
(a silent bake: a study's batched members would all run the value the group was built with, and a derivative would
not see it), from two builds of one experiment that differ only in that value. Float64 only.

**The trace form** (:func:`step_program`): each build's step, as MIME's chunked run executes it (each node's
run-constants made from the parameters, then the step reading them), traced with the state, the external inputs and the
whole parameter tree as arguments, compared by ``mime.build._in_graph.program_of`` (the text with addresses removed,
the constants it closes over bit for bit). A value derived from a leaf at build, a Python float, a NumPy or JAX array,
whatever its share of the result, prints as another literal or constant; a leaf, an argument, does not. It sees the
step only: a value that reaches the run through the state or the external inputs (a callback controller's writes)
leaves the step's program as it was. In float32 a literal prints in float32 and the form stops at about 3e-8 of the
value, so the check refuses to run there.

**The run form** (:class:`RunForm`): the base build's compiled chunk run with the leaves that differ written (and the
other's initial state where it differs), against the other build's own, :data:`RUN_STEPS` steps; each field's largest
difference (at the last step) judged against the program's own resolution there, measured as a saved folder's is
(``mime.saved._scales``): the same written run again from a start one unit in the last place off, every nonzero floating
element of the state and of the small leaves (at most ``mime.core.NUDGED_SIZE`` numbers) one ulp up, each field's
largest divergence from the written run. A field fires above :data:`RUN_FACTOR` times its resolution (any difference,
where it measured none). It sees what the trace form cannot (a value outside the step), and needs a compile of each
build's chunk.

:func:`two_builds` runs both on two builds and gives a :class:`TwoBuilds`, whose ``verdict`` names what was found;
:func:`two_builds_of` makes the second build the way a study makes a point, from a path ``mime.study`` resolves."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np

from mime.build._in_graph import program_of
from mime.build._maddening import (
    chunk_function,
    chunk_step,
    external_inputs,
    fed_inputs,
    full_state,
    graph_params,
    inputs,
    parameter_leaves,
    unreadable_leaves,
)
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.core import NUDGED_SIZE
from mime.experiment import BuiltExperiment, Experiment

#: The steps the run form runs: the spike's.
RUN_STEPS = 5
#: k, the run form's line over a field's measured resolution: declared, provisional. Two builds holding no bake differ
#: by the program's rounding of what each held at build, so k bounds how far the step magnifies that rounding past its
#: one-ulp resolution. Measured before the MRS drag's compared values were leaves: the largest such ratio, 14.6 (a ball's
#: blob in a tube, its matrix made in the program against the one made at build, through the wall's solve); and k times
#: a resolution (0.72 to 3.36 ulps of a field's scale in the plants) stays under the spike's 64-ulp line for k up to 19.
#: Margins of 1.1 and 1.2, so the gating plants sit at least four times k resolutions above the line
#: (``tests/two_builds/test_plants.py``). No pair of the registry walk now needs the run form to arbitrate a trace fire:
#: it is the trace form's backstop, and the one form that sees a value reaching the run outside the step.
RUN_FACTOR = 16.0

type Program = tuple[str, list[np.ndarray]]
type Arrays = dict[str, dict[str, np.ndarray]]


def _x64_only() -> None:
    if not bool(cast(Any, jax.config).jax_enable_x64):
        raise ArgumentRefused([Refusal(
            "two-builds.float32", RefusalClass.USER, "the two-builds check is called in a float32 process",
            "build both in float64 and call it there: `JAX_ENABLE_X64=1`, or `jax.config.update(\"jax_enable_x64\", "
            "True)` before the first build",
            why="in float32 a literal prints in float32, so the trace form cannot see a value moved by less than about "
                "3e-8 of itself")])


def step_program(built: BuiltExperiment) -> Program:
    """The trace form's program of ``built``: its step as MIME's chunked run executes it (``chunk_step``), traced with
    the state (``_meta`` included), the external inputs it is fed and the whole parameter tree as arguments, as
    comparable data (``program_of``). Leaves the graph reset."""
    return graph_program(built.graph, built.held_inputs())


def graph_program(graph: Any, fed: Mapping[str, Mapping[str, Any]] | None = None) -> Program:
    """:func:`step_program` of a graph, built or only assembled (``mime.build._construct.assemble``, as a study's point
    is), fed ``fed`` or, where none is given, a zero for each external input: the inputs are arguments, so their
    values do not enter the program. Compiles the graph with MADDENING where it is not, and leaves it reset."""
    _x64_only()
    if fed is None:
        zeros: dict[str, dict[str, Any]] = {}
        for node, field in external_inputs(graph):
            spec = inputs(graph.get_node(node))[field]
            zeros.setdefault(node, {})[field] = np.zeros(tuple(cast(Any, spec).shape),
                                                         dtype=np.dtype(spec.dtype or np.float64))
        fed = zeros
    traced_step = chunk_step(graph)
    graph.reset_state()
    state = {node: dict(fields) for node, fields in full_state(graph).items()}
    traced = cast(Any, jax).make_jaxpr(traced_step)(state, fed_inputs(graph, fed), graph_params(graph))
    return program_of(traced)


def programs_differ(a: Program, b: Program) -> str:
    """How two programs differ: ``"text"``, ``"consts-count"``, ``"consts-shape"``, ``"consts-value"``, or ``""``."""
    (text_a, consts_a), (text_b, consts_b) = a, b
    if text_a != text_b:
        return "text"
    if len(consts_a) != len(consts_b):
        return "consts-count"
    for x, y in zip(consts_a, consts_b, strict=True):
        if x.shape != y.shape or x.dtype != y.dtype:
            return "consts-shape"
        if not np.array_equal(x, y, equal_nan=True):
            return "consts-value"
    return ""


def differing_lines(a: Program, b: Program, limit: int = 6) -> tuple[str, ...]:
    """The first ``limit`` lines of the two programs' texts that differ, each as ``index: a || b``."""
    found: list[str] = []
    for index, (x, y) in enumerate(zip(a[0].splitlines(), b[0].splitlines())):
        if x != y:
            found.append(f"{index}: {x.strip()} || {y.strip()}")
            if len(found) == limit:
                break
    return tuple(found)


def _arrays(tree: Mapping[str, Mapping[str, Any]]) -> Arrays:
    return {node: {key: np.asarray(value) for key, value in fields.items()} for node, fields in tree.items()}


def _differ(a: Arrays, b: Arrays) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The ``(node, key)`` whose values differ, and those present in one only or of another shape or dtype."""
    moved: list[tuple[str, str]] = []
    reshaped: list[tuple[str, str]] = []
    for node in sorted(set(a) | set(b)):
        for key in sorted(set(a.get(node, {})) | set(b.get(node, {}))):
            x, y = a.get(node, {}).get(key), b.get(node, {}).get(key)
            if x is None or y is None or x.shape != y.shape or x.dtype != y.dtype:
                reshaped.append((node, key))
            elif not np.array_equal(x, y, equal_nan=True):
                moved.append((node, key))
    return moved, reshaped


def _initial(built: BuiltExperiment) -> dict[str, Any]:
    built.graph.reset_state()
    return jax.tree_util.tree_map(np.asarray, dict(full_state(built.graph)))


def _up(value: np.ndarray) -> np.ndarray:
    if value.dtype.kind != "f":
        return value
    up = np.nextafter(value, np.asarray(np.inf, dtype=value.dtype))
    return np.where((value != 0) & np.isfinite(value), up, value).astype(value.dtype)


@dataclass(frozen=True)
class FieldGap:
    """One field of the run form: its largest difference between the written run and the rebuilt one, its resolution
    (its largest divergence from the written run started one ulp off), and its scale (its largest magnitude over both
    runs)."""

    field: str
    difference: float
    resolution: float
    scale: float

    @property
    def ratio(self) -> float:
        """The difference over :data:`RUN_FACTOR` times the resolution: above one fires; infinite where the field
        differs and its resolution measured zero, zero where it does not differ."""
        if self.difference == 0.0:
            return 0.0
        return self.difference / (RUN_FACTOR * self.resolution) if self.resolution > 0.0 else float("inf")


@dataclass(frozen=True)
class RunForm:
    """The run form's finding, field by field (``fields``), and the field whose difference stands highest over its
    line (``worst``; ``None`` where no field differs)."""

    fields: tuple[FieldGap, ...]

    @property
    def worst(self) -> FieldGap | None:
        differing = [gap for gap in self.fields if gap.difference > 0.0]
        return max(differing, key=lambda gap: gap.ratio) if differing else None

    @property
    def ratio(self) -> float:
        worst = self.worst
        return 0.0 if worst is None else worst.ratio

    @property
    def fired(self) -> bool:
        return self.ratio > 1.0


@dataclass(frozen=True)
class TwoBuilds:
    """What the two-builds check found. ``verdict``: ``"quiet"``; ``"bake"`` (the trace form fired with no leaf of
    another shape: a value made from the moved one is a constant of the program); ``"shape"`` (it fired where a leaf
    or a constant changed shape: a value-keyed shape, legitimate only where the move crossed its threshold); ``"held"``
    (it fired where a moved leaf is one the step cannot read, ``unread``: a value its node consumes when it is built
    and declares so, which MADDENING refuses to write into a built graph and a study refuses to vary, so that the move
    takes a rebuild; the trace form cannot tell a bake beside it in the same pair, so a caller names each pair it
    accepts so); or ``"outside-step"`` (the trace form quiet and the run form fired: the value reaches the run outside
    the step, through the state or the external inputs, ``inputs_moved``)."""

    trace_fired: bool
    how: str
    lines: tuple[str, ...]
    moved: tuple[tuple[str, str], ...]
    reshaped: tuple[tuple[str, str], ...]
    state_moved: bool
    inputs_moved: bool
    run: RunForm | None
    unread: tuple[tuple[str, str], ...] = ()

    @property
    def verdict(self) -> str:
        if self.trace_fired:
            if self.reshaped or self.how in ("consts-count", "consts-shape"):
                return "shape"
            return "held" if self.unread else "bake"
        if self.run is not None and self.run.fired:
            return "outside-step"
        return "quiet"


def _run(graph: Any, steps: int, fed: Mapping[str, Mapping[str, Any]], state: Any, params: Any) -> Arrays:
    found = cast(Any, jax).jit(chunk_function(graph, steps, fed))(state, params)
    return {node: {key: np.asarray(value) for key, value in fields.items()}
            for node, fields in cast(dict[str, Any], found).items() if node != "_meta"}


def run_form(base: BuiltExperiment, other: BuiltExperiment, *, steps: int = RUN_STEPS) -> RunForm:
    """The run form on two builds (the module's docstring). Leaves both graphs reset."""
    _x64_only()
    leaves_a, leaves_b = _arrays(parameter_leaves(base.graph)), _arrays(parameter_leaves(other.graph))
    moved, reshaped = _differ(leaves_a, leaves_b)
    if reshaped:
        raise ArgumentRefused([Refusal(
            "two-builds.leaf-shapes", RefusalClass.USER,
            f"the two builds' leaves differ in shape or dtype, or one holds a leaf the other lacks: {reshaped}",
            "compare them with `two_builds(base, other)`, whose trace form names the difference as a `shape` verdict "
            "and runs no run form",
            why="the run form writes one build's leaves into the other's compiled program, which takes leaves of its "
                "own shapes only")])
    state_b = _initial(other)
    tree = graph_params(base.graph)
    written_nodes = {node: {key: (leaves_b[node][key] if (node, key) in moved else value)
                            for key, value in fields.items()} for node, fields in tree["nodes"].items()}
    written = {**tree, "nodes": written_nodes}
    fed = base.held_inputs()
    found = _run(base.graph, steps, fed, state_b, written)
    rebuilt = _run(other.graph, steps, other.held_inputs(), state_b, graph_params(other.graph))

    nudged_nodes = {node: {key: (_up(np.asarray(value)) if np.asarray(value).size <= NUDGED_SIZE else value)
                           for key, value in fields.items()} for node, fields in written_nodes.items()}
    nudged_state = {node: ({key: _up(np.asarray(value)) for key, value in fields.items()} if node != "_meta" else fields)
                    for node, fields in state_b.items()}
    nudged = _run(base.graph, steps, fed, nudged_state, {**tree, "nodes": nudged_nodes})
    base.graph.reset_state()
    other.graph.reset_state()

    gaps: list[FieldGap] = []
    for node in sorted(found):
        for key in sorted(found[node]):
            x, y, z = (np.asarray(each[node][key], dtype=np.float64) for each in (found, rebuilt, nudged))
            if x.size == 0:
                continue
            finite = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
            difference = float(np.max(np.abs(x - y))) if finite else \
                (0.0 if np.array_equal(x, y, equal_nan=True) else float("inf"))
            resolution = float(np.max(np.abs(z - x))) if np.all(np.isfinite(z - x)) else 0.0
            scale = float(max(np.max(np.abs(x)), np.max(np.abs(y)))) if finite else float("inf")
            gaps.append(FieldGap(f"{node}.{key}", difference, resolution, scale))
    return RunForm(tuple(gaps))


def two_builds(base: BuiltExperiment, other: BuiltExperiment, *, run: bool = True,
               steps: int = RUN_STEPS) -> TwoBuilds:
    """The two-builds check on two builds of one experiment that differ only in a declared value (the module's
    docstring): the trace form always, the run form where ``run`` (it compiles each build's chunk). Float64 only."""
    _x64_only()
    a, b = step_program(base), step_program(other)
    how = programs_differ(a, b)
    moved, reshaped = _differ(_arrays(parameter_leaves(base.graph)), _arrays(parameter_leaves(other.graph)))
    state_a, state_b = _initial(base), _initial(other)
    state_moved = bool(_differ({n: f for n, f in state_a.items() if n != "_meta"},
                               {n: f for n, f in state_b.items() if n != "_meta"})[0])
    inputs_moved = bool(_differ(_arrays(base.held_inputs()), _arrays(other.held_inputs()))[0])
    form = run_form(base, other, steps=steps) if run and not reshaped else None
    unread = tuple(each for each in moved if each in unreadable_leaves(base.graph))
    return TwoBuilds(bool(how), how, differing_lines(a, b) if how == "text" else (), tuple(moved), tuple(reshaped),
                     state_moved, inputs_moved, form, unread)


def moved_value(value: object, relative: float) -> object:
    """``value`` times ``1 + relative``: a quantity's SI value element by element, or a float; ``None`` for any other
    value (a direction, a count, a choice), which a two-builds pair does not move."""
    from mime.units import Quantity

    if isinstance(value, Quantity):
        held = np.asarray(value.value, dtype=float) * (1.0 + relative)
        return dataclasses.replace(value, value=tuple(held.tolist()) if held.ndim else float(held))
    if isinstance(value, float):
        return value * (1.0 + relative)
    return None


def two_builds_of(experiment: Experiment, path: object, *, relative: float = 1e-6, to: object | None = None,
                  run: bool = True, steps: int = RUN_STEPS, built: BuiltExperiment | None = None) -> TwoBuilds:
    """The two-builds check on ``experiment`` (or its ``built`` build, where given) and the same experiment with the
    value at ``path`` moved by ``relative`` (or set ``to`` a value), declared the way a study declares a point's
    (``mime.study``'s resolver), both built in full. Float64 only."""
    from mime.saved._pins import keeping_pins
    from mime.study._axes import STUDY_REASON, base_value, declared, resolve

    _x64_only()
    settings, declarations, saved = experiment.declared_as()
    axis = resolve(path, declarations)
    value = to if to is not None else moved_value(base_value(axis, settings, declarations), relative)
    if value is None:
        raise ArgumentRefused([Refusal(
            "two-builds.not-movable", RefusalClass.USER,
            f"the value at {path} is not a quantity or a float, so a relative move of it means nothing",
            f"give the other build's value: `two_builds_of(experiment, {str(path)!r}, to=...)`",
            why="a two-builds pair moves a number by a relative step; a direction, a count or a choice has none")])
    with keeping_pins(saved):
        held = declared(settings, declarations, {axis: value}, STUDY_REASON)
        other = Experiment.declared_from(held.settings, held.declarations).build()
        base = built if built is not None else experiment.build()
    return two_builds(base, other, run=run, steps=steps)


__all__ = ["RUN_FACTOR", "RUN_STEPS", "FieldGap", "RunForm", "TwoBuilds", "differing_lines", "graph_program", "moved_value", "programs_differ",
           "run_form", "step_program", "two_builds", "two_builds_of"]
