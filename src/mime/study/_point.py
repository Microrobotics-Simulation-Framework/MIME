"""One point of a study, or one overridden run: the experiment with its values declared, planned in full and assembled
with nothing compiled, so that every build-time check of that experiment is judged for it and the compiled program it
needs is found from what it builds (``mime.build._key``)."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import numpy as np

from mime.build import BuildRefused, Declaration, PlanSet, Refusal, RunSettings, plan_experiment
from mime.build._refusals import entry_defect
from mime.build._checks import node_refusals, node_lines  # noqa: PLC2701 - a node's own checks, judged per point
from mime.build._construct import assemble
from mime.build._key import structure_key
from mime.build._maddening import initial_fields
from mime.build._control import Mode
from mime.build._plan import Section
from mime.control import law_and_models
from mime.registry import ControlLaw
from mime.drives import Ramp
from mime.study._axes import Axis, Declared, built_values, declared
from mime.study._protocol import Protocol, declared_with

type Arrays = dict[str, dict[str, np.ndarray]]


@contextlib.contextmanager
def precision(x64: bool | None) -> Generator[None]:
    """The block in float64 (``True``) or float32 (``False``), or in the process's precision (``None``)."""
    if x64 is None:
        yield
        return
    with cast(Any, jax).enable_x64(x64):
        yield


def _bytes(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        held = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(f"{name}|{held.dtype}|{held.shape}|".encode())
        digest.update(held.tobytes())
    return digest.hexdigest()


@dataclass
class Point:
    """A point as built: its index and values, its experiment as data, its plan, the controller's law made from it,
    every refusal judged for it, and, where nothing refused it, its key and physics hash, its parameter tree and
    initial state on the host (``tree``, ``state``), and what its nodes' own checks were judged at (``checked``: by
    node, a digest of the node's leaves and starting inputs) with the lines they computed (``computed``). Its ``key``
    is the program's structure key (``structure``), with the values of the kinds' settings it writes that are built
    per value (``mime.study._axes.built_values``) where it writes any: each such value is a build of its own."""

    index: int
    values: Mapping[Axis, object]
    declared: Declared
    plans: PlanSet | None = None
    law: ControlLaw | None = None
    models: tuple[str, ...] = ()
    refusals: tuple[Refusal, ...] = ()
    key: str | None = None
    structure: str | None = None
    physics_hash: str | None = None
    tree: Arrays = field(default_factory=dict[str, dict[str, np.ndarray]])
    state: Arrays = field(default_factory=dict[str, dict[str, np.ndarray]])
    checked: dict[str, str] = field(default_factory=dict[str, str])
    computed: dict[str, tuple[str, ...]] | None = None
    ramp: Ramp | None = None

    @property
    def refused(self) -> bool:
        return bool(self.refusals)


def _share(own: Arrays, first: Arrays) -> None:
    """Each array of ``own`` equal to its group's first point's replaced by that point's own array: a point holds only
    what differs from it, so a table every point shares is held once, however many points there are."""
    for node, fields in own.items():
        held = first.get(node, {})
        for name, value in fields.items():
            other = held.get(name)
            if other is not None and other.shape == value.shape and other.dtype == value.dtype and \
                    np.array_equal(other, value):
                fields[name] = other


def _starting(graph: Any, name: str) -> dict[str, Any]:
    from mime.build._checks import starting_inputs  # noqa: PLC2701

    return starting_inputs(graph, name)


def checked_digests(graph: Any, tree: Arrays) -> dict[str, str]:
    """By node with checks of its own (``built_problems``, ``built_lines``), a digest of what they are judged at: the
    node's leaves and starting inputs."""
    checking = [name for name in graph.node_names
                if callable(getattr(graph.get_node(name), "built_problems", None))
                or callable(getattr(graph.get_node(name), "built_lines", None))]
    return {name: _bytes({**{f"leaf/{k}": v for k, v in tree.get(name, {}).items()},
                          **{f"input/{k}": v for k, v in _starting(graph, name).items()}}) for name in checking}


def build_point(index: int, settings: RunSettings, declarations: Sequence[Declaration],
                values: Mapping[Axis, object], reason: str,
                reference: Callable[[str], Point | None] | None = None,
                reasons: Mapping[str, str] | None = None, protocol: Protocol | None = None) -> Point:
    """Point ``index``: ``values`` declared in the base, then its ``protocol`` (a ramp's drive in place of its
    command's value; from rest, nothing), planned in full, its controller's law made, assembled with nothing compiled,
    and its key, tree and state read. ``reference`` gives the point already built with the same key, if any: a node
    with checks of its own (``built_problems``, ``built_lines``) whose leaves and starting inputs differ from that
    point's is judged here, on this point's nodes; one that is the same shares its verdict."""
    held = declared(settings, declarations, values, reason, reasons)
    ramp: Ramp | None = None
    if protocol is not None:
        ramped, ramp = declared_with(protocol, held.declarations)
        held = dataclasses.replace(held, declarations=ramped)
    point = Point(index, dict(values), held, ramp=ramp)
    with precision(held.x64):
        try:
            plans = plan_experiment(held.settings, list(held.declarations))
        except BuildRefused as refused:
            point.refusals = tuple(refused.refusals)
            return point
        point.plans = plans
        law, problems, models = law_and_models(plans)
        if problems and plans.controller is not None:
            point.refusals = tuple(entry_defect(plans.controller.registered.cls, problem) for problem in problems)
            return point
        point.law, point.models = law, models
        try:
            assembled = assemble(plans, law)
        except BuildRefused as refused:
            point.refusals = tuple(refused.refusals)
            return point
        graph = assembled.graph
        observing = [n for p in plans.entries if p.section is Section.OBSERVATION for n in p.node_names()]
        found = structure_key(graph, observing)
        built = built_values(list(values), held.declarations)
        point.structure, point.physics_hash, point.tree = found.key, found.physics_hash, \
            {name: dict(leaves) for name, leaves in found.tree.items()}
        point.key = f"{found.key}:{built}" if built else found.key
        point.state = {name: {key: np.asarray(value) for key, value in initial_fields(graph.get_node(name)).items()}
                       for name in graph.node_names}
        point.checked = checked_digests(graph, point.tree)
        checking = list(point.checked)
        first = reference(point.key) if reference is not None else None
        if first is not None:
            _share(point.tree, first.tree)
            _share(point.state, first.state)
        if checking and first is not None and first.checked != point.checked:
            refusals = node_refusals(graph, lambda node: node)
            if refusals:
                point.refusals = tuple(refusals)
                return point
            point.computed = node_lines(graph, plans)
    return point


def _arrays_equal(a: Arrays, b: Arrays) -> bool:
    if a.keys() != b.keys():
        return False
    for node, fields in a.items():
        other = b[node]
        if fields.keys() != other.keys():
            return False
        for name, value in fields.items():
            held = other[name]
            if value is not held and not (value.shape == held.shape and value.dtype == held.dtype
                                          and np.array_equal(value, held)):
                return False
    return True


def _law_context(point: Point) -> object:
    """What a controller called on the host between chunks makes its law from, as data (``None`` for no such
    controller): the entry itself, whose fields its law may read directly, its mode, period, latency and settings in
    SI, and the base step. A law that read a segment's built tree is made from more than this holds, so it compares
    unequal to every other."""
    plans = point.plans
    controller = plans.controller if plans is not None else None
    if plans is None or controller is None or controller.mode is not Mode.CALLBACK:
        return None
    if point.models:
        return object()
    return (controller.entry, controller.mode.value, controller.period, controller.latency, float(plans.dt),
            _bytes({name: np.asarray(value) for name, value in controller.settings.items()}))


def _laws_equal(a: Point, b: Point) -> bool:
    try:
        return bool(_law_context(a) == _law_context(b))
    except (TypeError, ValueError):  # an entry whose fields do not compare (an array's truth): taken as different
        return False


def same_run(a: Point, b: Point) -> bool:
    """Whether two points run one computation: one compiled program (their structure keys), equal parameter leaves and
    initial states, and, where a controller is called on the host, a law made from an equal entry and settings. Two
    such points differ in nothing a run computes from."""
    return a.key is not None and a.key == b.key and _arrays_equal(a.tree, b.tree) and \
        _arrays_equal(a.state, b.state) and _laws_equal(a, b)


def same_build(a: Point, b: Point, programs: dict[int, Any] | None = None) -> bool:
    """Whether two points that differ in a value built per value run one computation: one structure key, equal leaves,
    initial states and host laws, and one traced step, its constants included (each is built for its own value, so a
    value its nodes hold as built shows only in the program: both points are assembled again and traced, which only
    points the rest leaves equal need, each once where ``programs`` holds them by point index)."""
    from mime.study._ranges import assembled_program, programs_differ

    if a.structure is None or a.structure != b.structure or not _arrays_equal(a.tree, b.tree) or \
            not _arrays_equal(a.state, b.state) or not _laws_equal(a, b):
        return False
    held = programs if programs is not None else {}

    def traced(point: Point) -> Any:
        if point.index not in held or point.index < 0:
            with precision(point.declared.x64):
                held[point.index] = assembled_program(point.declared.settings, point.declared.declarations)
        return held[point.index]

    return not programs_differ(traced(a), traced(b))


__all__ = ["Point", "build_point", "checked_digests", "precision", "same_build", "same_run"]
