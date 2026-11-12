"""The checks a study makes of an experiment with declared ranges before its first sweep, cached by the experiment's
content: that the range reaches the build only through the choices its entries report (**the independence check**),
and that a value moved inside a range moves no constant of the compiled step (**the two-builds check**'s trace form).

**Independence, two classes.** A numerical setting an entry chooses over a range (a count of force points, a spacing)
is allowed: it is reported in the plan's price report and each validity line is judged at every corner of the range
(``mime.build._ranges``). Any other dependence on the range is forbidden. The check builds the experiment under its
ranges and again without them, every choice the first reported pinned at the point it was made at
(``mime.registry._ranges.pinned``), at the experiment's own value and at the ranges' low and high corners; the two are
one program (the structure key's record, every leaf, the initial state and the traced step, bit for bit) only where the
range reached the build through those choices alone. A difference is refused as the defect of the entry whose node it
is in.

**The trace form**, at a relative move of ``1e-6`` inside each range (towards its middle): the base and the moved
point assembled as a study assembles points, each step traced with the state, the inputs and the parameters as
arguments; a constant of one that the other lacks, in the program or in a sub-program nested in it (a node's ``jit``, a
``cond``, a ``scan``), is a value baked from the ranged one, which every member of a batched group would run at the
first point's value. Refused as that entry's defect, whether or not the move also writes a leaf the step does not read
(``held`` where it changes no constant). A move that changes no leaf, no initial state and not the program is
``not-discriminating``, never ``quiet``: the form had nothing to see. Both checks run in float64, whatever the
study's precision; each is made once for an experiment's content (its document, ranges included, and the code of its
entries' packages) in a process, and reused."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np

from mime.build import (
    BuildRefused,
    Declaration,
    PlanSet,
    RangeDeclaration,
    Refusal,
    RefusalClass,
    RunSettings,
    plan_experiment,
)
from mime.build._construct import assemble
from mime.build._in_graph import nested_program
from mime.build._key import structure_key
from mime.build._maddening import (
    chunk_step,
    external_inputs,
    fed_inputs,
    full_state,
    graph_params,
    initial_fields,
    inputs,
    unreadable_leaves,
)
from mime.build._plan import Section
from mime.build._ranges import corners
from mime.build._refusals import entry_defect
from mime.control import law_and_models
from mime.registry._ranges import pinned, pins_of
from mime.study._axes import Axis, declared, resolve
from mime.study._point import precision
from mime.units import Quantity

if TYPE_CHECKING:
    from mime.experiment import Experiment

#: The relative move the trace form makes inside a range.
MOVE = 1e-6
#: The trace form's verdict on a move that changed no leaf, no initial state and not the step's program: neither build
#: had anything to show, which is not ``quiet`` (the two-builds check's ``not-discriminating``).
NOT_DISCRIMINATING = ("not-discriminating: the move changed no leaf, no initial state and not the step's program, so "
                      "the trace form had nothing to see")
#: The trace form's verdict on a move that writes a leaf the step does not read and changes no constant of the program:
#: a sweep or an override of it is refused elsewhere (``study.leaf-unread``), and the leaf's value never reaches a run.
HELD = "held: the move writes a leaf the step does not read, and changes no constant of the program"
#: The reason the checks' points declare their values with.
CHECK_REASON = "a check of the experiment's declared ranges (mime.study)"

type Arrays = dict[str, dict[str, np.ndarray]]
type Program = tuple[str, list[np.ndarray]]


@dataclass(frozen=True)
class Built:
    """A point as the checks build it: its plan, its graph, the structure key's record, its leaves and initial state."""

    plans: PlanSet
    graph: Any
    record: Mapping[str, Any]
    tree: Arrays
    state: Arrays


@dataclass(frozen=True)
class RangeCheck:
    """What the checks found: each difference the independence check found (empty where none), by the point it was
    found at; each ranged path's trace-form verdict (``quiet``, ``held``, ``not-discriminating`` or ``fired``); their
    refusals; and what they cost, in seconds, where they ran (``0.0`` where the verdicts were reused)."""

    differences: tuple[str, ...]
    traced: Mapping[str, str]
    refusals: tuple[Refusal, ...]
    seconds: float
    reused: bool = False


def _built(settings: RunSettings, declarations: Sequence[Declaration]) -> Built:
    plans = plan_experiment(settings, list(declarations))
    law, problems, _ = law_and_models(plans)
    if problems and plans.controller is not None:
        raise BuildRefused([entry_defect(plans.controller.registered.cls, problem) for problem in problems])
    graph = assemble(plans, law).graph
    observing = [n for p in plans.entries if p.section is Section.OBSERVATION for n in p.node_names()]
    found = structure_key(graph, observing)
    tree = {name: {key: np.asarray(value) for key, value in leaves.items()} for name, leaves in found.tree.items()}
    state = {name: {key: np.asarray(value) for key, value in initial_fields(graph.get_node(name)).items()}
             for name in graph.node_names}
    return Built(plans, graph, found.record, tree, state)


def step_program(graph: Any) -> Program:
    """An assembled graph's step, as MIME's chunked run executes it, traced with the state, a zero for each external
    input and the parameters as arguments: its text without addresses and the constants it closes over, with those of
    every nested sub-program (``mime.build._in_graph.nested_program``: a node's step runs inside a ``jit``, whose
    closed-over arrays are that sub-program's constants)."""
    graph.compile()
    zeros: dict[str, dict[str, Any]] = {}
    for node, field in external_inputs(graph):
        spec = inputs(graph.get_node(node))[field]
        zeros.setdefault(node, {})[field] = np.zeros(tuple(cast(Any, spec).shape),
                                                     dtype=np.dtype(spec.dtype or np.float64))
    traced_step = chunk_step(graph)
    graph.reset_state()
    state = {node: dict(fields) for node, fields in full_state(graph).items()}
    traced = cast(Any, jax).make_jaxpr(traced_step)(state, fed_inputs(graph, zeros), graph_params(graph))
    return nested_program(traced)


def assembled_program(settings: RunSettings, declarations: Sequence[Declaration]) -> Program:
    """The step of the experiment ``declarations`` declare, assembled as a study assembles a point, traced
    (:func:`step_program`)."""
    return step_program(_built(settings, declarations).graph)


def programs_differ(a: Program, b: Program) -> bool:
    (text_a, consts_a), (text_b, consts_b) = a, b
    return text_a != text_b or len(consts_a) != len(consts_b) or any(
        x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=True)
        for x, y in zip(consts_a, consts_b, strict=True))


def _array_differences(a: Arrays, b: Arrays, what: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node in sorted(set(a) | set(b)):
        for key in sorted(set(a.get(node, {})) | set(b.get(node, {}))):
            x, y = a.get(node, {}).get(key), b.get(node, {}).get(key)
            if x is None or y is None or x.shape != y.shape or x.dtype != y.dtype or \
                    not np.array_equal(x, y, equal_nan=True):
                found.append((node, f"{what} {key}"))
    return found


def _record_nodes(record: Mapping[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    for part in ("physics", "observations"):
        section = cast(Mapping[str, Any], record.get(part, {}))
        for node in cast(list[Mapping[str, Any]], section.get("nodes", [])):
            found[str(node["name"])] = json.dumps(node, sort_keys=True, default=str)
        for name, held in cast(Mapping[str, Any], section.get("static") or {}).items():
            found[f"{name}/static"] = json.dumps(held, sort_keys=True, default=str)
    return found


def _differences(ranged: Built, bare: Built) -> list[tuple[str, str]]:
    """``(node, what)`` for each way the two builds differ: a node's description or static data, a leaf, an initial
    state, or (naming no node) the graph's other sections and the traced step."""
    found: list[tuple[str, str]] = []
    a, b = _record_nodes(ranged.record), _record_nodes(bare.record)
    found += [(name.removesuffix("/static"), "its description" if not name.endswith("/static") else "its static data")
              for name in sorted(set(a) | set(b)) if a.get(name) != b.get(name)]
    for section in ("edges", "external_inputs", "coupling_groups"):
        for part in ("physics", "observations"):
            if cast(Mapping[str, Any], ranged.record.get(part, {})).get(section) != \
                    cast(Mapping[str, Any], bare.record.get(part, {})).get(section):
                found.append(("", f"the graph's {section.replace('_', ' ')}"))
    found += _array_differences(ranged.tree, bare.tree, "the leaf")
    found += _array_differences(ranged.state, bare.state, "the initial")
    if not found and programs_differ(step_program(ranged.graph), step_program(bare.graph)):
        found.append(("", "the traced step (a constant the range's build holds and the pinned build does not)"))
    return found


def _point_text(values: Mapping[Axis, object]) -> str:
    return ", ".join(f"{axis.path} {value}" for axis, value in values.items()) or "the experiment's own value"


def _owner_class(plans: PlanSet, node: str) -> type | None:
    for plan in plans.entries:
        if node in plan.node_names():
            return plan.registered.cls
    return None


def independence(settings: RunSettings, declarations: Sequence[Declaration]) -> tuple[list[str], list[Refusal]]:
    """The independence check (the module's docstring): each difference found, in words, and its refusal."""
    held_ranges = [d for d in declarations if isinstance(d, RangeDeclaration)]
    if not held_ranges:
        return [], []
    plans = plan_experiment(settings, list(declarations))
    axes = {path: resolve(path, declarations) for path in plans.ranges}
    every = corners(plans.ranges)
    points: list[dict[Axis, object]] = [{}]
    points += [{axes[path]: cast(object, value) for path, value in every[i].items()} for i in sorted({0, len(every) - 1})]
    found: list[str] = []
    by_node: dict[str, list[str]] = {}
    classes: dict[str, type | None] = {}
    for values in points:
        at = declared(settings, declarations, values, CHECK_REASON)
        ranged = _built(at.settings, at.declarations)
        bare_declarations = [d for d in at.declarations if not isinstance(d, RangeDeclaration)]
        with pinned(pins_of(ranged.plans.range_choices)):
            bare = _built(at.settings, bare_declarations)
        differing: dict[str, list[str]] = {}
        for node, what in _differences(ranged, bare):
            differing.setdefault(node, []).append(what)
        for node, whats in differing.items():
            text = f"at {_point_text(values)}, {', '.join(whats)}"
            found.append(f"{f'node {node}' if node else 'the build'}: {text}")
            by_node.setdefault(node, []).append(text)
            classes[node] = _owner_class(ranged.plans, node) if node else None
    refusals: list[Refusal] = []
    for node, texts in by_node.items():
        problem = (f"{f'node {node}' if node else 'the build'} differs between the build over the declared ranges and "
                   f"the build without them with every reported choice pinned ({'; '.join(texts)}): it reads the range "
                   f"other than through a choice it reports (mime.registry.choose_over), a dependence a declared range "
                   f"forbids")
        cls = classes[node]
        refusals.append(entry_defect(cls, problem) if cls is not None else Refusal(
            "study.range-dependent", RefusalClass.INTERNAL, problem,
            "report it: a declared range reaches the build only through the choices entries report"))
    return found, refusals


def _moved(value: Quantity, low: float, high: float) -> Quantity:
    number = float(cast(float, value.value))
    middle = 0.5 * (low + high)
    moved = number * (1.0 + MOVE) if number <= middle else number * (1.0 - MOVE)
    return replace(value, value=moved)


def traced(settings: RunSettings, declarations: Sequence[Declaration]) -> tuple[dict[str, str], list[Refusal]]:
    """The trace form at a move of :data:`MOVE` inside each range (the module's docstring): each ranged path's verdict,
    and the refusal of each that fired."""
    from mime.study._axes import base_value

    plans = plan_experiment(settings, list(declarations))
    base = _built(settings, declarations)
    first = step_program(base.graph)
    verdicts: dict[str, str] = {}
    refusals: list[Refusal] = []
    for path, held in plans.ranges.items():
        axis = resolve(path, declarations)
        own = base_value(axis, settings, declarations)
        if not isinstance(own, Quantity):
            verdicts[path] = "not moved: the experiment writes no value there (a record or a default gives it)"
            continue
        at = declared(settings, declarations, {axis: _moved(own, held.low_si, held.high_si)}, CHECK_REASON)
        other = _built(at.settings, at.declarations)
        unread = unreadable_leaves(base.graph)
        moved = [(node, key) for node, key in _array_differences(other.tree, base.tree, "")]
        moved = [(node, key.strip()) for node, key in moved]
        if not programs_differ(first, step_program(other.graph)):
            verdicts[path] = (HELD if any(pair in unread for pair in moved) else
                              "quiet" if moved or _array_differences(other.state, base.state, "") else
                              NOT_DISCRIMINATING)
            continue
        verdicts[path] = "fired"
        node = next((n for n, _ in moved), "")
        cls = _owner_class(base.plans, node) if node else None
        problem = (f"moving {path} by a relative {MOVE:g} inside its declared range changes a constant of the compiled "
                   f"step: a value made from it at build is held in the program, so every member of a batched group "
                   f"would run the first point's value")
        refusals.append(entry_defect(cls, problem) if cls is not None else Refusal(
            "study.range-bake", RefusalClass.INTERNAL, problem,
            "report it: a value made from a declared value is computed in the program, never held as a constant"))
    return verdicts, refusals


def _sources(declarations: Sequence[Declaration]) -> dict[str, str]:
    """The sha256 of every file the checks' verdicts rest on: MIME's package, and the module of each class an
    experiment's declarations hold (a lab's entry, a test's planted one)."""
    from mime.study._content import source_hashes

    held = source_hashes(())
    modules: set[str] = set()
    for declaration in declarations:
        for value in vars(declaration).values() if hasattr(declaration, "__dict__") else ():
            modules.add(type(value).__module__)
    for name in sorted(modules):
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None)
        if isinstance(file, str) and Path(file).is_file() and not name.startswith("mime."):
            held[name] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    return held


def check_key(experiment: Experiment) -> str | None:
    """What the checks' verdicts are cached by: the experiment's document (its ranges included) and its code; ``None``
    where it has no document (a setting the document cannot hold), and the checks run every time."""
    from mime.study._content import digest_of

    try:
        document = experiment.document()
    except Exception:  # noqa: BLE001 - an experiment the document cannot hold is checked every time
        return None
    _, declarations, _ = experiment.declared_as()
    held = {"document": json.loads(json.dumps(document, sort_keys=True, default=str)),
            "sources": digest_of(_sources(declarations))}
    return hashlib.sha256(json.dumps(held, sort_keys=True).encode()).hexdigest()


_CACHE: dict[str, RangeCheck] = {}


def forget() -> None:
    """Drop every cached verdict: what a test that patches an entry in place does, since the files do not change."""
    _CACHE.clear()


def range_check(experiment: Experiment) -> RangeCheck:
    """Both checks on ``experiment`` (the module's docstring), or the verdicts already made for its content."""
    from mime.saved._pins import keeping_pins

    key = check_key(experiment)
    if key is not None and key in _CACHE:
        return replace(_CACHE[key], seconds=0.0, reused=True)
    settings, declarations, saved = experiment.declared_as()
    start = time.perf_counter()
    with keeping_pins(saved), precision(True):
        differences, independent = independence(settings, declarations)
        verdicts, baked = traced(settings, declarations) if not independent else ({}, [])
    found = RangeCheck(tuple(differences), verdicts, tuple(independent + baked), time.perf_counter() - start)
    if key is not None:
        _CACHE[key] = found
    return found


__all__ = ["CHECK_REASON", "HELD", "MOVE", "NOT_DISCRIMINATING", "RangeCheck", "assembled_program", "forget",
           "independence", "programs_differ", "range_check", "step_program", "traced"]
