"""A run's overrides (``built.run(d, override={field.commands.frequency: Hz(20.0)})``): the run's point is the built
experiment with the values declared, planned in full and assembled, so its checks are judged again, and it runs on the
built's compiled step with its own parameter tree and initial state. What would need another compiled step is refused
with its route: a value baked into a static or structure needs a rebuild, a numerical setting or a rung is a
variant."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._construct import assemble
from mime.build._control import Mode
from mime.build._key import structure_key
from mime.build._maddening import static_dependencies, unreadable_leaves
from mime.build._plan import Section
from mime.registry import Role
from mime.saved._pins import keeping_pins
from mime.study._axes import OVERRIDE_REASON, Axis, AxisKind, base_value, differ, resolve, value_text
from mime.study._point import Point, build_point, checked_digests, same_run
from mime.units import Quantity, parse_unit, si, unit_symbol

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment



@dataclass(frozen=True)
class Overridden:
    """A run's point: the point as built (its plan, tree and state), each override's record as ``run.json`` holds it,
    and the parameter tree the run takes."""

    point: Point
    records: tuple[dict[str, Any], ...]
    params: Mapping[str, Any]
    #: The nodes whose initial state the overrides change, set before a run from the start.
    starting: tuple[str, ...] = ()


def _site(axis: Axis) -> str:
    """The setting's path as the profile names it: a command's value is ``<apparatus>.<command>``."""
    return f"{axis.owner}.{axis.name}" if axis.kind is AxisKind.COMMAND else axis.path


def _variants(built: BuiltExperiment, axes: Sequence[Axis]) -> list[Refusal]:
    roles = built.plans.profile.roles
    found: list[Refusal] = []
    for axis in axes:
        role = roles.get(_site(axis))
        rung = axis.kind is AxisKind.SETTING and axis.name.split(".")[-1] == "rung"
        if axis.kind in (AxisKind.DT, AxisKind.PRECISION) or rung or (role is not None and role.role is Role.NUMERICAL):
            what = "the rung" if rung else "the run's step" if axis.kind is AxisKind.DT else \
                "the precision" if axis.kind is AxisKind.PRECISION else "a numerical setting"
            found.append(Refusal(
                "override.variant", RefusalClass.USER,
                f"{axis.path} is {what}: it changes how the experiment is computed, not what it is, which is a variant "
                f"of the experiment and not an override of one run",
                "compare it as a variant: a Study over it (Study(exp).sweep(grid({...: [...]}), ...)), each value built "
                "and compiled as its own group", why="an override changes a design or physical value for one run"))
    return found


def _leaves(tree: Mapping[str, Mapping[str, np.ndarray]], other: Mapping[str, Mapping[str, np.ndarray]]
            ) -> list[tuple[str, str]]:
    return [(node, leaf) for node, leaves in tree.items() for leaf, value in leaves.items()
            if not np.array_equal(value, other.get(node, {}).get(leaf, value[None]))]


def _record(axis: Axis, value: object, leaves: Sequence[tuple[str, str]]) -> dict[str, Any]:
    held: dict[str, Any] = {"path": axis.path, "value": value_text(value), "si": None, "unit": None,
                            "leaves": [f"{node}.{leaf}" for node, leaf in leaves]}
    if isinstance(value, Quantity) and not isinstance(value.value, np.ndarray):
        magnitude = si(value, value.dimension, what=axis.path)
        held.update(si=list(magnitude) if isinstance(magnitude, tuple) else magnitude, unit=unit_symbol(value.dimension))
    return held


def replayed(record: Mapping[str, Any]) -> tuple[str, object]:
    """An override as a run folder's ``run.json`` records it, as ``run(override=...)`` takes it again: its path, and
    its value in SI with its unit's dimension."""
    held = record.get("si")
    unit = record.get("unit")
    if held is None or unit is None:
        raise BuildRefused([Refusal(
            "override.unreplayable", RefusalClass.USER,
            f"the override of {record.get('path')!r} was recorded without its SI value and unit ({record.get('value')}), "
            "so it cannot be replayed",
            "run it again with the override given as a quantity: "
            "`built.run_to(out, duration, override={field.commands.frequency: Hz(20.0)})`")])
    value = tuple(float(v) for v in cast(Sequence[float], held)) if isinstance(held, list) else float(held)
    return str(record["path"]), Quantity(value, parse_unit(str(unit)))


def base_point(built: BuiltExperiment) -> Point:
    """The built experiment's own point, assembled as an overridden run's is, so that the two are compared alike."""
    from mime.study._axes import Declared

    held = cast(object, built.own_point)
    if isinstance(held, Point):
        return held
    declarations, settings = built.declared
    plans = built.plans
    with keeping_pins(built.saved_form):
        assembled = assemble(plans, built.law)
    observing = [n for p in plans.entries if p.section is Section.OBSERVATION for n in p.node_names()]
    found = structure_key(assembled.graph, observing)
    assert declarations is not None and settings is not None
    point = Point(0, {}, Declared(settings, declarations, None), plans, built.law)
    point.key, point.tree = found.key, {name: dict(leaves) for name, leaves in found.tree.items()}
    point.state = {name: {key: np.asarray(value) for key, value in
                          cast(Any, assembled.graph.get_node(name)).initial_state().items()}
                   for name in assembled.graph.node_names}
    point.checked = checked_digests(assembled.graph, point.tree)
    built.own_point = point
    return point


def overridden(built: BuiltExperiment, overrides: Mapping[object, object]) -> Overridden:
    """The point of a run of ``built`` with ``overrides``, or the refusal of each one an override cannot be."""
    declarations, settings = built.declared
    if declarations is None or settings is None:
        raise BuildRefused([Refusal(
            "override.no-declarations", RefusalClass.USER, "this built experiment holds no declarations to override",
            "build it from an Experiment, `exp.build().run(duration, override={...})`")])
    axes = {resolve(path, declarations): value for path, value in overrides.items()}
    controller = built.plans.controller
    refusals = _variants(built, list(axes))
    if controller is not None and controller.mode is Mode.CALLBACK:
        refusals.append(Refusal(
            "override.callback", RefusalClass.USER,
            "this experiment's controller is called on the host between chunks, with its law as the build made it, so a "
            "run cannot take values its law may have been made from", "run the values as a Study, whose members each "
            "make their own law, or run the controller inside the compiled step (mode='graph')"))
    if refusals:
        raise BuildRefused(refusals)
    base = base_point(built)
    with keeping_pins(built.saved_form):
        # The base as the reference: a node with checks of its own (the MRS rung's) whose leaves or starting inputs the
        # overrides change is judged again on the run's own point, by the code that judges a study's.
        point = build_point(0, settings, declarations, axes, OVERRIDE_REASON,
                            lambda key: base if key == base.key else None)
    if point.refused:
        raise BuildRefused(list(point.refusals))
    if point.key != base.key:
        baked = [(node, leaf) for node, leaf in _leaves(point.tree, base.tree)
                 if any(leaf in params for params in static_dependencies(built.graph.get_node(node)).values())]
        if baked:
            raise BuildRefused([Refusal(
                "override.baked-leaf", RefusalClass.USER,
                f"the override writes {', '.join(f'{n}.{k}' for n, k in baked)}, which a static of its node is derived "
                f"from when the node is built, so the compiled step does not read it",
                "this needs a rebuild: a study, which builds each value it needs, `Study(exp).sweep(grid({...}), "
                "duration=...)`, or a new experiment with the value declared, `Experiment.based_on(...)`")])
        raise BuildRefused([Refusal(
            "override.structural", RefusalClass.USER,
            f"the override changes the compiled step's structure ({', '.join(a.path for a in axes)}): another node set, "
            f"a static array or a setting the program is built from",
            "this needs a rebuild: a study, which builds each value it needs, `Study(exp).sweep(grid({...}), "
            "duration=...)`, or a new experiment with the value declared, `Experiment.based_on(...)`")])
    changed = _leaves(point.tree, base.tree)
    unread = unreadable_leaves(built.graph)
    dead = [(node, leaf) for node, leaf in changed if (node, leaf) in unread and not leaf.startswith("initial_")]
    if dead:
        raise BuildRefused([Refusal(
            "override.leaf-unread", RefusalClass.USER,
            "; ".join(f"the override writes params['nodes'][{n!r}][{k!r}], which changes nothing in the compiled step: "
                      f"{unread[(n, k)]}" for n, k in dead),
            "this needs a rebuild: a study, which builds each value it needs, `Study(exp).sweep(grid({...}), "
            "duration=...)`, or a new experiment with the value declared, `Experiment.based_on(...)`")])
    starting = [name for name, fields in point.state.items()
                if any(not np.array_equal(value, base.state[name][key]) for key, value in fields.items())]
    if starting and built.steps_taken:
        raise BuildRefused([Refusal(
            "override.initial-after-start", RefusalClass.USER,
            f"the override sets the initial state ({', '.join(starting)}), and this built has run {built.steps_taken} "
            f"steps, so its state is no longer the initial one",
            "build again and run that build with the override from its start: `exp.build().run(duration, "
            "override={...})`")])
    records: list[dict[str, Any]] = []
    inert: list[Refusal] = []
    for axis, value in axes.items():
        with keeping_pins(built.saved_form):
            alone = point if len(axes) == 1 else build_point(0, settings, declarations, {axis: value}, OVERRIDE_REASON)
        own = base_value(axis, settings, declarations)
        if not alone.refused and same_run(alone, base) and (own is None or differ(own, value)):
            inert.append(_no_effect(axis, value, own))
        records.append(_record(axis, value, _leaves(alone.tree, base.tree)))
    if inert:
        raise BuildRefused(inert)
    return Overridden(point, tuple(records), {"nodes": point.tree, "mappings": {}}, tuple(starting))


def _no_effect(axis: Axis, value: object, own: object | None) -> Refusal:
    given = f"the experiment's own {value_text(own)}" if own is not None else "the value the experiment holds there"
    return Refusal(
        "override.no-effect", RefusalClass.USER,
        f"the override {axis.path} = {value_text(value)} changes nothing this run computes from: against {given}, it "
        f"writes no parameter leaf, sets no initial state and changes nothing the compiled step is built from, so the run "
        f"would be the base's own. Either the value is the one the experiment already holds, or the entry that "
        f"consumes it keeps it outside its node's leaves and declared statics",
        "run without the override for the experiment's own value; for a value an entry keeps outside its leaves and "
        "statics, declare it and build again, `Experiment.based_on(...)`, which bakes it into that build, and report "
        "the entry: a node holds every value it consumes in a leaf or a declared static")


__all__ = ["Overridden", "base_point", "overridden", "replayed"]


def _units() -> dict[str, Any]:
    import mime.units as units

    return {held.symbol.replace(" ", ""): held for held in vars(units).values() if isinstance(held, units.Unit)}


_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"


def parse_value(text: str) -> Quantity | None:
    """A value as ``--override`` and a spec file write it: a number or numbers split by commas, with a unit of
    ``mime.units`` (``20Hz``, ``1.2 mT``, ``0,0,-1 mm``) or a unit symbol of the saved form (``m^2 kg s^-1``), or a bare
    number for a dimensionless value; ``None`` for text that is not one."""
    import re

    found = re.fullmatch(rf"\s*\(?\s*((?:{_NUMBER})(?:\s*,\s*{_NUMBER})*)\s*\)?\s*([A-Za-z][A-Za-z0-9/^ -]*)?\s*", text)
    if found is None:
        return None
    numbers = [float(n) for n in found[1].split(",")]
    magnitude = numbers[0] if len(numbers) == 1 else tuple(numbers)
    symbol = (found[2] or "1").strip()
    unit = _units().get(symbol.replace(" ", ""))
    if unit is not None:
        return cast(Quantity, unit(magnitude))
    try:
        return Quantity(magnitude, parse_unit(symbol))
    except ValueError:
        return None


def parse_override(text: str) -> tuple[str, object]:
    """An override as ``mime run --override`` takes it, ``path=value``: the path as a study's axis is written
    (``field.commands.frequency``), and the value as :func:`parse_value` reads it."""
    path, equals, value = text.partition("=")
    quantity = parse_value(value)
    if not equals or not path.strip() or quantity is None:
        raise BuildRefused([Refusal(
            "override.written", RefusalClass.USER,
            f"--override {text!r} is not path=value with a value of numbers and a unit of mime.units",
            "write it as --override field.commands.frequency=20Hz, --override umr.initial.position=0,0,-1mm or "
            "--override umr.mass=8.5e-5kg")])
    return path.strip(), quantity


def verify_run(folder: Any) -> Any:
    """``mime verify --run``: the run a run folder holds, made again from its saved form (``experiment/``) with the
    overrides, the duration and the series its ``run.json`` records, and compared with its ``outputs.npz``, every field
    of the final state and every recorded series: bit for bit in the environment the run was made in, unless it was
    made on a GPU with deterministic mode off, and otherwise within its lock's tolerance, from the tenth form each field
    block by block, on each block's largest magnitude or its resolution as the save measured it over the reference
    steps (a series of an observation's output by its output's). A run written with its chunks set by
    ``stage_every`` is replayed in automatic chunks, which compute the same bits for chunks of two steps or more."""
    import json
    from pathlib import Path

    from mime.env import apply_lock_precision
    from mime.experiment import Experiment
    from mime.experiment._experiment import RUN_EXPERIMENT, RUN_RECORD
    from mime.results import RunResult, Series
    from mime.saved._folder import DETERMINISTIC_OFF, VerifyReport, environment_differences, moved_line, \
        nondeterministic_gpu, recorded_resolution
    from mime.saved._scales import OWN, SCALED_FROM, FieldUnits
    from mime.saved._record import environment
    from mime.units import s

    root = Path(folder)
    record = json.loads((root / RUN_RECORD).read_text(encoding="utf-8"))
    saved = RunResult.read(root)
    apply_lock_precision(root / RUN_EXPERIMENT)
    experiment = Experiment.load(root / RUN_EXPERIMENT)
    built = experiment.build()
    overrides = dict(replayed(held) for held in record.get("overrides", []))
    every = record.get("every")
    again = built.run(s(float(record["duration"])), history=bool(saved.history is not None),
                      record=[] if saved.history is not None else list(record.get("recorded", [])),
                      every=s(float(every)) if every is not None and saved.history is None else None,
                      override=overrides or None)
    form = experiment.declared_as()[2]
    assert form is not None
    lock = form.lock
    same = not environment_differences(lock)
    made = cast(Mapping[str, object], lock["environment"])
    bits, scaled = same and not nondeterministic_gpu(made), form.version >= SCALED_FROM
    reference = cast(Mapping[str, Any], lock["reference"])
    rtol, atol = float(reference["rtol"]), float(reference["atol"])
    finals = {node: {name: np.asarray(value) for name, value in fields.items()} for node, fields in saved.final.items()}
    units = built.field_units(finals)
    resolutions, factor = recorded_resolution(lock)
    pairs: dict[str, tuple[np.ndarray, np.ndarray, str, FieldUnits, tuple[np.ndarray, np.ndarray] | None]] = {}
    for node, fields in finals.items():
        for name, value in fields.items():
            if name not in again.final.get(node, {}):
                continue
            held = units.get(f"{node}.{name}", OWN)
            forces = None
            if held.forces is not None:
                held_node, _, held_name = held.forces.partition(".")
                if held_name in finals.get(held_node, {}) and held_name in again.final.get(held_node, {}):
                    forces = (finals[held_node][held_name], np.asarray(again.final[held_node][held_name]))
            pairs[f"final {node}.{name}"] = (value, np.asarray(again.final[node][name]), f"{node}.{name}", held, forces)
    for path in saved.recorded:
        observation, _, output = path.rpartition(".")
        mine, theirs = saved[observation][output], again[observation][output]
        if isinstance(mine, Series) and isinstance(theirs, Series):
            pairs[f"series {path}"] = (np.asarray(mine.raw), np.asarray(theirs.raw), path, FieldUnits(mine.unit), None)
    moved: list[str] = []
    for label, (a, b, source, held, forces) in sorted(pairs.items()):
        if a.shape != b.shape:
            moved.append(f"{label}: max |difference| inf")
            continue
        if bits or a.dtype.kind not in "fc":
            line = None if np.array_equal(a, b, equal_nan=True) else \
                f"{label}: max |difference| {float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))):.3e}"
        elif scaled:
            line = moved_line(label, a, b, bits=False, scaled=True, rtol=rtol, atol=atol, field=held,
                              resolutions=resolutions.get(source), factor=factor, forces=forces, steps=False)
        else:
            line = None if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True) else \
                f"{label}: max |difference| {float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))):.3e}"
        if line is not None:
            moved.append(line)
    here = environment()
    return VerifyReport(int(saved.steps), same, rtol, atol, f"{here['device']}, {'float64' if here['x64'] else 'float32'}",
                        tuple(moved), tuple(sorted(pairs)), bits=bits, scaled=scaled,
                        warning=DETERMINISTIC_OFF if nondeterministic_gpu(made) else None, factor=factor)


__all__ += ["parse_override", "parse_value", "verify_run"]
