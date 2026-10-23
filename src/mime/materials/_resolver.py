"""Where each value of a thing comes from, in one place: a declaration in the experiment, which takes the place of
anything else and says so; a medium's record, read at the experiment's conditions; or a body's parts, through the
part resolver, whose own inputs (a part's density, a magnet's remanence) come from the part's material or a
declaration in the same way."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from mime.materials._read import OperatingCondition, ReadOutcome, read_field
from mime.materials._records import LoadedRecord, Record
from mime.materials._values import (
    DECLARED_IN_THE_EXPERIMENT,
    DERIVED_FROM_PARTS,
    PARTS,
    Declared,
    Directed,
    RecordField,
    ResolvedValue,
    ValueKind,
    ValuePath,
    kind_of_path,
    resolve_declared,
)
from mime.things import DERIVED, Derived, PartMass, ThingKind, derive, magnetisation
from mime.units import unit_symbol

if TYPE_CHECKING:
    from materica.core.record import RecordStore

type Problem = tuple[str, str, str]
type Vector = tuple[float, float, float]


@dataclass(frozen=True)
class PartPlan:
    """A body's part as the build resolved it: its mass properties in the body's frame, its material, and, for a
    magnet part, its magnetisation direction in the body's frame and where that direction came from."""

    mass: PartMass
    material: Record | None = None
    direction: Vector | None = None
    direction_source: str = ""


@dataclass(frozen=True)
class Resolution:
    """A value, or why there is none: ``problems`` to refuse, or ``blocked`` when an input was refused already (a
    record file that could not be read), so that nothing refuses it twice."""

    value: ResolvedValue | None
    problems: tuple[Problem, ...] = ()
    blocked: bool = False
    reads: tuple[ReadOutcome, ...] = field(default=(), compare=False)


def _vec(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{float(c) + 0.0:.6g}" for c in values) + ")"


class Resolver:
    """Every value of the experiment's things and parts, resolved once each, on demand."""

    def __init__(self, things: Mapping[str, ThingKind], declared: Mapping[str, Declared], directed: Directed,
                 conditions: Mapping[str, OperatingCondition], media: Mapping[str, Record],
                 bodies: Mapping[str, Mapping[str, PartPlan]], loaded: Mapping[str, LoadedRecord],
                 store: RecordStore | None) -> None:
        self.things = things
        self.declared = declared
        self.directed = directed
        self.conditions = conditions
        self.media = media
        self.bodies = bodies
        self.loaded = loaded
        self.store = store
        self._values: dict[str, Resolution] = {}
        self._reads: dict[tuple[str, str], ReadOutcome] = {}
        self._derived: dict[str, tuple[dict[str, Derived], dict[str, Resolution]]] = {}

    def value(self, path: str) -> Resolution:
        if path not in self._values:
            self._values[path] = self._resolve(path)
        return self._values[path]

    def _resolve(self, path: str) -> Resolution:
        parsed = ValuePath.parse(path)
        kind = kind_of_path(parsed, self.things)
        if kind is None:
            return Resolution(None)
        held = self.declared.get(path)
        if held is not None:
            resolved = resolve_declared(held, kind, self.directed)
            in_place = self._in_place_of(parsed, kind) if resolved is not None else None
            if resolved is not None and in_place is not None:
                resolved = replace(resolved, source=f"{resolved.source}, in place of {in_place}")
            return Resolution(resolved)
        if parsed.part is not None:
            part = self.bodies.get(parsed.thing, {}).get(parsed.part)
            material = part.material if part is not None else None
            if material is None or kind.record is None:
                return Resolution(None)
            return self._read(material, kind.record, path, kind)
        thing = self.things.get(parsed.thing)
        if thing is ThingKind.MEDIUM and parsed.thing in self.media and kind.record is not None:
            return self._read(self.media[parsed.thing], kind.record, path, kind)
        if thing is ThingKind.BODY and parsed.thing in self.bodies and parsed.name in DERIVED:
            return self._from_parts(parsed, kind)
        return Resolution(None)

    def _in_place_of(self, path: ValuePath, kind: ValueKind) -> str | None:
        """What a declaration at ``path`` takes the place of: the record a medium's value would be read from (not
        read), or the value the body's parts would give."""
        if path.part is not None:
            part = self.bodies.get(path.thing, {}).get(path.part)
            if part is not None and part.material is not None and kind.record is not None:
                return f"material {part.material.name!r}'s {kind.record.field} (not read)"
            return None
        if path.thing in self.media and kind.record is not None:
            return f"material {self.media[path.thing].name!r}'s {kind.record.field} (not read)"
        if path.thing in self.bodies and path.name in DERIVED:
            derived = self._derivation(path.thing)[0][path.name]
            if derived.value is None:
                return "the value its parts would give (none: " + "; ".join(p[1] for p in derived.problems) + ")"
            shown = _vec(derived.value) if isinstance(derived.value, tuple) else f"{derived.value!r}"
            return f"the value its parts give, {shown} {unit_symbol(kind.dimension)}"
        return None

    def _read(self, material: Record, record_field: RecordField, path: str, kind: ValueKind) -> Resolution:
        loaded = self.loaded.get(material.name)
        if loaded is None or self.store is None:
            return Resolution(None, blocked=True)
        key = (material.name, record_field.field)
        if key not in self._reads:
            self._reads[key] = read_field(loaded, material, record_field, self.conditions, self.store)
        outcome = self._reads[key]
        if outcome.value is None:
            return Resolution(None, outcome.problems, reads=(outcome,))
        source = f"record of material {material.name!r}, {loaded.describe()}, {record_field.field}"
        return Resolution(ResolvedValue(path, outcome.value, unit_symbol(kind.dimension), source, outcome.use),
                          reads=(outcome,))

    def _derivation(self, body: str) -> tuple[dict[str, Derived], dict[str, Resolution]]:
        """What the parts of ``body`` give, and the resolutions of the part values it took (by path)."""
        if body in self._derived:
            return self._derived[body]
        parts = self.bodies[body]
        inputs: dict[str, Resolution] = {}
        density: dict[str, float] = {}
        for name, part in parts.items():
            if part.mass.inside is None:
                path = f"{body}.{PARTS}.{name}.density"
                inputs[path] = self.value(path)
                held = inputs[path].value
                if held is not None and isinstance(held.value, float):
                    density[name] = held.value
        moments: dict[str, Vector] = {}
        for name, part in parts.items():
            if part.direction is not None:
                path = f"{body}.{PARTS}.{name}.remanence"
                inputs[path] = self.value(path)
                held = inputs[path].value
                if held is not None and isinstance(held.value, float):
                    moments[name] = magnetisation(held.value, part.direction)
        magnets = [name for name, part in parts.items() if part.direction is not None]
        complete = moments if len(moments) == len(magnets) else {}
        found = derive(body, [part.mass for part in parts.values()], density, complete)
        self._derived[body] = (found, inputs)
        return found, inputs

    def _from_parts(self, path: ValuePath, kind: ValueKind) -> Resolution:
        found, inputs = self._derivation(path.thing)
        derived = found[path.name]
        needs = {"mass": "density", "inertia": "density", "magnetic_moment": "remanence"}.get(path.name)
        used = {key: held for key, held in inputs.items() if needs is not None and key.endswith(f".{needs}")}
        problems = tuple(p for held in used.values() for p in held.problems)
        blocked = any(held.blocked for held in used.values())
        reads = tuple(r for held in used.values() for r in held.reads)
        missing = [key for key, held in used.items() if held.value is None and not held.problems and not held.blocked]
        for key in missing:
            input_kind = kind_of_path(ValuePath.parse(key), self.things)
            meaning = f", {input_kind.meaning}" if input_kind is not None else ""
            problems += (("value.undeclared", f"{path.thing}.{path.name} is derived from its parts, and nothing "
                                              f"provides {key}{meaning}: the part has no material that holds it, and "
                                              f"the experiment declares no value",
                          f"give the part a material, or declare it: exp.declare({key!r}, ..., reason=\"...\")"),)
        if derived.value is None or problems or blocked:
            return Resolution(None, problems or (() if blocked else derived.problems), blocked, reads)
        details = list(derived.details)
        for key, held in used.items():
            if held.value is not None:
                details.append(f"{key} = {held.value.value!r} {held.value.unit}, {held.value.source} (its own line)")
        summary = ", ".join(f"{name} ({part.mass.solid})" for name, part in self.bodies[path.thing].items())
        return Resolution(ResolvedValue(f"{path.thing}.{path.name}", derived.value, unit_symbol(kind.dimension),
                                        DERIVED_FROM_PARTS, f"from {summary}", tuple(details)),
                          derived.problems, reads=reads)

    def inputs_of(self, body: str) -> dict[str, Resolution]:
        """The part values the resolver took to derive ``body``'s values, by path."""
        return dict(self._derivation(body)[1]) if body in self.bodies else {}

    def reads(self) -> list[ReadOutcome]:
        return [self._reads[key] for key in sorted(self._reads)]


__all__ = ["DECLARED_IN_THE_EXPERIMENT", "PartPlan", "Resolution", "Resolver"]
