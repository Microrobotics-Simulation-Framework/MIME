"""The experiment's declared ranges (``exp.declare_range``), as phase 1 plans them and the build judges them.

A range is declared on a value the experiment writes, named as a study names an axis (``mime.study``'s one resolver): a
command's constant drive, a thing's value, a law's setting, a body's initial value or a kind's number. It is refused
on the run's step, the precision, a rung and a numerical setting (each a variant, structural by design), on anything
not one quantity, and where the experiment's own value lies outside it. Every choice keyed on the value is made over
the range by the entry that makes it (``mime.registry.choose_over``), so a run of the experiment uses the range's
discretisation, not its value's own. The build plans the experiment at every corner of its ranges, so that each
validity line is judged over the range, and refuses one whose corner it refuses; past :data:`MAX_RANGES` ranges it
refuses, since the corners double with each."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from mime.build._declarations import Declaration, RangeDeclaration, RunSettings
from mime.build._refusals import BuildRefused, Refusal, RefusalClass
from mime.units import Quantity, unit_symbol

if TYPE_CHECKING:
    from mime.build._plan import PlanSet
    from mime.registry import RangeChoice

#: The most ranges an experiment declares: its build plans the experiment at each of the ``2 ** n`` corners (64 here),
#: a plan of a few milliseconds each and, where an entry chooses over a body's hull, its choice made at each of the
#: hull's corners again. Each range past it doubles the work, and a design space of more parameters than this is a
#: study's axes, each value its own build.
MAX_RANGES = 6
#: The reason a corner's plan declares its values with.
CORNER_REASON = "a corner of the experiment's declared ranges, planned to judge its validity over them"


@dataclass(frozen=True)
class PlannedRange:
    """A declared range as phase 1 checked it: its axis's path and kind, its ends as written and in SI, its unit and its
    reason."""

    path: str
    kind: str
    low: Quantity
    high: Quantity
    low_si: float
    high_si: float
    unit: str
    reason: Any

    def text(self) -> str:
        return f"{self.path} in [{self.low_si:.6g}, {self.high_si:.6g}] {self.unit}"


def _to_si(value: Quantity) -> float | None:
    """A quantity's SI value, where it is one number."""
    held = value.value
    return float(held) if isinstance(held, (int, float)) and not isinstance(held, bool) else None


def planned_ranges(settings: RunSettings, declarations: Sequence[Declaration]
                   ) -> tuple[dict[str, PlannedRange], list[Refusal]]:
    """Each declared range checked against what the experiment declares, with the refusal of each it cannot hold; a
    setting whose role is numerical is refused once the roles are known (:func:`numerical_refusals`)."""
    from mime.study._axes import AxisKind, base_value, resolve

    found: dict[str, PlannedRange] = {}
    refusals: list[Refusal] = []

    def refuse(code: str, message: str, action: str, why: str = "") -> None:
        refusals.append(Refusal(code, RefusalClass.USER, message, action, why=why or None))

    held = [d for d in declarations if isinstance(d, RangeDeclaration)]
    if len(held) > MAX_RANGES:
        refuse("range.too-many", f"the experiment declares {len(held)} ranges, and an experiment declares at most "
                                 f"{MAX_RANGES}: its build plans it at each corner of its ranges, {2 ** len(held)} "
                                 f"plans here, doubling with each range",
               f"declare at most {MAX_RANGES} ranges, `exp.declare_range(...)`, and sweep the other values as a study's "
               f"axes, each value its own build: Study(exp).sweep(grid({{...}}), duration=...)")
        return found, refusals
    for declaration in held:
        written = declaration.path
        path = str(written)
        try:
            axis = resolve(written, declarations)
        except BuildRefused as refused:
            refusals += [replace(r, code="range.path") if r.code == "study.unknown-path" else r
                         for r in refused.refusals]
            continue
        if path in found or axis.path in found:
            refuse("range.duplicate", f"the range of {axis.path} is declared twice",
                   "declare each range once: one exp.declare_range(...) for each path")
            continue
        rung = axis.kind is AxisKind.SETTING and axis.name.split(".")[-1] == "rung"
        if axis.kind in (AxisKind.DT, AxisKind.PRECISION) or rung:
            what = "the rung" if rung else "the run's step" if axis.kind is AxisKind.DT else "the precision"
            refusals.append(variant_refusal(axis.path, what))
            continue
        low, high = declaration.low, declaration.high
        if not isinstance(low, Quantity) or not isinstance(high, Quantity) or low.dimension != high.dimension:
            refuse("range.units", f"the range of {axis.path} is [{low!r}, {high!r}], not two quantities of one "
                                  f"dimension", "write its ends with their units: exp.declare_range(path, mm(1.0), "
                                                "mm(1.2), reason=...)")
            continue
        own = base_value(axis, settings, declarations)
        if isinstance(own, Quantity) and own.dimension != low.dimension:
            refuse("range.units", f"the range of {axis.path} is in {unit_symbol(low.dimension)}, and its value "
                                  f"{own} is in {unit_symbol(own.dimension)}",
                   f"write the range's ends in the value's dimension, {unit_symbol(own.dimension)}")
            continue
        low_si, high_si = _to_si(low), _to_si(high)
        if low_si is None or high_si is None:
            refuse("range.units", f"the range of {axis.path} is [{low}, {high}], and a range is of one number",
                   "declare a range of a value that is one number; a vector's components are not ranged")
            continue
        if not low_si < high_si:
            refuse("range.order", f"the range of {axis.path} is [{low}, {high}], whose low end is not below its high "
                                  f"end", "write the low end first, below the high: exp.declare_range(path, low, high, "
                                          "reason=...)")
            continue
        if own is not None and not isinstance(own, Quantity):
            refuse("range.kind", f"{axis.path} is {own!r}, not a quantity, so it has no range",
                   "declare ranges of values written as quantities (a command's constant value, a law's dimensioned "
                   "setting, a kind's number, a thing's declared value)")
            continue
        own_si = _to_si(own) if isinstance(own, Quantity) else None
        if isinstance(own, Quantity) and own_si is None:
            refuse("range.units", f"{axis.path} is {own}, not one number, so it has no range",
                   "declare a range of a value that is one number")
            continue
        if own_si is not None and not low_si <= own_si <= high_si:
            refuse("range.outside", f"{axis.path} is {own}, outside its declared range [{low}, {high}]",
                   f"declare a range that holds the experiment's own value, or write a value inside it",
                   why="the experiment's run uses the range's choices, so its own value is one of the range's")
            continue
        found[axis.path] = PlannedRange(axis.path, axis.kind.value, low, high, low_si, high_si,
                                        unit_symbol(low.dimension), declaration.reason)
    return found, refusals


def variant_refusal(path: str, what: str) -> Refusal:
    return Refusal("range.kind", RefusalClass.USER,
                   f"{path} is {what}: it changes how the experiment is computed, not what it is, so it has no range: "
                   f"each value of it is a variant of the experiment",
                   "compare it as a variant: a Study over it (Study(exp).sweep(grid({...: [...]}), ...)), each value "
                   "built as its own group; declare ranges of physical or design values",
                   why="a range lets every value inside it share one program, and a variant is another program")


def corners(ranges: Mapping[str, PlannedRange]) -> list[dict[str, Quantity]]:
    """Every corner of ``ranges``: each range at its low or its high end, as the quantities written."""
    paths = sorted(ranges)
    return [dict(zip(paths, ends, strict=True))
            for ends in itertools.product(*[(ranges[p].low, ranges[p].high) for p in paths])]


def corner_text(corner: Mapping[str, Quantity]) -> str:
    return ", ".join(f"{path} {value}" for path, value in sorted(corner.items()))


def judged_over_ranges(settings: RunSettings, declarations: Sequence[Declaration], plans: PlanSet) -> None:
    """The experiment planned at each corner of its declared ranges: refused (``range.corner``) where a corner's plan
    is, with the corner and each of its refusals."""
    from mime.build._plan import plan_experiment
    from mime.study._axes import declared, resolve

    if not plans.ranges:
        return
    found: list[Refusal] = []
    axes = {path: resolve(path, declarations) for path in plans.ranges}
    for corner in corners(plans.ranges):
        held = declared(settings, declarations, {axes[path]: value for path, value in corner.items()}, CORNER_REASON)
        try:
            plan_experiment(held.settings, list(held.declarations))
        except BuildRefused as refused:
            reasons = "; ".join(f"[{r.code}] {r.message}" for r in refused.refusals)
            found.append(Refusal(
                "range.corner", RefusalClass.USER,
                f"at the corner {corner_text(corner)} of the declared ranges the experiment is refused: {reasons}",
                "narrow the range so that every value inside it is one the experiment allows, "
                "exp.declare_range(path, low, high, reason=...), or change what refuses it there",
                why="a run of the experiment uses the range's choices, which hold only where every value of the range "
                    "does"))
    if found:
        raise BuildRefused(found)


def range_lines(plans: PlanSet) -> tuple[str, ...]:
    """The price report: each declared range, and each choice it moved with its cost against the experiment's own
    value's choice."""
    lines: list[str] = []
    units = {path: held.unit for path, held in plans.ranges.items()}
    for path, held in sorted(plans.ranges.items()):
        moved = [choice for choice in plans.range_choices if path in choice.over and choice.moved]
        lines.append(f"{held.text()}: " + ("; ".join(choice.text(units) for choice in moved) if moved else
                                           "no choice of the build moved from the experiment's own value"))
    return tuple(lines)


def choice_refusals(plans_choices: Sequence[RangeChoice], owner_class: Mapping[str, type]) -> list[Refusal]:
    """The entry's defect of each choice whose value needs more than the range's choice gives: the rule it declared
    missed the value (a choice declared monotone that is not)."""
    from mime.build._refusals import entry_defect

    found: list[Refusal] = []
    for choice in plans_choices:
        if not choice.short:
            continue
        problem = (f"its choice {choice.what!r} over the declared range ({', '.join(choice.over)}) gives "
                   f"{choice.chosen:g} at {choice.at}, and the experiment's own value needs {choice.own:g}: the rule it "
                   f"declares ({type(choice.rule).__name__}: {choice.rule.basis}) misses this value, so a run of the "
                   f"range would be coarser here than the value alone; declare the choice Sampled(count, basis) at a "
                   f"count that finds its extreme, or correct the rule")
        found.append(entry_defect(owner_class[choice.owner], problem))
    return found


__all__ = ["CORNER_REASON", "MAX_RANGES", "PlannedRange", "choice_refusals", "corner_text", "corners",
           "judged_over_ranges", "planned_ranges", "range_lines", "variant_refusal"]
