"""One read of a record's field at the experiment's conditions, through OpenMaterica's plain read, and what the build
does with the verdict. OpenMaterica judges whether the value applies; MIME passes its verdict and its reasons through
unchanged, and decides only what the build does with them, under the experiment's profile (its judge reads each
reason's class and the profile's rule for it, never a code):

- ``met``: the value is used.
- ``not_met``: the build is refused, with OpenMaterica's reasons as it gives them, in every profile. The one exception
  is a table read outside its readable range under the experiment's own extrapolation: OpenMaterica returns the end
  value and MIME computes the extrapolated one by the declared method, recording its basis.
- ``unknown``: each reason is covered by the experiment's acceptance for that read, with its reason ("accepting it
  anyway is your declaration, not the reader's"), or else the profile's rule for the reason's class decides
  (:func:`mime.validity.decide`, the one function every site applies its rules through): a waiver it generates, with a
  pending basis (exploratory, for an assumption: an unstated condition, a value stated without its origin), or a
  listing; any other reason, and one whose class has no rule, refuses the build. ``no_pick`` is refused even then: no
  one value exists until OpenMaterica reads picks.
- An interval: the bounds are read, the experiment's nominal value must lie inside them, and both are recorded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import materica

from mime.build._refusals import ISSUES, Explained
from mime.materials._records import Extrapolation, LoadedRecord, Record
from mime.materials._values import RecordField
from mime.validity import Action, Decision, Finding, Judge, basis_text, decide
from mime.units import (
    FREQUENCY,
    LENGTH,
    MAGNETIC_FLUX_DENSITY,
    TEMPERATURE,
    Condition,
    Dimension,
    UnitError,
    si,
)

if TYPE_CHECKING:
    from materica.core.record import RecordStore

type Problem = tuple[str, str, str]
type Json = object

#: The unit OpenMaterica registers each numeric condition in, by the condition's dimension: SI, so MIME's values
#: pass unconverted.
CONDITION_UNITS: Mapping[Dimension, str] = {TEMPERATURE: "K", FREQUENCY: "1/s", MAGNETIC_FLUX_DENSITY: "T",
                                            LENGTH: "m"}


@dataclass(frozen=True)
class OperatingCondition:
    """An experiment condition as a read sends it: its name, value and tolerance in SI, and OpenMaterica's unit."""

    name: str
    value: float
    plus_minus: float
    unit: str

    def text(self) -> str:
        return f"{self.name} {self.value!r} {self.unit} +- {self.plus_minus!r} {self.unit}"


def operating_conditions(declared: object) -> tuple[dict[str, OperatingCondition], list[Problem]]:
    """The experiment's ``conditions=`` checked: each a :class:`~mime.units.Condition` (a value and its tolerance) in
    a dimension OpenMaterica registers conditions in."""
    if declared is None:
        return {}, []
    if not isinstance(declared, Mapping):
        return {}, [("conditions.form", f"conditions={declared!r} is not a mapping of names to conditions",
                     "write conditions={\"temperature\": degC(20, plus_minus=0.5)}")]
    found: dict[str, OperatingCondition] = {}
    problems: list[Problem] = []
    for name, condition in cast(Mapping[object, object], declared).items():
        if not isinstance(name, str) or not name:
            problems.append(("conditions.form", f"the condition name {name!r} is not a string",
                             "name each condition as OpenMaterica registers it: conditions={\"temperature\": degC(20, "
                             "plus_minus=0.5)}"))
            continue
        if not isinstance(condition, Condition):
            problems.append(("conditions.tolerance", f"the condition {name!r} is {condition!r}, with no tolerance: a "
                                                     f"read checks a record's stated condition against the "
                                                     f"experiment's within a tolerance",
                             f"give it one: conditions={{{name!r}: degC(20, plus_minus=0.5)}}, or "
                             f"Condition(K(...), plus_minus=K(...))"))
            continue
        unit = CONDITION_UNITS.get(condition.value.dimension)
        if unit is None:
            problems.append(("conditions.units", f"the condition {name!r} is {condition}, in "
                                                 f"{condition.value.dimension}, which no condition is registered in",
                             "write a temperature, a shear rate (a frequency), a flux density or a length: "
                             "conditions={\"temperature\": degC(20, plus_minus=0.5)}"))
            continue
        try:
            value = si(condition.value, condition.value.dimension, what=f"the condition {name!r}")
            tolerance = si(condition.plus_minus, condition.value.dimension, what=f"the condition {name!r}'s tolerance")
        except (UnitError, TypeError) as error:
            problems.append(("conditions.units", str(error), f"write {name}'s tolerance in its value's units: "
                                                             "Condition(value, plus_minus=<same units>)"))
            continue
        if isinstance(value, tuple) or isinstance(tolerance, tuple) or not (
                math.isfinite(value) and math.isfinite(tolerance) and tolerance >= 0.0):
            problems.append(("conditions.value", f"the condition {name!r} is {condition}: one finite number with a "
                                                 f"tolerance >= 0", "write one value and its tolerance: "
                                                                    "conditions={\"temperature\": degC(20, "
                                                                    "plus_minus=0.5)}"))
            continue
        found[name] = OperatingCondition(name, value, tolerance, unit)
    return found, problems


@dataclass(frozen=True)
class ReadOutcome:
    """One read and what the build made of it: the request as sent (the record by its pin), OpenMaterica's status and
    reasons as it gave them, the value used in SI (``None`` when refused), the bounds of an interval, how MIME used
    the verdict, and the refusals."""

    material: str
    field: str
    request: Mapping[str, Json]
    status: str
    reasons: tuple[str, ...]
    value: float | None
    use: str
    bounds: tuple[float, float] | None = None
    problems: tuple[Problem, ...] = ()
    #: What the profile's rules decided of the reasons no acceptance covers and its rules let through: a waiver each
    #: (``generate``), with a pending basis, where profiles act on waivers alone, or a listing (``list``).
    decisions: tuple[Decision, ...] = ()


def _reason_text(reason: Mapping[str, object]) -> str:
    text = str(reason.get("code"))
    if "condition" in reason:
        text += f" {reason['condition']}"
    if "alternative" in reason:
        text += f" [alternative {reason['alternative']}]"
    if reason.get("detail"):
        text += f": {reason['detail']}"
    return text


def _keyed(reason: Mapping[str, object]) -> str:
    """A reason as an acceptance names it: its code, or ``code:condition``."""
    code = str(reason.get("code"))
    return f"{code}:{reason['condition']}" if "condition" in reason else code


def _covers(reason: Mapping[str, object], covers: Sequence[str]) -> bool:
    return str(reason.get("code")) in covers or _keyed(reason) in covers


def read_target(material: str, field: str) -> str:
    """A read's name in the experiment, as a waiver of it is named."""
    return f"material {material!r} {field}"


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _stated(result: Mapping[str, object]) -> dict[str, list[Mapping[str, object]]]:
    """What the record states of each condition, by its name: for the value read and for each alternative that
    stays, in their order."""
    groups: list[object] = [result.get("conditions", ())]
    groups += [cast(Mapping[str, object], held).get("conditions", ())
               for held in cast(Sequence[object], result.get("alternative_values", ())) if isinstance(held, Mapping)]
    found: dict[str, list[Mapping[str, object]]] = {}
    for conditions in groups:
        for condition in cast(Sequence[object], conditions):
            if not isinstance(condition, Mapping):
                continue
            held = cast(Mapping[str, object], condition)
            record = held.get("record")
            if isinstance(record, Mapping) and ("value" in record or "token" in record):
                stated = cast(Mapping[str, object], record)
                found.setdefault(str(held.get("name")), [])
                if stated not in found[str(held.get("name"))]:
                    found[str(held.get("name"))].append(stated)
    return found


def _stated_text(stated: Mapping[str, object]) -> str:
    if "token" in stated:
        return str(stated["token"])
    value = _number(stated.get("value"))
    if stated.get("unit") == "K" and value is not None:
        return f"{value - 273.15:.6g} degC"
    return f"{stated.get('value')} {stated.get('unit')}"


def _condition_code(name: str, stated: Mapping[str, object]) -> str:
    """The condition as the experiment's ``conditions=`` writes it, at the value the record states."""
    value = _number(stated.get("value"))
    if stated.get("unit") == "K" and value is not None:
        return f"{name!r}: degC({value - 273.15:.6g}, plus_minus=0.5)"
    return f"{name!r}: Condition(<{stated.get('value')} {stated.get('unit')}>, plus_minus=<its tolerance>)"


def _unchecked_route(names: Sequence[str], stated: Mapping[str, list[Mapping[str, object]]],
                     material: str) -> list[str]:
    """What to write for each condition the read checked against nothing: a numeric one in the experiment's
    ``conditions=``, a named one (a token, such as a test method) in the record's ``require=``, at what the record
    states."""
    numeric = [name for name in names if any("value" in s for s in stated.get(name, ()))]
    named = [name for name in names if name not in numeric]
    routes: list[str] = []
    if numeric:
        routes.append("Experiment(..., conditions={" + ", ".join(_condition_code(name, stated[name][0])
                                                                  for name in numeric) + "})")
    if named:
        tokens = ", ".join(f"{name!r}: ({', '.join(repr(_stated_text(s)) for s in stated.get(name, ())) or '...'},)"
                           for name in named)
        routes.append(f"Record({material!r}, ..., require={{{tokens}}})")
    return routes


def _codes(reasons: Sequence[Mapping[str, object]]) -> str:
    """OpenMaterica's reasons, by their codes and the conditions they name, each once."""
    held: list[str] = []
    for reason in reasons:
        text = str(reason.get("code")) + (f" {reason['condition']}" if "condition" in reason else "")
        if text not in held:
            held.append(text)
    return ", ".join(held)


def request_for(loaded: LoadedRecord, material: Record, field: RecordField,
                conditions: Mapping[str, OperatingCondition]) -> dict[str, Json]:
    """The plain read's request: the field of the record its pin names, at the experiment's conditions, each with its
    tolerance; the tokens the value's meaning fixes and those the experiment accepts; the bounds when the experiment
    declares a nominal value for the field; the experiment's extrapolation as OpenMaterica's acceptance."""
    requirement: dict[str, Json] = {
        name: {"tolerance": {"value": c.plus_minus, "unit": c.unit}} for name, c in conditions.items()}
    for name, tokens in (*field.require.items(), *material.require.items()):
        requirement[name] = {"accept": list(tokens)}
    request: dict[str, Json] = {
        "target": {"record": loaded.pin(), "field": field.field},
        "operating_conditions": {name: {"value": c.value, "unit": c.unit} for name, c in conditions.items()},
        "requirement": {"want": "bounds" if field.field in material.nominal else "value", "conditions": requirement},
    }
    extrapolation = material.extrapolation.get(field.field)
    if extrapolation is not None:
        method, law = (("constant", "the value at the readable range's nearest end") if extrapolation.method ==
                       "constant" else ("consumer_law", "linear continuation of the table's end segment, computed by "
                                                        "MIME from the table the read returns"))
        request["acceptances"] = [{"acceptance": "extrapolation", "record": loaded.pin(), "field": field.field,
                                   "method": method, "law": law, "basis": extrapolation.basis,
                                   "reason": extrapolation.reason}]
    return request


def _extrapolated(result: Mapping[str, object], extrapolation: Extrapolation,
                  conditions: Mapping[str, OperatingCondition]) -> tuple[float | None, str]:
    """The extrapolated value from a refused table read carrying its end value, by the declared method, with how it
    was computed; or ``None`` and why it cannot be."""
    end = _number(result.get("value"))
    table = result.get("table")
    if end is None or not isinstance(table, Mapping):
        return None, "the read carries no end value and table to extrapolate from"
    held = cast(Mapping[str, object], table)
    axis = str(held.get("axis"))
    x = [_number(v) for v in cast(Sequence[object], held.get("x", ()))]
    y = [_number(v) for v in cast(Sequence[object], held.get("y", ()))]
    if axis not in conditions or len(x) < 2 or len(x) != len(y) or None in x or None in y:
        return None, f"the table's axis {axis!r} has no operating condition, or the table is not two knots or more"
    xs, ys = cast(list[float], x), cast(list[float], y)
    at = conditions[axis].value
    low = at < xs[0]
    near = 0 if low else -1
    if extrapolation.method == "constant":
        return end, (f"the value at the readable range's nearest end, {ys[near]!r} at {axis} {xs[near]!r} "
                     f"{conditions[axis].unit}")
    if held.get("scheme") != "linear":
        return None, f"a linear continuation needs a linear table, and this one's scheme is {held.get('scheme')!r}"
    (x0, y0), (x1, y1) = ((xs[0], ys[0]), (xs[1], ys[1])) if low else ((xs[-2], ys[-2]), (xs[-1], ys[-1]))
    slope = (y1 - y0) / (x1 - x0)
    edge_x, edge_y = (x0, y0) if low else (x1, y1)
    value = edge_y + slope * (at - edge_x)
    return value, (f"computed by MIME: the table's end segment continued linearly from {edge_y!r} at {axis} "
                   f"{edge_x!r} to {at!r} {conditions[axis].unit}")


def read_field(loaded: LoadedRecord, material: Record, field: RecordField,
               conditions: Mapping[str, OperatingCondition], store: RecordStore, judge: Judge) -> ReadOutcome:
    """``field`` of ``material``'s record at ``conditions``, and the build's verdict on it under ``judge``'s
    profile."""
    request = request_for(loaded, material, field, conditions)
    where = f"material {material.name!r}: {loaded.describe()}'s {field.field}"
    at = "; ".join(c.text() for c in conditions.values()) or "no stated conditions"
    try:
        result = cast(Mapping[str, object], materica.read(cast(Any, request), store).to_json())
    except NotImplementedError as error:
        return ReadOutcome(material.name, field.field, request, "not read", (), None, "", problems=((
            "record.not-read", f"{where} at {at}: OpenMaterica's reader does not read it yet ({error})",
            "declare the value in the experiment with its reason: exp.declare(...)"),))
    status = str(result.get("status"))
    reasons = [cast(Mapping[str, object], r) for r in cast(Sequence[object], result.get("reasons", ()))
               if isinstance(r, Mapping)]
    texts = tuple(_reason_text(r) for r in reasons)
    given = f"{status} at {at}" + (f" ({'; '.join(texts)})" if texts else "")

    decisions: list[Decision] = []

    def refused(code: str, message: str, action: str, *, why: str | None = None) -> ReadOutcome:
        return ReadOutcome(material.name, field.field, request, status, texts, None, given,
                           problems=((code, Explained(message, why) if why else message, action),),
                           decisions=tuple(decisions))

    extrapolation = material.extrapolation.get(field.field)
    extrapolated: tuple[float | None, str] | None = None
    if status == "not_met":
        if extrapolation is None or result.get("assumption") is None or any(
                str(r.get("code")) != "outside_range" for r in reasons):
            return refused("record.not-met", f"{where} at {at} reads not_met, so the build refuses it. OpenMaterica's "
                                             f"reasons: {'; '.join(texts) or 'none given'}",
                           "read the record at conditions it states, Experiment(..., conditions={...}), name another "
                           "record, or "
                           "declare the value in the experiment with its reason, exp.declare(...)")
        extrapolated = _extrapolated(result, extrapolation, conditions)
    elif status == "unknown":
        stated = _stated(result)
        unchecked = list(dict.fromkeys(str(r["condition"]) for r in reasons
                                       if str(r.get("code")) == "condition_unchecked" and "condition" in r))
        if any(str(r.get("code")) == "no_pick" for r in reasons):
            differ = ", ".join(f"{name} ({' or '.join(_stated_text(s) for s in held)})"
                               for name, held in stated.items() if len(held) > 1) or "conditions it states"
            routes = _unchecked_route(unchecked, stated, material.name) or [
                "narrow the conditions' tolerances in Experiment(..., conditions={...}) to one of its values"]
            return refused("record.no-pick", f"{where}: the record holds more than one value, at different {differ}, "
                                             f"and the experiment's conditions ({at}) do not choose one "
                                             f"(OpenMaterica's verdict: unknown; {_codes(reasons)})",
                           "state the conditions to read it at: " + "; and ".join(routes) + "; or declare the value "
                           "in the experiment with its reason, exp.declare(...)",
                           why="a record's values each hold at the conditions it states, and MIME reads a record at "
                               "the experiment's own conditions, so it needs conditions that leave one value")
        acceptance = material.accept.get(field.field)
        uncovered_reasons: list[Mapping[str, object]] = []
        for reason, text in zip(reasons, texts, strict=True):
            if acceptance is not None and _covers(reason, acceptance.covers):
                continue
            decision = decide(judge, Finding(read_target(material.name, field.field), _keyed(reason), text),
                              Action.REFUSE)
            if decision.action in (Action.GENERATE, Action.LIST):
                decisions.append(decision)
            if not decision.proceeds:
                uncovered_reasons.append(reason)
        if uncovered_reasons:
            unchecked = list(dict.fromkeys(str(r["condition"]) for r in uncovered_reasons
                                           if str(r.get("code")) == "condition_unchecked" and "condition" in r))
            assumed = [r for r in uncovered_reasons if str(r.get("code")) != "condition_unchecked"]
            routes = _unchecked_route(unchecked, stated, material.name)
            if assumed:
                covers = ", ".join(repr(f"{r.get('code')}:{r['condition']}" if "condition" in r
                                        else str(r.get("code"))) for r in assumed)
                routes.append(f"accept what OpenMaterica cannot check, with your reason, if it does not matter here: "
                              f"Record({material.name!r}, ..., accept={{{field.field!r}: "
                              f"Acceptance(covers=({covers},), "
                              f"reason=\"...\")}})")
            return refused("record.unknown", f"{where} at {at} reads unknown, and the experiment covers none of "
                                             f"these reasons (OpenMaterica's verdict: {_codes(uncovered_reasons)})",
                           "; and ".join(routes) + "; or declare the value in the experiment with its reason, "
                                                   "exp.declare(...)",
                           why="OpenMaterica reads unknown where the record leaves a condition unstated, or where "
                               "the experiment states nothing to check a stated condition against")
    elif status != "met":
        return refused("record.not-met", f"{where} at {at}: OpenMaterica's status {status!r}",
                       f"report it to MIME at {ISSUES}, with what `pip show openmaterica` prints: a read answers met, "
                       f"not_met or unknown")
    unit = result.get("unit")
    if unit != field.unit:
        return refused("record.unit", f"{where} is stored in {unit!r}, and MIME reads it in {field.unit!r}",
                       "name a record that stores the field in its registered unit: Record(..., "
                       "packaged_record(\"<record>\"))")
    use = given
    held_acceptance = material.accept.get(field.field)
    if status == "unknown" and held_acceptance is not None and len(decisions) < len(reasons):
        use += f", accepted by the experiment: {basis_text(held_acceptance.reason)}"
    waived = [d.finding.code for d in decisions if d.action is Action.GENERATE]
    listed = [d.finding.code for d in decisions if d.action is Action.LIST]
    if waived:
        use += f", waived by the {judge.profile.value} profile, its reason pending: {', '.join(waived)}"
    if listed:
        use += f", let through by the {judge.profile.value} profile, listed: {', '.join(listed)}"
    if isinstance(result.get("table"), Mapping) and extrapolated is None:
        axis = cast(Mapping[str, object], result["table"]).get("axis")
        use += f"; from the record's table over {axis}, read at {conditions[str(axis)].text()}" \
            if str(axis) in conditions else f"; from the record's table over {axis}"
    if extrapolated is not None:
        value, how = extrapolated
        assert extrapolation is not None
        if value is None:
            return refused("record.extrapolation", f"{where} at {at}: {how}",
                           "declare the value with its reason, exp.declare(...), or read the table inside its range")
        use += f"; extrapolated ({extrapolation.method}): {how}; basis: {extrapolation.basis}; " \
               f"reason: {extrapolation.reason}"
        return ReadOutcome(material.name, field.field, request, status, texts, value, use, decisions=tuple(decisions))
    nominal = material.nominal.get(field.field)
    if nominal is not None:
        bounds = result.get("value")
        held = cast(Mapping[str, object], bounds) if isinstance(bounds, Mapping) else dict[str, object]()
        lower, upper = _number(held.get("lower")), _number(held.get("upper"))
        if lower is None or upper is None:
            return refused("record.not-an-interval", f"{where} at {at} gives no bounds: the experiment declares a "
                                                     f"nominal value for it, which only an interval takes",
                           f"drop the nominal for {field.field} from Record(..., nominal={{...}}), or name a record "
                           "that holds an interval")
        try:
            chosen = si(nominal.value, nominal.value.dimension, what=f"the nominal {field.field}")
        except (UnitError, TypeError) as error:
            return refused("record.nominal", str(error), f"write the nominal in {field.unit}: Nominal(<value in "
                                                         f"{field.unit}>, reason=...)")
        if isinstance(chosen, tuple) or not lower <= chosen <= upper:
            return ReadOutcome(material.name, field.field, request, status, texts, None, use, (lower, upper), ((
                "record.outside-bounds", f"{where}: the experiment uses {nominal.value} from the bounds "
                                         f"[{lower!r}, {upper!r}] {field.unit} ({held.get('basis')}), outside them",
                f"declare a value within [{lower!r}, {upper!r}] {field.unit}: Nominal(..., reason=...)"),))
        use += (f"; bounds [{lower!r}, {upper!r}] {field.unit} ({held.get('basis')}); {chosen!r} {field.unit} "
                f"declared within them: {basis_text(nominal.reason)}")
        return ReadOutcome(material.name, field.field, request, status, texts, chosen, use, (lower, upper),
                           decisions=tuple(decisions))
    value = _number(result.get("value"))
    if value is None:
        return refused("record.form", f"{where} at {at} gives {result.get('value')!r}, not one number",
                       "declare a nominal value for an interval, Record(..., nominal={...}), or name a record that "
                       "holds one number")
    return ReadOutcome(material.name, field.field, request, status, texts, value, use, decisions=tuple(decisions))


__all__ = ["CONDITION_UNITS", "OperatingCondition", "ReadOutcome", "operating_conditions", "read_field", "read_target",
           "request_for"]
