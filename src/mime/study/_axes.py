"""What a study varies, and a point as the experiment with its values declared.

An axis is named by a typed path from a handle (``field.commands.frequency``, ``umr.initial.position``,
``drag.lubrication.epsilon``), by :data:`DT` or :data:`PRECISION`, or by the path's dotted string, which is also how it
saves. A point is, by definition, the base experiment with the point's values declared in it: a command's drive set to
the value, a thing's value declared, a law's setting written, a body's initial state set, the run's step changed. Its
plan judges every build-time check of that experiment, as it would judge the experiment written so."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np

from mime.build import (
    Attachment,
    Declaration,
    RunSettings,
    ThingDeclaration,
    ValueDeclaration,
    entry_id,
)
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.things import (
    Apparatus,
    Body,
    CommandPath,
    InitialPath,
    SettingPath,
    State,
    TypedPath,
    did_you_mean,
)
from mime.drives import Constant
from mime.units import Quantity


class RunPath(TypedPath):
    """A setting of the run itself: :data:`DT`, the base step, or :data:`PRECISION`, ``"float32"`` or ``"float64"``."""

    __slots__ = ()


#: The run's base step, as an axis: a value is a time, a whole divisor of the run's duration.
DT = RunPath("dt")
#: The run's precision, as an axis: ``"float32"`` or ``"float64"``. Each precision is built and run in its own.
PRECISION = RunPath("precision")
PRECISIONS = ("float32", "float64")


class AxisKind(Enum):
    COMMAND = "command"
    VALUE = "value"
    SETTING = "setting"
    INITIAL = "initial"
    DT = "dt"
    PRECISION = "precision"


@dataclass(frozen=True)
class Axis:
    """One thing a study varies: its path as it saves (``field.commands.frequency``), its kind, what it belongs to (an
    apparatus, a thing, a law's id) and its name there (a command, a value, a setting, an initial value)."""

    path: str
    kind: AxisKind
    owner: str = ""
    name: str = ""

    def text(self) -> str:
        return self.path


def _things(declarations: Sequence[Declaration]) -> dict[str, Any]:
    return {d.thing.name: d.thing for d in declarations if isinstance(d, ThingDeclaration)}


def _laws(declarations: Sequence[Declaration]) -> dict[str, Attachment]:
    return {entry_id(type(d.law), d.bindings): d for d in declarations if isinstance(d, Attachment)}


def candidates(declarations: Sequence[Declaration]) -> list[str]:
    """Every axis path the experiment offers by name: each apparatus's commands, each body's initial values, each
    attached law's settings, ``dt`` and ``precision``. A thing's value (``umr.mass``) is named as ``declare`` takes it,
    and the plan checks it."""
    found = ["dt", "precision"]
    for name, thing in _things(declarations).items():
        if isinstance(thing, Apparatus):
            found += [f"{name}.commands.{command}" for command in thing.commands]
        if isinstance(thing, Body):
            found += [f"{name}.initial.{key}" for key in getattr(type(thing.dynamics), "initial_state", {})]
    for law_id, attachment in _laws(declarations).items():
        found += [f"{law_id}.{field.name}" for field in dataclasses.fields(attachment.law)]
    return found


def _unknown(path: str, declarations: Sequence[Declaration], why: str) -> ArgumentRefused:
    return ArgumentRefused([Refusal(
        "study.unknown-path", RefusalClass.USER, f"the axis {path!r} names nothing this experiment declares: {why}",
        f"name it by a handle's typed path (`field.commands.frequency`, `umr.initial.position`, `drag.spacing`), "
        f"`DT` or `PRECISION`, or a thing's value as declare takes it (`\"umr.mass\"`). "
        f"{did_you_mean(path, candidates(declarations))}")])


def resolve(path: object, declarations: Sequence[Declaration]) -> Axis:
    """The axis ``path`` names, in this experiment's declarations; refused (``study.unknown-path``) where it names
    nothing they declare, with the paths they offer."""
    if isinstance(path, RunPath):
        return Axis(str(path), AxisKind.DT if path == DT else AxisKind.PRECISION)
    if isinstance(path, CommandPath):
        return _command(str(path), path.owner, path.command, declarations)
    if isinstance(path, InitialPath):
        return _initial(str(path), path.owner, path.name, declarations)
    if isinstance(path, SettingPath):
        return _setting(str(path), path.owner, path.setting, declarations)
    if isinstance(path, TypedPath):
        return Axis(str(path), AxisKind.VALUE, path.parts[0], ".".join(path.parts[1:]))
    if not isinstance(path, str):
        raise ArgumentRefused([Refusal(
            "call.wrong-type", RefusalClass.USER, f"an axis is a typed path or its string, not {type(path).__name__}",
            "name it by a handle: `field.commands.frequency`, `umr.initial.position`, `drag.spacing`; or "
            "`\"umr.mass\"`")])
    if path in ("dt", "precision"):
        return Axis(path, AxisKind.DT if path == "dt" else AxisKind.PRECISION)
    # A law's id may hold dots of its own (``toy.spring:probe``): the longest id the path starts with is its owner.
    owner = max((law_id for law_id in _laws(declarations) if path.startswith(law_id + ".")), key=len, default=None)
    if owner is not None:
        return _setting(path, owner, path[len(owner) + 1:], declarations)
    head, _, rest = path.partition(".")
    if not rest:
        raise _unknown(path, declarations, "an axis is a thing, a law or the run, and a name in it")
    if ":" in head:
        return _setting(path, head, rest, declarations)
    middle, _, last = rest.partition(".")
    if middle == "commands" and last:
        return _command(path, head, last, declarations)
    if middle == "initial" and last:
        return _initial(path, head, last, declarations)
    if head not in _things(declarations):
        raise _unknown(path, declarations, f"no thing is named {head!r}")
    return Axis(path, AxisKind.VALUE, head, rest)


def _command(path: str, owner: str, command: str, declarations: Sequence[Declaration]) -> Axis:
    thing = _things(declarations).get(owner)
    if not isinstance(thing, Apparatus) or command not in thing.commands:
        raise _unknown(path, declarations, f"no apparatus {owner!r} with a command {command!r}")
    return Axis(path, AxisKind.COMMAND, owner, command)


def _initial(path: str, owner: str, name: str, declarations: Sequence[Declaration]) -> Axis:
    thing = _things(declarations).get(owner)
    if not isinstance(thing, Body) or name not in getattr(type(thing.dynamics), "initial_state", {}):
        raise _unknown(path, declarations, f"no body {owner!r} whose dynamics starts from a value {name!r}")
    return Axis(path, AxisKind.INITIAL, owner, name)


def _setting(path: str, owner: str, setting: str, declarations: Sequence[Declaration]) -> Axis:
    attachment = _laws(declarations).get(owner)
    held: object = attachment.law if attachment is not None else None
    parts = setting.split(".")
    for at, part in enumerate(parts):
        if at and held is None:
            raise ArgumentRefused([Refusal(
                "study.unset-group", RefusalClass.USER,
                f"the setting {'.'.join(parts[:at])!r} of {owner} is not given, so {path} has no group to set",
                f"give the group in the experiment ({parts[at - 1]}=...), then vary its settings")])
        # A quantity is a setting's value, not a group of settings: its own fields are no axis.
        if held is None or isinstance(held, Quantity) or not dataclasses.is_dataclass(held) or \
                part not in {f.name for f in dataclasses.fields(held)}:
            raise _unknown(path, declarations, f"no law {owner!r} with a setting {setting!r}")
        held = getattr(held, part)
    return Axis(path, AxisKind.SETTING, owner, setting)


#: The reason a point's declared value carries: a study chose it. A value that brings its own reason keeps it.
STUDY_REASON = "a study's point: the value the study declares at this point (mime.study)"
OVERRIDE_REASON = "a run's override: the value one run declares for itself (run(override=...))"


def _with_setting(entry: Any, names: Sequence[str], value: object) -> Any:
    if len(names) == 1:
        return dataclasses.replace(entry, **{names[0]: value})
    inner = getattr(entry, names[0])
    if inner is None:
        raise ArgumentRefused([Refusal(
            "study.unset-group", RefusalClass.USER,
            f"the setting {names[0]!r} of {type(entry).__name__} is not given, so {'.'.join(names)} has no group to set",
            f"give the group in the experiment ({names[0]}=...), then vary its settings")])
    return dataclasses.replace(entry, **{names[0]: _with_setting(inner, names[1:], value)})


@dataclass(frozen=True)
class Declared:
    """A point's experiment as data: its run settings and declarations, and the precision it is built in (``None``:
    the process's)."""

    settings: RunSettings
    declarations: tuple[Declaration, ...]
    x64: bool | None


def declared(settings: RunSettings, declarations: Sequence[Declaration], values: Mapping[Axis, object],
             reason: str, reasons: Mapping[str, str] | None = None) -> Declared:
    """The base experiment with ``values`` declared in it: each axis's declaration replaced by one holding its value,
    nothing else changed. A thing's value carries its axis's own reason (``reasons``, by path) where one is given, else
    ``reason``."""
    held = list(declarations)
    x64: bool | None = None
    for axis, value in values.items():
        if axis.kind is AxisKind.DT:
            settings = dataclasses.replace(settings, dt=value)
        elif axis.kind is AxisKind.PRECISION:
            if value not in PRECISIONS:
                raise ArgumentRefused([Refusal(
                    "study.precision", RefusalClass.USER, f"precision is 'float32' or 'float64', not {value!r}",
                    "write the precisions as strings: `grid({PRECISION: [\"float32\", \"float64\"]})`")])
            x64 = value == "float64"
        elif axis.kind is AxisKind.VALUE:
            given = cast(object, getattr(value, "reason", None))
            declaration = ValueDeclaration(axis.path, value,
                                           None if given is not None else (reasons or {}).get(axis.path, reason))
            at = next((i for i, d in enumerate(held) if isinstance(d, ValueDeclaration) and str(d.path) == axis.path),
                      None)
            if at is None:
                held.append(declaration)
            else:
                held[at] = declaration
        else:
            held = [_replaced(d, axis, value) for d in held]
    return Declared(settings, tuple(held), x64)


def _replaced(declaration: Declaration, axis: Axis, value: object) -> Declaration:
    if axis.kind is AxisKind.SETTING and isinstance(declaration, Attachment) and \
            entry_id(type(declaration.law), declaration.bindings) == axis.owner:
        return Attachment(_with_setting(declaration.law, axis.name.split("."), value), declaration.bindings)
    if not isinstance(declaration, ThingDeclaration) or declaration.thing.name != axis.owner:
        return declaration
    thing = declaration.thing
    if axis.kind is AxisKind.COMMAND and isinstance(thing, Apparatus):
        return ThingDeclaration(dataclasses.replace(thing, commands={**thing.commands, axis.name: value}))
    if axis.kind is AxisKind.INITIAL and isinstance(thing, Body):
        return ThingDeclaration(dataclasses.replace(thing, initial=State(**{**dict(thing.initial), axis.name: value})))
    return declaration


def value_text(value: object) -> str:
    return str(value) if isinstance(value, (Quantity, str)) else repr(value)


def differ(a: object, b: object) -> bool:
    """Whether two values of an axis are different values: quantities by their SI values and dimension (``Hz(10.0)``
    and ``kHz(0.01)`` are one value), anything else as it is written."""
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        return a.dimension != b.dimension or not np.array_equal(np.asarray(a.value, dtype=float),
                                                                np.asarray(b.value, dtype=float))
    return value_text(a) != value_text(b)


def base_value(axis: Axis, settings: RunSettings, declarations: Sequence[Declaration]) -> object | None:
    """The value the experiment itself declares where ``axis`` writes: a command's drive, a thing's declared value, a
    law's setting, a body's initial value, the run's step. ``None`` where it declares none there (a value a record or
    a default gives) or for the precision."""
    if axis.kind is AxisKind.DT:
        return settings.dt
    if axis.kind is AxisKind.VALUE:
        return next((d.value for d in declarations if isinstance(d, ValueDeclaration) and str(d.path) == axis.path), None)
    if axis.kind is AxisKind.SETTING:
        held: object = getattr(_laws(declarations).get(axis.owner), "law", None)
        for part in axis.name.split("."):
            held = getattr(held, part, None)
        return held
    thing = _things(declarations).get(axis.owner)
    if axis.kind is AxisKind.COMMAND and isinstance(thing, Apparatus):
        drive = thing.commands.get(axis.name)
        return drive.value if isinstance(drive, Constant) else drive
    if axis.kind is AxisKind.INITIAL and isinstance(thing, Body):
        return dict(thing.initial).get(axis.name)
    return None


__all__ = ["DT", "OVERRIDE_REASON", "PRECISION", "PRECISIONS", "STUDY_REASON", "Axis", "AxisKind", "Declared", "RunPath",
           "base_value", "candidates", "declared", "differ", "resolve", "value_text"]
