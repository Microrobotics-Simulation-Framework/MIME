"""A geometry kind read back from its prim, as the stage writer wrote it: its name and version, its settings in SI
from its native attributes (in the stage's units) or its ``mime:geometry:<setting>`` attributes, and its placement
settings from its transform. Reading a kind is how a stage round trip is checked; taking a stage's kind as the
authority is stage-first authoring, which arrives later."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from mime.registry import EntryKind, Geometry, lookup, versions_of
from mime.scene._derived import GEOMETRY, GEOMETRY_API, GEOMETRY_VERSION, kind_parameters
from mime.scene._frame import Problem
from mime.scene._kinds import Vector, placement_settings, rotate
from mime.scene._pxr import Usd, UsdGeom
from mime.units import Magnitude


@dataclass(frozen=True)
class KindOnStage:
    """A kind as its prim records it: its registered class, name and version, and its settings in SI."""

    cls: type[Geometry]
    name: str
    version: str
    settings: dict[str, Magnitude]


def geometry_schema_problems(prim: Any) -> list[Problem]:
    """A prim's ``mime:geometry`` attributes without ``MimeGeometryAPI`` in its composed ``apiSchemas``, or the
    schema without the kind's name, each refused: the schema's name is read through ``GetPrimTypeInfo``, which sees an
    applied schema no plugin registers."""
    path = str(prim.GetPath())
    applied = GEOMETRY_API in list(prim.GetPrimTypeInfo().GetAppliedAPISchemas())
    named = prim.GetAttribute(GEOMETRY)
    authored = bool(named) and bool(named.HasAuthoredValue())
    others = any(str(a.GetName()).startswith(f"{GEOMETRY}:") and a.HasAuthoredValue() for a in prim.GetAttributes())
    if (authored or others) and not applied:
        return [("geometry.schema-not-applied", f"prim {path} writes {GEOMETRY} attributes without the {GEOMETRY_API} "
                                                f"schema applied", f"apply it on the prim: prepend apiSchemas = "
                                                                   f"[\"{GEOMETRY_API}\"]")]
    if applied and not authored:
        return [("geometry.schema-incomplete", f"prim {path} applies {GEOMETRY_API} and names no kind in {GEOMETRY}",
                 f"write the kind's registered name as {GEOMETRY}, and its version as {GEOMETRY_VERSION}")]
    return []


def _local(prim: Any) -> tuple[Vector, Vector]:
    """The prim's own transform: its translation (in the stage's units) and where it takes its +Z. A prim placed as the
    writer places one, by a translate and an orient op, is read from the two ops, its +Z turned by the quaternion;
    any other from the matrix USD composes, whose third row is where +Z goes (its matrices act on rows). USD builds
    an orient op's matrix less exactly than the quaternion turns a vector (measured: a thousand ulps off near +Z)."""
    xformable = UsdGeom.Xformable(prim)
    ops = list(xformable.GetOrderedXformOps())
    if [op.GetOpType() for op in ops] == [UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient] and \
            not xformable.GetResetXformStack():
        origin, turn = ops[0].Get(), ops[1].Get()
        imaginary = turn.GetImaginary()
        quaternion = (float(turn.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2]))
        return (float(origin[0]), float(origin[1]), float(origin[2])), rotate(quaternion, (0.0, 0.0, 1.0))
    matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    origin = matrix.ExtractTranslation()
    return ((float(origin[0]), float(origin[1]), float(origin[2])),
            (float(matrix[2][0]), float(matrix[2][1]), float(matrix[2][2])))


def read_kind(prim: Any, meters_per_unit: float) -> tuple[KindOnStage | None, list[Problem]]:
    """The kind on ``prim``, in a stage of ``meters_per_unit``; or every problem keeping it from being read."""
    problems = geometry_schema_problems(prim)
    if problems:
        return None, problems
    path = str(prim.GetPath())
    name, version = str(prim.GetAttribute(GEOMETRY).Get()), str(prim.GetAttribute(GEOMETRY_VERSION).Get())
    record = lookup(EntryKind.GEOMETRY, name, version)
    if record is None:
        return None, [("geometry.unknown-kind", f"prim {path} is the kind {name} {version}, which is not registered "
                                                f"(registered versions: {versions_of(EntryKind.GEOMETRY, name)})",
                       "install the package that registers it")]
    cls = cast(type[Geometry], record.cls)
    settings: dict[str, Magnitude] = {}
    native = cls.native
    if native is not None:
        if str(prim.GetTypeName()) != native.primitive:
            return None, [("geometry.native", f"prim {path} is the kind {name}, whose native form is a "
                                              f"{native.primitive}, and is a {prim.GetTypeName() or 'typeless prim'}",
                           f"write it as a {native.primitive}")]
        for attribute, setting in native.lengths.items():
            held = prim.GetAttribute(attribute)
            if not (held and held.HasAuthoredValue()):
                return None, [("geometry.native", f"prim {path} authors no {attribute}: its fallback is not a "
                                                  f"parameter", f"author {attribute} on the prim")]
            settings[setting] = float(held.Get()) * meters_per_unit
        for attribute, token in native.tokens.items():
            if str(prim.GetAttribute(attribute).Get()) != token:
                return None, [("geometry.native", f"prim {path}'s {attribute} is "
                                                  f"{prim.GetAttribute(attribute).Get()!r}, not {token!r}: the kind's "
                                                  f"axis is its own +Z, turned by its transform", f"write {token}")]
    else:
        for setting, value in kind_parameters(prim).items():
            settings[setting] = tuple(value) if isinstance(value, list) else value
    translation, z_axis = _local(prim)
    placed = placement_settings(cls, (translation[0] * meters_per_unit, translation[1] * meters_per_unit,
                                      translation[2] * meters_per_unit), z_axis)
    settings.update(placed)
    if not all(math.isfinite(c) for value in settings.values()
               for c in (value if isinstance(value, tuple) else (value,))):
        return None, [("geometry.native", f"prim {path} holds a setting that is not finite", "write finite numbers")]
    return KindOnStage(cls, name, version, settings), []


__all__ = ["KindOnStage", "geometry_schema_problems", "read_kind"]
