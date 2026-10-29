"""What MIME hands OpenMaterica's region-writing helper: each region of a structure's geometry kind as one of the
scene manifest's analytic primitives (``geometry.md`` section 3.1), exactly or refused (the tube's lumen, which has
no ends as a region, clipped at the tube's length), in metres in the scene's frame, its directions unit vectors, with
the prim the stage writes it as, and the material reference of the medium that fills it. The helper (OpenMaterica's S6)
does not exist yet: the build stops here, at the seam.

A body is in no manifest (its kinds move; the robot stays out of the plain manifest), so only structures' regions are
handed over. A region's name in the manifest is ``<structure>_<region>``; one that is not ``[a-z0-9_]+`` is refused.
A region with no material reference (no medium fills it, or its medium's values are declared in the experiment) is
handed over without one, for the helper to decide."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from mime.build._plan import MANIFEST_NAME, PlanSet
from mime.build._refusals import MIME_DEFECT, BuildRefused, Refusal, RefusalClass
from mime.build._stage import region_prim_path
from mime.registry import REGION_PRIMITIVES, NoPrimitive, RegionPrimitive, implements
from mime.scene import placed

#: How far from one a direction's or a normal's length may be: MIME's own check, at 64 units in the last place of a
#: float64 (the harness's rounding allowance). OpenMaterica's ``geometry.md`` section 3.1 asks for unit vectors and
#: states no tolerance, and it has no code or case for the check yet. A direction MIME hands over is a unit
#: vector a kind has turned: within 1.5 units in the last place of one over 20,000 random tube axes.
UNIT_TOLERANCE = 64.0 * float(np.finfo(np.float64).eps)

type Vector = tuple[float, float, float]


def primitive_problems(primitive: RegionPrimitive) -> list[str]:
    """Every way ``primitive`` is not one OpenMaterica's section 3.1 reads (MIME's own check of it): a primitive it does
    not define, a field missing or extra, a length not finite and above zero, a point not three finite numbers, a
    direction or normal not a unit vector."""
    fields = REGION_PRIMITIVES.get(primitive.primitive)
    if fields is None:
        return [f"{primitive.primitive!r} is no primitive of the scene manifest's: {sorted(REGION_PRIMITIVES)}"]
    found: list[str] = []
    if set(primitive.fields) != set(fields):
        found.append(f"a {primitive.primitive} has the fields {list(fields)}, and this one {sorted(primitive.fields)}")
    for name, value in primitive.fields.items():
        if name in ("centre", "point", "direction", "normal"):
            held = cast(object, value)
            if not (isinstance(held, tuple) and len(cast(tuple[object, ...], held)) == 3 and all(
                    isinstance(c, float) and math.isfinite(c) for c in cast(tuple[object, ...], held))):
                found.append(f"its {name} is {value!r}, not three finite numbers")
                continue
            if name in ("direction", "normal"):
                vector = cast(Vector, value)
                length = math.sqrt(math.fsum(c * c for c in vector))
                if not abs(length - 1.0) <= UNIT_TOLERANCE:
                    found.append(f"its {name} {list(vector)} has length {length!r}, not one within {UNIT_TOLERANCE}")
        elif not (isinstance(value, float) and math.isfinite(value) and value > 0.0):
            found.append(f"its {name} is {value!r}, not a length above zero")
    return found


def manifest_geometry(primitive: RegionPrimitive) -> dict[str, object]:
    """The primitive as a scene manifest's ``geometry`` member."""
    return {"primitive": primitive.primitive,
            **{name: list(value) if isinstance(value, tuple) else value for name, value in primitive.fields.items()}}


@dataclass(frozen=True)
class HandedRegion:
    """A region handed to the helper: its manifest name, the region it is (``tube.lumen``), its primitive in the
    scene's frame, the prim the stage writer wrote it as (its kind's own prim where the kind is that primitive, a
    sphere's solid; a prim below it for the tube's lumen; ``None`` for a primitive with no USD form, a slab), and the
    medium filling it with its material reference (``None`` where it has none, and ``material_note`` says why)."""

    name: str
    region: str
    primitive: RegionPrimitive
    prim: str | None
    medium: str | None
    material: Mapping[str, str] | None
    material_note: str = ""

    def manifest_region(self) -> dict[str, object]:
        """The region as the manifest's ``regions`` member, for a region with a material; priorities are the
        helper's."""
        if self.material is None:
            raise ValueError(f"region {self.name!r} has no material reference: {self.material_note}")
        return {"name": self.name, "material": dict(self.material), "geometry": manifest_geometry(self.primitive)}


@dataclass(frozen=True)
class RefusedRegion:
    """A region no primitive gives exactly, or with no name the manifest reads, and why."""

    region: str
    reason: str


@dataclass(frozen=True)
class RegionHandoff:
    """Every region of the experiment's structures: handed over, or refused with its reason."""

    regions: tuple[HandedRegion, ...]
    refused: tuple[RefusedRegion, ...]


def _defect(cls: type, label: str, problem: str) -> Refusal:
    return Refusal("entry.defect", RefusalClass.INTERNAL, f"{label} ({cls.__module__}): {problem}",
                   f"a defect in {label}, defined in {cls.__module__}: report it to that package. If that package is "
                   f"MIME: {MIME_DEFECT}")


def region_handoff(plans: PlanSet) -> RegionHandoff:
    """The regions of ``plans``' structures, for OpenMaterica's region-writing helper. Raises ``BuildRefused`` when a
    kind gives a malformed primitive, which is the kind's defect and is never handed over."""
    filled_by = {region: medium for medium, region in plans.fills.items()}
    materials = {held.material: held for held in plans.materials.records}
    media = plans.materials.media
    handed: list[HandedRegion] = []
    refused: list[RefusedRegion] = []
    defects: list[Refusal] = []
    for structure, plan in sorted(plans.geometries.items()):
        kind = plan.geometry
        if not implements(kind, "region_primitive"):
            refused += [RefusedRegion(f"{structure}.{region}", f"its kind, {plan.registered.cls.label()}, gives no "
                                                               f"region primitives") for region in kind.regions]
            continue
        for region in kind.regions:
            path = f"{structure}.{region}"
            primitive = kind.region_primitive(region, plan.settings)
            if isinstance(primitive, NoPrimitive):
                refused.append(RefusedRegion(path, primitive.reason))
                continue
            problems = primitive_problems(primitive)
            if problems:
                defects.append(_defect(type(kind), plan.registered.cls.label(), f"region_primitive({region!r}) "
                                                                                f"gave {'; '.join(problems)}"))
                continue
            name = f"{structure}_{region}"
            if not MANIFEST_NAME.fullmatch(name) or name == "background":
                refused.append(RefusedRegion(path, f"its manifest name {name!r} is not [a-z0-9_]+, or is "
                                                   f"'background'"))
                continue
            medium = filled_by.get(path)
            material, note = None, "no medium fills it"
            if medium is not None:
                record = materials.get(media.get(medium) or "")
                material = record.pin() if record is not None else None
                note = "" if record is not None else f"its medium {medium!r} reads no record: its values are declared"
            prim = region_prim_path(plans.scene.prims[structure], kind, region, primitive)
            handed.append(HandedRegion(name, path, placed(primitive, kind, plan.settings), prim, medium, material,
                                       note))
    if defects:
        raise BuildRefused(defects)
    return RegionHandoff(tuple(handed), tuple(refused))


__all__ = ["UNIT_TOLERANCE", "HandedRegion", "RefusedRegion", "RegionHandoff", "manifest_geometry",
           "primitive_problems", "region_handoff"]
