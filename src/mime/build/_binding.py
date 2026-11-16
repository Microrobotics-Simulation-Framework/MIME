"""The stage binding: which field of the built graph's state moves which prim of the stage, published with the stage
so that a client needs no knowledge of MIME's state layout.

Every bound prim is placed in the world: a body's prim sits below ``/World``, which carries no transform, and a chain's
frame prims reset their transform stack. So a pose entry's value is its prim's transform as it stands: the translation
``(x, y, z)`` in metres and the orientation ``(w, x, y, z)``, applied to its ``translate`` and ``orient`` ops with no
composition. A vector entry is a world vector (a field at a body) drawn at the position of the pose it names. A surface's
``points`` entry is its mesh's points, ``(V, 3)`` metres in its thing's frame, and each of its ``field`` entries the values
at its vertices, ``(V,)``, held in the mesh's primvar ``primvars:mime:<field>``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.build._plan import PlanSet
from mime.build._stage import body_poses, segment_prims, mesh_sources, vector_sources
from mime.vocabulary import FRAME_POSES
from mime.vocabulary import POSE as BODY_POSE

BINDING_FORMAT = "mime.binding/1"
POSE, VECTOR, POINTS, FIELD = "pose", "vector", "points", "field"


@dataclass(frozen=True)
class Bound:
    """One prim the state moves: its path; the graph's node and field, and the row of a stacked field (a frame's row
    of its segment's ``frame_poses``); ``kind``, ``pose`` (seven numbers, the prim's world pose) or ``vector`` (three, in
    the world's axes, drawn at the position of the pose ``at`` names, by node and field); the thing it belongs to, its
    public name (``rpm.arm.link_1``) and its units."""

    prim: str
    node: str
    field: str
    index: int | None
    kind: str
    thing: str
    name: str
    unit: str
    at: tuple[str, str] | None = None
    #: A surface's vertices, for its ``points`` and ``field`` entries.
    count: int | None = None
    #: A ``field`` entry's field name, its primvar ``primvars:mime:<name>``.
    surface_field: str | None = None

    def record(self) -> dict[str, Any]:
        held: dict[str, Any] = {"prim": self.prim, "node": self.node, "field": self.field, "index": self.index,
                                "kind": self.kind, "thing": self.thing, "name": self.name, "unit": self.unit,
                                "space": "thing" if self.kind in (POINTS, FIELD) else "world"}
        if self.at is not None:
            held["at"] = {"node": self.at[0], "field": self.at[1]}
        if self.count is not None:
            held["vertices"] = self.count
        if self.surface_field is not None:
            held["surface_field"] = self.surface_field
        return held


@dataclass(frozen=True)
class StageBinding:
    """Every bound prim of a built experiment's stage, in a fixed order: bodies, then each segment's frames in its
    own order, then vectors, then each surface's points and its fields."""

    entries: tuple[Bound, ...]

    def record(self) -> dict[str, Any]:
        return {"format": BINDING_FORMAT, "entries": [entry.record() for entry in self.entries]}

    def fields(self) -> list[tuple[str, str]]:
        """Each (node, field) the binding reads, once, in order."""
        seen: dict[tuple[str, str], None] = {}
        for entry in self.entries:
            seen.setdefault((entry.node, entry.field), None)
        return list(seen)

    def problems(self, prims: Sequence[str], state: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """Every way the binding does not fit a stage (its prim paths, ``prims``) and a state (its arrays by node and
        field): a prim it names that the stage does not hold, a field the state does not have, a row past
        the field's rows, or a field of another shape than its kind."""
        held = set(prims)
        found: list[str] = []
        for entry in self.entries:
            if entry.prim not in held:
                found.append(f"{entry.name}: the prim {entry.prim} is not in the stage")
            value = state.get(entry.node, {}).get(entry.field)
            if value is None:
                found.append(f"{entry.name}: the state has no field {entry.node}.{entry.field}")
                continue
            shape = tuple(int(n) for n in np.shape(value))
            if entry.kind in (POINTS, FIELD):
                expected = (entry.count, 3) if entry.kind == POINTS else (entry.count,)
                if shape != expected:
                    found.append(f"{entry.name}: {entry.node}.{entry.field} has shape {shape}, not {expected}")
                continue
            width = 7 if entry.kind == POSE else 3
            if entry.index is None:
                if shape != (width,):
                    found.append(f"{entry.name}: {entry.node}.{entry.field} has shape {shape}, not ({width},)")
            elif len(shape) != 2 or shape[1] != width or not 0 <= entry.index < shape[0]:
                found.append(f"{entry.name}: row {entry.index} of {entry.node}.{entry.field}, of shape {shape}")
        return found


def stage_binding(plans: PlanSet) -> StageBinding:
    """The binding of a planned experiment's stage: each body's prim to its dynamics' pose, each frame of each chain
    segment to its row of the segment's frame poses, and each field a law reads at a body (a vector in the world's
    axes) to its prim below the field's apparatus, drawn at the body."""
    entries = [Bound(held.prim, held.node, held.field, None, POSE, held.thing, held.thing, BODY_POSE.units)
               for held in body_poses(plans)]
    for owner, held in segment_prims(plans).items():
        entries += [Bound(path, held.node, held.field, index, POSE, held.thing, f"{owner}.{frame}", FRAME_POSES.units)
                    for index, (frame, path) in enumerate(held.frames.items())]
    entries += [Bound(held.prim, held.node, held.field, None, VECTOR, held.thing, held.name, held.unit,
                      (held.at.node, held.at.field)) for held in vector_sources(plans)]
    for held in mesh_sources(plans):
        count = len(held.surface.rest)
        name = f"{held.thing}.{held.surface.name}"
        if held.points is not None:
            entries.append(Bound(held.prim, *held.points, None, POINTS, held.thing, name, "m", count=count))
        entries += [Bound(held.prim, node, output, None, FIELD, held.thing, f"{name}.{declared.name}", declared.unit,
                          count=count, surface_field=declared.name)
                    for declared, (node, output) in zip(held.surface.fields, held.fields, strict=True)]
    return StageBinding(tuple(entries))


__all__ = ["BINDING_FORMAT", "FIELD", "POINTS", "POSE", "VECTOR", "Bound", "StageBinding", "stage_binding"]
