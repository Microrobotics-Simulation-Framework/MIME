"""The imageable prims a composed stage holds that no thing owns: the scenery an in-loop camera is to image (a vessel's
STL in a referenced asset layer). A ``Gprim`` (a mesh, an implicit shape, curves, points) is scenery unless a thing owns
it: it lies below a prim a thing is bound to (a body's, an apparatus's), or it is a kind's derived prim
(``mime:derived_from``, imaged by its kind). A ``Mesh`` is read as the stage holds it: its points in metres in the world
(the transforms down to it applied), its triangles (polygons fanned, wound counter-clockwise seen from outside whatever
the prim's orientation and its transform's handedness), and its display colour and opacity with the prim that authored
each (``FindPrimvarWithInheritance``: a mesh takes the colour of its visual prim; indices resolved, a non-constant
primvar reduced to one value and said so). Every other gprim is found and named (``SceneryGprim``, its type) but not read:
nothing is dropped silently. Time-sampled prims are read at their earliest sample and flagged ``animated``. Conservative on
purpose: a prim with ``visibility = invisible``, ``purpose`` ``guide`` or ``proxy``, or a point instancer's prototype is
scenery too. Nothing here decimates or draws; a camera that images scenery reads these."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.scene._camera import display_authors, read_display
from mime.scene._derived import DERIVED_FROM
from mime.scene._pxr import Usd, UsdGeom


@dataclass(frozen=True, eq=False)
class SceneryGprim:
    """An unowned imageable prim: its prim ``path``, its USD ``type_name`` and the stage's display ``colour`` and
    ``opacity`` for it (``None`` where no prim on its path authors one) with the prim each came from (``colour_from``,
    ``opacity_from``) and ``reduced``, a line for each display primvar that was not constant and was reduced to one
    value. Found and named, not read: a camera refuses it by name until it can image it."""

    path: str
    type_name: str
    colour: tuple[float, float, float] | None = None
    opacity: float | None = None
    colour_from: str | None = None
    opacity_from: str | None = None
    reduced: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False, kw_only=True)
class SceneryMesh(SceneryGprim):
    """A scenery ``Mesh``, read: its ``points`` (``(n, 3)``, metres, world frame) and ``triangles`` (``(m, 3)`` indices),
    and whether it is ``animated`` (time-sampled points or placement: read at its earliest sample). Whether the stage had
    decimated it is not recorded in a layer: that is unknown here, not zero."""

    points: np.ndarray
    triangles: np.ndarray
    animated: bool = False


def _triangles(counts: Sequence[int], indices: Sequence[int], flip: bool) -> np.ndarray:
    """Every face fanned from its first corner (a face of fewer than three corners has no area and is dropped)."""
    found: list[tuple[int, int, int]] = []
    start = 0
    for count in counts:
        face = indices[start:start + count]
        start += count
        for k in range(1, count - 1):
            found.append((face[0], face[k + 1], face[k]) if flip else (face[0], face[k], face[k + 1]))
    return np.asarray(found, dtype=np.int64).reshape(-1, 3)


def _display(prim: Any) -> dict[str, Any]:
    held = read_display(prim)
    colour_from, opacity_from = display_authors(prim)
    return {"colour": held.colour, "opacity": held.opacity, "colour_from": colour_from, "opacity_from": opacity_from,
            "reduced": held.reduced}


def read_mesh(prim: Any, meters_per_unit: float) -> SceneryMesh:
    """The mesh on ``prim`` in the world, in metres, with its display primvars."""
    mesh = UsdGeom.Mesh(prim)
    time = Usd.TimeCode.EarliestTime()
    points = np.asarray(mesh.GetPointsAttr().Get(time) or [], dtype=np.float64).reshape(-1, 3)
    xformable = UsdGeom.Xformable(prim)
    world = np.array(xformable.ComputeLocalToWorldTransform(time)).T
    points = (points @ world[:3, :3].T + world[:3, 3]) * meters_per_unit
    flip = (str(mesh.GetOrientationAttr().Get()) == "leftHanded") != (float(np.linalg.det(world[:3, :3])) < 0.0)
    counts = list(mesh.GetFaceVertexCountsAttr().Get(time) or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get(time) or [])
    animated = (mesh.GetPointsAttr().GetNumTimeSamples() > 0 or mesh.GetFaceVertexCountsAttr().GetNumTimeSamples() > 0
                or mesh.GetFaceVertexIndicesAttr().GetNumTimeSamples() > 0 or bool(xformable.TransformMightBeTimeVarying()))
    return SceneryMesh(str(prim.GetPath()), "Mesh", points=points, triangles=_triangles(counts, indices, flip),
                       animated=animated, **_display(prim))


def read_gprim(prim: Any) -> SceneryGprim:
    """The unread gprim on ``prim``: its path, type and display primvars."""
    return SceneryGprim(str(prim.GetPath()), str(prim.GetTypeName()), **_display(prim))


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def scenery(stage: Any, owned: Sequence[str], meters_per_unit: float) -> tuple[SceneryGprim, ...]:
    """The scenery of ``stage``, by prim path: every ``Gprim`` below none of the ``owned`` prims and not derived, a
    ``SceneryMesh`` for a ``Mesh`` and a ``SceneryGprim`` for any other."""
    found: list[SceneryGprim] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        path = str(prim.GetPath())
        derived = prim.GetAttribute(DERIVED_FROM)
        if any(_within(path, root) for root in owned) or (derived and derived.HasAuthoredValue()):
            continue
        found.append(read_mesh(prim, meters_per_unit) if prim.IsA(UsdGeom.Mesh) else read_gprim(prim))
    return tuple(sorted(found, key=lambda held: held.path))


__all__ = ["SceneryGprim", "SceneryMesh", "read_gprim", "read_mesh", "scenery"]
