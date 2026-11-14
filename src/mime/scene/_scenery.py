"""The meshes a composed stage holds that no thing owns: the scenery an in-loop camera is to image (a vessel's STL in a
referenced asset layer). A ``Mesh`` prim is scenery unless a thing owns it: it lies below a prim a thing is bound to (a
body's, an apparatus's), or it is a kind's derived prim (``mime:derived_from``, imaged by its kind). Each is read as the
stage holds it: its points in metres in the world (the transforms down to it applied), its triangles (polygons fanned,
wound counter-clockwise seen from outside whatever the prim's orientation and its transform's handedness), and its
display colour and opacity with the prim that authored each (``FindPrimvarWithInheritance``: a mesh takes the colour of
its visual prim). Nothing here decimates or draws; a camera that images scenery reads these."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.scene._camera import display_authors, display_primvars
from mime.scene._derived import DERIVED_FROM
from mime.scene._pxr import Usd, UsdGeom


@dataclass(frozen=True, eq=False)
class SceneryMesh:
    """A scenery mesh: its prim ``path``, its ``points`` (``(n, 3)``, metres, world frame) and ``triangles`` (``(m, 3)``
    indices), and the stage's display ``colour`` and ``opacity`` for it (``None`` where no prim on its path authors one)
    with the prim each came from (``colour_from``, ``opacity_from``). Whether the stage had decimated it is not
    recorded in a layer: that is unknown here, not zero."""

    path: str
    points: np.ndarray
    triangles: np.ndarray
    colour: tuple[float, float, float] | None = None
    opacity: float | None = None
    colour_from: str | None = None
    opacity_from: str | None = None


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


def read_mesh(prim: Any, meters_per_unit: float) -> SceneryMesh:
    """The mesh on ``prim`` in the world, in metres, with its display primvars."""
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or [], dtype=np.float64).reshape(-1, 3)
    world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    points = (points @ world[:3, :3].T + world[:3, 3]) * meters_per_unit
    flip = (str(mesh.GetOrientationAttr().Get()) == "leftHanded") != (float(np.linalg.det(world[:3, :3])) < 0.0)
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    colour, opacity = display_primvars(prim)
    colour_from, opacity_from = display_authors(prim)
    return SceneryMesh(str(prim.GetPath()), points, _triangles(counts, indices, flip), colour, opacity, colour_from,
                       opacity_from)


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def scenery(stage: Any, owned: Sequence[str], meters_per_unit: float) -> tuple[SceneryMesh, ...]:
    """The scenery of ``stage``, by prim path: every ``Mesh`` prim below none of the ``owned`` prims and not derived."""
    found: list[SceneryMesh] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        derived = prim.GetAttribute(DERIVED_FROM)
        if any(_within(path, root) for root in owned) or (derived and derived.HasAuthoredValue()):
            continue
        found.append(read_mesh(prim, meters_per_unit))
    return tuple(sorted(found, key=lambda mesh: mesh.path))


__all__ = ["SceneryMesh", "read_mesh", "scenery"]
