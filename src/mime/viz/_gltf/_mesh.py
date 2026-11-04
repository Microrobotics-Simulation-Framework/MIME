# Derived from usd2gltf 0.3.5 (https://github.com/mikelyndon/usd2gltf), MIT, Copyright (c) 2020 mikelyndon.
# Modified for MIME: typed and vectorised with numpy; the fan triangulation with its handedness and the first-sample
# topology kept; positions kept indexed where every attribute is per vertex, and spread to the faces' corners only where
# one is per face or per corner; normals and displayColor kept, texture coordinates, tangents, geometry subsets and
# material bindings dropped (MIME's stages hold displayColor and displayOpacity primvars, no material).
# SPDX-License-Identifier: MIT
"""A ``UsdGeom.Mesh`` as triangles: its points, its faces fanned into triangles (turned where the mesh is
left-handed, so that glTF's counter-clockwise faces are its front faces), and its normals and colours where it holds
them, at the first time sample: glTF cannot change a mesh's topology over time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.scene._pxr import UsdGeom
from mime.viz._gltf._common import static_value


@dataclass(frozen=True)
class Triangles:
    """A mesh's triangles over its points, with a normal and a colour per point where the mesh holds them."""

    points: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray | None = None
    colours: np.ndarray | None = None


def fanned(counts: np.ndarray, *, left_handed: bool = False) -> np.ndarray:
    """Each face of ``counts`` corners fanned from its first corner into ``counts - 2`` triangles, in the faces' order,
    each a row of three corner positions (into the face-vertex indices), turned where the mesh is left-handed; a face of
    fewer than three corners gives none."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    fans = np.maximum(counts - 2, 0)
    face = np.repeat(np.arange(len(counts)), fans)
    along = np.arange(len(face)) - np.repeat(np.cumsum(fans) - fans, fans)
    a = starts[face]
    b = a + along + 1
    return np.stack([a, b + 1, b] if left_handed else [a, b, b + 1], axis=1).astype(np.int64)


def _primvar(mesh: Any, name: str) -> tuple[np.ndarray | None, str]:
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar(name)
    if not primvar or not primvar.HasAuthoredValue():
        return None, ""
    value = primvar.ComputeFlattened()
    if value is None or len(value) == 0:
        return None, ""
    return np.asarray(value, dtype=np.float64), str(primvar.GetInterpolation())


def triangles_of(mesh: Any) -> Triangles | None:
    """The mesh prim ``mesh``'s triangles at its first time sample, ``None`` where it holds no face."""
    points_value = static_value(mesh.GetPointsAttr())
    counts_value = static_value(mesh.GetFaceVertexCountsAttr())
    indices_value = static_value(mesh.GetFaceVertexIndicesAttr())
    if not points_value or not counts_value or not indices_value:
        return None
    points = np.asarray(points_value, dtype=np.float64).reshape(-1, 3)
    counts = np.asarray(counts_value, dtype=np.int64)
    indices = np.asarray(indices_value, dtype=np.int64)
    left = str(mesh.GetOrientationAttr().Get()) == str(UsdGeom.Tokens.leftHanded)
    corners = fanned(counts, left_handed=left)
    if not len(corners):
        return None
    faces = np.repeat(np.arange(len(counts)), np.maximum(counts - 2, 0))
    normals_value = static_value(mesh.GetNormalsAttr()) if mesh.GetNormalsAttr().HasAuthoredValue() else None
    normals = np.asarray(normals_value, dtype=np.float64).reshape(-1, 3) if normals_value else None
    normal_rate = str(mesh.GetNormalsInterpolation()) if normals is not None else ""
    colours, colour_rate = _primvar(mesh, "displayColor")
    if colours is not None and colour_rate == "constant":
        colours, colour_rate = None, ""
    per_vertex = {"", "vertex", "varying"}
    if normal_rate in per_vertex and colour_rate in per_vertex:
        held_normals = normals if normals is not None and len(normals) == len(points) else None
        held_colours = colours if colours is not None and len(colours) == len(points) else None
        return Triangles(points, indices[corners], held_normals, held_colours)
    # An attribute per face or per corner: every triangle's corners get points of their own.
    flat = corners.reshape(-1)

    def spread(values: np.ndarray | None, rate: str) -> np.ndarray | None:
        if values is None:
            return None
        if rate in ("vertex", "varying"):
            return values[indices[flat]]
        if rate == "faceVarying":
            return values[flat]
        if rate == "uniform":
            return values[np.repeat(faces, 3)]
        return None

    return Triangles(points[indices[flat]], np.arange(len(flat), dtype=np.int64).reshape(-1, 3),
                     spread(normals, normal_rate), spread(colours, colour_rate))


__all__ = ["Triangles", "fanned", "triangles_of"]
