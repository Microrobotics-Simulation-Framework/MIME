#!/usr/bin/env python3
"""Decimate over-tessellated robot meshes in a USD file.

The AR4 arm meshes come from CAD/STL at full tessellation (~1M triangles
total). At that density most triangles are sub-pixel in the viewport, which
produces geometric-aliasing shimmer during camera motion. This reduces each
mesh to at most TARGET_FACES triangles via VTK quadric decimation — plenty of
detail for a clean render, with no sub-pixel shimmer — and renders far faster.

Usage:  python scripts/decimate_ar4_meshes.py FILE.usdc [FILE2.usdc ...]
"""
import sys

import numpy as np
import vtk
from vtk.util import numpy_support
from pxr import Gf, Usd, UsdGeom, Vt

TARGET_FACES = 20000


def decimate(points, tris, target_faces):
    """points (N,3) float, tris (M,3) int -> (new_points, new_tris)."""
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(points, np.float32), deep=1))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vpts)

    cells = np.empty((tris.shape[0], 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = tris
    ca = vtk.vtkCellArray()
    ca.SetCells(tris.shape[0], numpy_support.numpy_to_vtkIdTypeArray(
        np.ascontiguousarray(cells.ravel()), deep=1))
    poly.SetPolys(ca)

    # CAD/STL meshes are triangle soup (no shared vertices); quadric
    # decimation needs real edge connectivity, so weld coincident points.
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.SetTolerance(1e-6)

    dec = vtk.vtkQuadricDecimation()
    dec.SetInputConnection(clean.GetOutputPort())
    dec.SetTargetReduction(1.0 - target_faces / float(tris.shape[0]))
    dec.Update()
    out = dec.GetOutput()

    new_pts = numpy_support.vtk_to_numpy(out.GetPoints().GetData()).reshape(-1, 3)
    new_tris = numpy_support.vtk_to_numpy(
        out.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return new_pts, new_tris


def decimate_stage(path):
    print(f"--- {path} ---")
    stage = Usd.Stage.Open(path)
    before = after = 0
    changed = False
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        pts = mesh.GetPointsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not fvc or not pts or not idx:
            continue
        nfaces = len(fvc)
        before += nfaces
        if nfaces <= TARGET_FACES:
            after += nfaces
            print(f"  keep {prim.GetPath()}  ({nfaces} faces)")
            continue
        if mesh.GetPointsAttr().GetTimeSamples():
            after += nfaces
            print(f"  skip {prim.GetPath()}  (animated points)")
            continue
        if any(c != 3 for c in fvc):
            after += nfaces
            print(f"  skip {prim.GetPath()}  (not a pure-triangle mesh)")
            continue

        new_pts, new_tris = decimate(
            np.array(pts, np.float32),
            np.array(idx, np.int64).reshape(-1, 3),
            TARGET_FACES)
        if new_tris.shape[0] == 0:
            after += nfaces
            print(f"  skip {prim.GetPath()}  (decimation produced no faces)")
            continue

        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_pts.astype(np.float32)))
        mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray.FromNumpy(np.full(new_tris.shape[0], 3, np.int32)))
        mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(np.ascontiguousarray(new_tris.ravel(), np.int32)))
        # Topology changed: drop stale normals (Storm recomputes them) and pin
        # subdivisionScheme to 'none' so it renders as a plain polygon mesh.
        mesh.GetNormalsAttr().Clear()
        prim.RemoveProperty("primvars:normals")
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        mn, mx = new_pts.min(0), new_pts.max(0)
        mesh.GetExtentAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*map(float, mn)), Gf.Vec3f(*map(float, mx))]))
        after += new_tris.shape[0]
        changed = True
        print(f"  {prim.GetPath()}: {nfaces} -> {new_tris.shape[0]} faces")

    if changed:
        stage.Save()
        print(f"  saved.  total faces {before} -> {after}")
    else:
        print("  nothing to do.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        decimate_stage(arg)
