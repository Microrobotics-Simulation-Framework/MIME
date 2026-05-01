#!/usr/bin/env python3
"""Bake AR4 STL meshes into a USD layer.

Mirrors ``dejongh_confined/scene/_bake_world.py`` in spirit: reads the
visual STLs vendored under ``../assets/meshes/ar4_mk5/`` and writes a
single ``../assets/ar4_meshes.usda`` containing one ``UsdGeom.Mesh``
prim per visual link. ``world.usda`` then references those prims by
path under each ``Arm/Lk`` Xform, so the renderer sees real arm
geometry bouncing through the link frames the runner emits each tick.

Run once at experiment-author time:

    .venv/bin/python experiments/ar4_helical_drive/scene/_bake_meshes.py

Re-run if the STL set changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
from stl import mesh as stl_mesh


SCENE_DIR = Path(__file__).resolve().parent
EXP_DIR = SCENE_DIR.parent
MESH_DIR = EXP_DIR / "assets" / "meshes" / "ar4_mk5"
# Binary USD: an order of magnitude smaller than ascii .usda for dense
# triangle meshes. world.usda references the prims inside this layer.
OUT_LAYER = EXP_DIR / "assets" / "ar4_meshes.usdc"


# Per-link visual STLs we bake. Each tuple is
# (USD prim name, list of STL files to merge for that prim).
# Sub-meshes (aluminum / motor / cover / logo) get merged so each
# Xform holds one composite prim.
LINK_VISUAL_STLS = {
    "Base":   ["Link_Base_Aluminum.STL", "Link_Base_Enclosure.STL", "Link_Base_Motor.STL"],
    "Link_1": ["Link_1_Aluminum.STL", "Link_1_Motor.STL"],
    "Link_2": ["Link_2_Aluminum.STL", "Link_2_Motor.STL", "Link_2_Cover.STL", "Link_2_Logo.STL"],
    "Link_3": ["Link_3_Aluminum.STL", "Link_3_Motor.STL"],
    "Link_4": ["Link_4_Aluminum.STL", "Link_4_Motor.STL", "Link_4_Cover.STL", "Link_4_Logo.STL"],
    "Link_5": ["Link_5_Aluminum.STL", "Link_5_Motor.STL"],
    "Link_6": ["Link_6_Aluminum.STL"],
}


def _load_stls(filenames):
    """Concatenate triangle data from one or more STL files.

    Returns ``(points (M, 3), face_vertex_counts (F,), face_vertex_indices (3F,))``
    suitable for ``UsdGeom.Mesh``.
    """
    all_pts = []
    for name in filenames:
        path = MESH_DIR / name
        if not path.exists():
            print(f"  WARN: missing {path.name}, skipping")
            continue
        m = stl_mesh.Mesh.from_file(str(path))
        # Each triangle is 3 vertices × 3 coords; flatten and stack.
        all_pts.append(m.vectors.reshape(-1, 3))
    if not all_pts:
        return np.empty((0, 3), dtype=np.float32), [], []
    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    n_tri = pts.shape[0] // 3
    counts = [3] * n_tri
    indices = list(range(n_tri * 3))
    return pts, counts, indices


def main() -> int:
    print(f"Baking AR4 STLs from {MESH_DIR}")
    print(f"           into {OUT_LAYER}")
    layer = Sdf.Layer.CreateNew(str(OUT_LAYER))
    stage = Usd.Stage.Open(layer)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    root = UsdGeom.Xform.Define(stage, "/AR4")

    for prim_name, stls in LINK_VISUAL_STLS.items():
        print(f"  - {prim_name}: {len(stls)} STL(s)")
        pts, counts, indices = _load_stls(stls)
        if pts.shape[0] == 0:
            continue
        mesh_prim = UsdGeom.Mesh.Define(stage, f"/AR4/{prim_name}")
        mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        # Faceted normals: one normal per vertex (= per-triangle since we
        # don't share vertices). This keeps the renderer happy without a
        # subdivision pass.
        # We skip explicit normals — Hydra computes per-face normals when
        # they're missing.

    stage.SetDefaultPrim(root.GetPrim())
    layer.Save()
    print(f"  wrote {OUT_LAYER.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
