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

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
from stl import mesh as stl_mesh


SCENE_DIR = Path(__file__).resolve().parent
EXP_DIR = SCENE_DIR.parent
MESH_DIR = EXP_DIR / "assets" / "meshes" / "ar4_mk5"
URDF_PATH = EXP_DIR / "assets" / "ar4.urdf"
# Binary USD: an order of magnitude smaller than ascii .usda for dense
# triangle meshes. world.usda references the prims inside this layer.
OUT_LAYER = EXP_DIR / "assets" / "ar4_meshes.usdc"


# Map from baked-prim name to the URDF visual sub-link names we merge
# under it. Each sub-link has its own ``<visual><origin xyz=... rpy=.../>``
# in the URDF — the bake applies that transform to each sub-mesh's
# vertices before concatenating, so the resulting prim's vertex
# coordinates are expressed in the parent link's frame. Without this
# the rendered arm appears "disconnected" — sub-meshes drift by 1-10
# cm from their intended attachment points.
LINK_VISUAL_SUBLINKS = {
    "Base":   ["base_link_aluminum", "base_link_enclosure", "base_link_motor"],
    "Link_1": ["link_1_aluminum", "link_1_motor"],
    "Link_2": ["link_2_aluminum", "link_2_motor", "link_2_cover", "link_2_logo"],
    "Link_3": ["link_3_aluminum", "link_3_motor"],
    "Link_4": ["link_4_aluminum", "link_4_motor", "link_4_cover", "link_4_logo"],
    "Link_5": ["link_5_aluminum", "link_5_motor"],
    "Link_6": ["link_6_aluminum"],
}


def _rpy_to_matrix(rpy):
    """URDF roll/pitch/yaw → 3x3 rotation. URDF convention is
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` applied as ``v' = R @ v``."""
    r, p, y = rpy
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _read_visual_origins(urdf_path):
    """Walk the URDF and return ``{ link_name: (xyz, rpy, mesh_filename) }``
    for every link whose ``<visual>`` references a mesh."""
    out = {}
    tree = ET.parse(urdf_path)
    for link in tree.getroot().findall("link"):
        name = link.get("name")
        visual = link.find("visual")
        if visual is None:
            continue
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        origin = visual.find("origin")
        xyz = np.array([0.0, 0.0, 0.0])
        rpy = np.array([0.0, 0.0, 0.0])
        if origin is not None:
            if origin.get("xyz"):
                xyz = np.array([float(x) for x in origin.get("xyz").split()])
            if origin.get("rpy"):
                rpy = np.array([float(x) for x in origin.get("rpy").split()])
        mesh_file = Path(mesh.get("filename", "")).name
        out[name] = (xyz, rpy, mesh_file)
    return out


def _load_and_transform_sublinks(sublinks, visual_origins):
    """For each sub-link, load its STL and apply the URDF visual
    ``<origin>`` transform to every vertex before concatenating —
    so the resulting points are expressed in the parent link's frame.
    """
    all_pts = []
    for sublink_name in sublinks:
        if sublink_name not in visual_origins:
            print(f"  WARN: no visual origin for {sublink_name}, skipping")
            continue
        xyz, rpy, mesh_file = visual_origins[sublink_name]
        path = MESH_DIR / mesh_file
        if not path.exists():
            print(f"  WARN: missing {path.name}, skipping")
            continue
        m = stl_mesh.Mesh.from_file(str(path))
        pts = m.vectors.reshape(-1, 3).astype(np.float64)
        # URDF visual origin: v' = R · v + t
        R = _rpy_to_matrix(rpy)
        pts_xformed = pts @ R.T + xyz
        all_pts.append(pts_xformed)
    if not all_pts:
        return np.empty((0, 3), dtype=np.float32), [], []
    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    n_tri = pts.shape[0] // 3
    counts = [3] * n_tri
    indices = list(range(n_tri * 3))
    return pts, counts, indices


def main() -> int:
    print(f"Baking AR4 STLs from {MESH_DIR}")
    print(f"     URDF source:    {URDF_PATH}")
    print(f"     into {OUT_LAYER}")
    visual_origins = _read_visual_origins(URDF_PATH)
    print(f"     URDF visuals:   {len(visual_origins)} entries with non-trivial origins")

    layer = Sdf.Layer.CreateNew(str(OUT_LAYER))
    stage = Usd.Stage.Open(layer)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    root = UsdGeom.Xform.Define(stage, "/AR4")

    for prim_name, sublinks in LINK_VISUAL_SUBLINKS.items():
        print(f"  - {prim_name}: {len(sublinks)} sub-link(s)")
        pts, counts, indices = _load_and_transform_sublinks(
            sublinks, visual_origins,
        )
        if pts.shape[0] == 0:
            continue
        mesh_prim = UsdGeom.Mesh.Define(stage, f"/AR4/{prim_name}")
        mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        # Hydra computes per-face normals when none are authored.

    stage.SetDefaultPrim(root.GetPrim())
    layer.Save()
    print(f"  wrote {OUT_LAYER.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
