#!/usr/bin/env python3
"""Bake world.usda for the de Jongh confined-swimming experiment.

One-shot generator. Produces a USD stage with:
  /World/Actors/UMR        — helical microrobot mesh (FL-9 by default)
  /World/Environment/Vessel — cylindrical silicone tube (1/4" inner Ø)
  /World/Camera             — wide framing of the swim region

Run once from the repo root after editing the parametric mesh; the
runner consumes the static .usda thereafter. Re-run if DESIGN_NAME
or vessel diameter changes.

    .venv/bin/python experiments/dejongh_confined/scene/_bake_world.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, Sdf, Gf, Vt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mime.nodes.environment.stokeslet.dejongh_geometry import (
    R_CYL_DEFAULT, EPSILON_DEFAULT, N_STARTS_DEFAULT, FL_L_UMR, FL_TABLE,
)


# ── Mesh generation (mirrors scripts/dejongh_record_usdc.py:build_mesh) ──
def build_fl_mesh(fl_n: int = 9, n_theta: int = 30, n_zeta: int = 60):
    nu = FL_TABLE[fl_n]["nu"]
    L = FL_L_UMR
    R_cyl = R_CYL_DEFAULT
    eps = EPSILON_DEFAULT
    N = N_STARTS_DEFAULT

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0, L, n_zeta)
    TH, ZE = np.meshgrid(theta, zeta, indexing='ij')
    rho = R_cyl * (1 + eps * np.sin(N * TH))
    alpha = nu * ZE / R_cyl + TH
    X = rho * np.cos(alpha)
    Y = rho * np.sin(alpha)
    Z = ZE
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    verts[:, 2] -= L / 2  # centre along z

    tris = []
    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        for j in range(n_zeta - 1):
            v00 = i * n_zeta + j
            v10 = i_next * n_zeta + j
            v11 = i_next * n_zeta + (j + 1)
            v01 = i * n_zeta + (j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    tris = np.array(tris, dtype=np.int32)

    # Convert mm → m to match the runner's SI state
    return verts * 1e-3, tris


def main():
    out_path = Path(__file__).resolve().parent / "world.usda"
    R_VES_MM = 3.175  # 1/4" inner-radius vessel
    VESSEL_LENGTH_M = 0.10  # 100 mm — long enough for several seconds of travel

    verts_m, tris = build_fl_mesh(fl_n=9)
    print(f"UMR mesh: {len(verts_m)} verts, {len(tris)} triangles")

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    # /World/Actors
    UsdGeom.Scope.Define(stage, "/World/Actors")
    umr = UsdGeom.Mesh.Define(stage, "/World/Actors/UMR")
    umr.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts_m.astype(np.float32)))
    umr.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    umr.CreateFaceVertexIndicesAttr(Vt.IntArray(tris.ravel().tolist()))
    umr.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    umr_xform = UsdGeom.Xformable(umr)
    umr_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    umr_xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    umr.GetPrim().SetCustomDataByKey("mimeNodeName", "body")

    # /World/Environment — cylindrical vessel along z
    UsdGeom.Scope.Define(stage, "/World/Environment")
    vessel = UsdGeom.Cylinder.Define(stage, "/World/Environment/Vessel")
    vessel.CreateRadiusAttr(R_VES_MM * 1e-3)
    vessel.CreateHeightAttr(VESSEL_LENGTH_M)
    vessel.CreateAxisAttr(UsdGeom.Tokens.z)
    # Make the vessel translucent so the robot is visible inside.
    vessel.GetPrim().CreateAttribute(
        "primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray
    ).Set([0.18])
    vessel.GetPrim().CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    ).Set([Gf.Vec3f(0.55, 0.65, 0.78)])
    vessel.GetPrim().SetCustomDataByKey("mimeRole", "physics_boundary")

    # /World/Camera — angled view down the vessel
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam_xform = UsdGeom.Xformable(cam)
    cam_xform.AddTranslateOp().Set(Gf.Vec3d(0.012, 0.006, 0.04))
    cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(-15, 25, 0))
    cam.CreateFocalLengthAttr(35.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1.0))

    stage.GetRootLayer().Save()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
