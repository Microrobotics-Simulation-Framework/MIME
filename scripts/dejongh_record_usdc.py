#!/usr/bin/env python3
"""USDC recording from saved trajectory JSON files.

Reads smoke trajectory JSON (from dejongh_dynamic_simulation.py) and
produces a time-sampled .usdc animation playable in usdview / MICROBOTICA.

Usage:
    python scripts/dejongh_record_usdc.py FL-9
    python scripts/dejongh_record_usdc.py FL-3
"""
import os, sys, json
import numpy as np

# Force CPU (for mesh generation via JAX)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
from mime.nodes.environment.stokeslet.dejongh_geometry import (
    dejongh_fl_mesh, dejongh_fw_mesh, FL_TABLE, FW_TABLE,
)
from pxr import Usd, UsdGeom, Gf, Sdf

DATA_DIR = Path(__file__).parent.parent / "data" / "dejongh_benchmark"
REC_DIR = DATA_DIR / "recordings"


def build_mesh(design_name):
    """Generate UMR mesh for visualization (low-res)."""
    group, num = design_name.split("-")
    num = int(num)
    if group == "FL":
        m = dejongh_fl_mesh(num, n_theta=30, n_zeta=60)
    else:
        m = dejongh_fw_mesh(num, n_theta=30, n_zeta=60)
    # Build vertices + triangles
    # Re-generate the (θ, ζ) grid to produce triangle indices
    pts = np.array(m.points)
    # We need to regenerate as indexed mesh — use parametric grid directly
    from mime.nodes.environment.stokeslet.dejongh_geometry import R_CYL_DEFAULT, EPSILON_DEFAULT, N_STARTS_DEFAULT, FL_L_UMR
    if group == "FL":
        nu = FL_TABLE[num]["nu"]
        L = FL_L_UMR
    else:
        nu = FW_TABLE[num]["nu"]
        L = FW_TABLE[num]["L_UMR"]
    R_cyl = R_CYL_DEFAULT
    eps = EPSILON_DEFAULT
    N = N_STARTS_DEFAULT

    n_theta, n_zeta = 30, 60
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0, L, n_zeta)
    TH, ZE = np.meshgrid(theta, zeta, indexing='ij')
    rho = R_cyl * (1 + eps * np.sin(N * TH))
    alpha = nu * ZE / R_cyl + TH
    X = rho * np.cos(alpha)
    Y = rho * np.sin(alpha)
    Z = ZE
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # (n_theta × n_zeta, 3)

    # Center along z (shift to body center at origin)
    verts[:, 2] -= L / 2

    # Build triangle indices
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

    # Convert to mm for USD (USD meters_per_unit=1.0 → pass meters, but we'll scale)
    return verts * 1e-3, tris  # to meters (sim is in meters)


def record_usdc(design_name, vessel_name='1/4"', scenario_tag="scenarioA"):
    """Produce .usdc from trajectory JSON."""
    tag = f"{design_name}_{vessel_name.replace(chr(34), '').replace('/', '_')}"
    traj_path = REC_DIR / f"{scenario_tag}_{tag}_trajectory.json"
    if not traj_path.exists():
        # fallback to smoke tag
        traj_path = REC_DIR / f"smoke_{tag}_trajectory.json"
    if not traj_path.exists():
        print(f"Trajectory not found: {traj_path}")
        return False

    data = json.load(open(traj_path))
    meta = data["meta"]
    trajectory = data["trajectory"]
    print(f"Loaded {len(trajectory)} frames from {traj_path}")

    # Build UMR mesh
    verts_m, tris = build_mesh(design_name)
    print(f"Generated mesh: {len(verts_m)} verts, {len(tris)} triangles")

    # Vessel radius
    R_ves_m = meta["R_ves_mm"] * 1e-3

    # Create USD stage
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    # ── Robot mesh ────────────────────────────────────────────
    robot = UsdGeom.Mesh.Define(stage, "/World/Robot")
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts_m]
    robot.GetPointsAttr().Set(points)
    face_counts = [3] * len(tris)
    face_indices = [int(i) for tri in tris for i in tri]
    robot.GetFaceVertexCountsAttr().Set(face_counts)
    robot.GetFaceVertexIndicesAttr().Set(face_indices)
    robot.GetDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.65, 0.3)])  # copper
    robot.GetDoubleSidedAttr().Set(True)

    xformable = UsdGeom.Xformable(robot.GetPrim())
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()

    # ── Vessel (transparent cylinder) ─────────────────────────
    # Length along z: span from first to last frame z
    all_z = [t["position_mm"][2] for t in trajectory]
    z_span = (max(all_z) - min(all_z)) * 1e-3 + 0.02  # meters + padding
    vessel = UsdGeom.Cylinder.Define(stage, "/World/Vessel")
    vessel.GetRadiusAttr().Set(R_ves_m)
    vessel.GetHeightAttr().Set(z_span)
    vessel.GetAxisAttr().Set("Z")
    vessel.GetDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.85, 0.95)])  # light blue
    vessel.GetDoubleSidedAttr().Set(True)
    vessel_prim = vessel.GetPrim()
    vessel_prim.CreateAttribute("primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray).Set([0.2])
    # Position vessel centered around mid-z of trajectory
    z_mid = (max(all_z) + min(all_z)) / 2.0 * 1e-3
    vessel_xf = UsdGeom.Xformable(vessel.GetPrim())
    vessel_xf.ClearXformOpOrder()
    vessel_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_mid))

    # ── Time sampling ────────────────────────────────────────
    # Sample at 20 fps for playback, from trajectory (log_interval=100 steps = 50 ms = 20 fps)
    fps = 20
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(trajectory) - 1)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)

    for frame_idx, t in enumerate(trajectory):
        tc = Usd.TimeCode(frame_idx)
        pos_m = np.array(t["position_mm"]) * 1e-3  # mm → m
        q = np.array(t["orientation"])  # [w, x, y, z]

        translate_op.Set(Gf.Vec3d(float(pos_m[0]), float(pos_m[1]), float(pos_m[2])), tc)
        orient_op.Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])), tc)

    # Metadata
    stage.GetRootLayer().customLayerData = {
        "mime:design": design_name,
        "mime:vessel": vessel_name,
        "mime:n_frames": len(trajectory),
        "mime:fps": fps,
        "mime:duration_s": meta["dt"] * data["summary"]["n_steps"],
        "mime:equilibrium_v_z_mm_s": data["summary"]["equilibrium_v_z_mm_s"],
        "mime:model": "MLPResistanceNode (Cholesky MLP surrogate)",
    }

    out_path = REC_DIR / f"{scenario_tag}_{design_name}.usdc"
    stage.GetRootLayer().Export(str(out_path))
    print(f"Saved: {out_path}")
    return True


def main():
    if len(sys.argv) < 2:
        record_usdc("FL-9", scenario_tag="scenarioA")
        record_usdc("FL-3", scenario_tag="scenarioA")
    elif sys.argv[1] == "B":
        record_usdc("FL-9", scenario_tag="scenarioB")
    else:
        tag = sys.argv[2] if len(sys.argv) > 2 else "scenarioA"
        record_usdc(sys.argv[1], scenario_tag=tag)


if __name__ == "__main__":
    main()
