#!/usr/bin/env python3
"""MOD-2 / EP-1 second half — single-RPM (gradient) vs dual-RPM (gradient-cancelled)
step-out at matched |B|. Headless coupled two-scale graph, emergent inertial body,
no-arm. Ramps the drive, tracks body spin + wobble β, detects self-trip, saves a trace
.npz AND a .usdc recording (body + driving magnet) for qualitative confirmation. The
single-vs-dual step-out / β(ω) difference is the gradient's contribution (D1).

Usage (GPU):
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
    python scripts/mod2_stepout_compare.py --magnetic-model dual --hz0 5 --hz1 250 \
      --ramp-s 3.0 --dt 1e-4 --mag-dipole 40.5 --out <prefix>
(--out is a path prefix; writes <prefix>.npz and <prefix>.usdc)
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
import numpy as np
import jax.numpy as jnp
from mime.core.quaternion import rotate_vector_inverse
from pxr import Usd, UsdGeom, Gf

WORLD_USDA = REPO / "experiments" / "schwarz_vessel_helix" / "scene" / "world.usda"


def write_usdc(frames, out, fps=30.0):
    """frames: list of {prim_path: (translate3, quat_wxyz)}. Animates body + magnet."""
    out = Path(out); stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.OverridePrim("/World").GetReferences().AddReference(
        os.path.relpath(WORLD_USDA.resolve(), out.parent))
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(len(frames) - 1)
    stage.SetTimeCodesPerSecond(fps); stage.SetFramesPerSecond(fps)
    op = {}
    for pth in (frames[0] if frames else {}):
        xf = UsdGeom.Xformable(stage.OverridePrim(pth)); xf.ClearXformOpOrder()
        op[pth] = (xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, ""),
                   xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, ""))
    for fi, fr in enumerate(frames):
        tc = Usd.TimeCode(fi)
        for pth, (t, q) in fr.items():
            tr, oo = op[pth]
            tr.Set(Gf.Vec3d(*[float(v) for v in t]), tc)
            oo.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])), tc)
    stage.GetRootLayer().Save()
    flat = Usd.Stage.Open(str(out)).Flatten()
    def _strip(s, root):
        if not root: s.ClearReferenceList(); s.ClearPayloadList()
        for c in s.nameChildren: _strip(c, False)
    _strip(flat.GetPrimAtPath("/"), True); flat.Export(str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnetic-model", choices=["single", "dual"], required=True)
    ap.add_argument("--hz0", type=float, default=5.0)
    ap.add_argument("--hz1", type=float, default=250.0)
    ap.add_argument("--ramp-s", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--mag-dipole", type=float, default=40.5)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--hz-step", type=float, default=12.0,
                    help="staircase frequency step [Hz]; drive held constant per level "
                         "(avoids the per-step JIT recompile that a continuous ramp + "
                         "external pose triggers — see notes)")
    ap.add_argument("--hold-steps", type=int, default=1000,
                    help="steps to hold each frequency level (quasi-static, continued state)")
    ap.add_argument("--out", required=True, help="path prefix → <prefix>.npz / .usdc")
    a = ap.parse_args()
    from mime.experiments import schwarz_vessel_helix as S

    p = {"MAGNETIC_MODEL": a.magnetic_model, "INCLUDE_ARM": False,
         "BODY_MODEL": "inertial", "MAG_DIPOLE": a.mag_dipole, "DT": a.dt,
         "SWIM_MODE": "free"}
    print(f"building {a.magnetic_model} graph (no-arm, inertial, MAG_DIPOLE={a.mag_dipole})...",
          flush=True)
    gm = S.build_graph(p)
    ref = S.screw_points(p)
    nodes = set(gm._nodes)
    mkey = "motor" if "motor" in nodes else "motor0"   # driving RPM for the recording
    # Fixed motor pose(s): no-arm, magnet above/below the pipe, spin axis = world-x.
    z = 0.15; q = [0.7071068, 0.0, 0.7071068, 0.0]
    pose0 = jnp.array([0.0, 0.0, z, *q]); pose1 = jnp.array([0.0, 0.0, -z, *q])
    # Build ext directly so the ramped drive is passed as jnp.float32 (TRACED → no
    # per-step recompile; default_external_inputs bakes jnp.asarray(2π·DRIVE_HZ) as a
    # constant which recompiles on every change → 11 s/step).
    def make_ext(omega, prev):
        pts = S.body_world_points(ref, prev)
        e = {"fvm": {"sample_points": pts, "forcing_points": pts}}
        if a.magnetic_model == "dual":
            e["motor0"] = {"commanded_velocity": jnp.float32(omega), "parent_pose_world": pose0}
            e["motor1"] = {"commanded_velocity": jnp.float32(omega), "parent_pose_world": pose1}
        else:
            e["motor"] = {"commanded_velocity": jnp.float32(omega), "parent_pose_world": pose0}
        return e
    levels = np.arange(a.hz0, a.hz1 + 1e-6, a.hz_step)
    prev = {nm: gm.get_node_state(nm) for nm in gm._nodes}
    drv, spn, bet, frames = [], [], [], []
    t0 = time.perf_counter(); last = t0; step = 0; blew = False
    for lvl in levels:
        om = 2.0 * math.pi * float(lvl)      # constant within the level → no recompile
        for k in range(a.hold_steps):
            gm.step(make_ext(om, prev))
            prev = {nm: gm.get_node_state(nm) for nm in gm._nodes}
            if step % a.sample_every == 0:
                q = np.asarray(prev["body"]["orientation"]); w, x, y, zz = q
                beta = math.degrees(math.acos(max(-1.0, min(1.0, 2 * (x * zz + w * y)))))
                wz = float(rotate_vector_inverse(jnp.asarray(q),
                           jnp.asarray(prev["body"]["angular_velocity"]))[2])
                drv.append(float(lvl)); spn.append(abs(wz) / (2 * math.pi)); bet.append(beta)
                rp = np.asarray(prev[mkey]["rotor_pose_world"])
                frames.append({
                    "/World/Actors/UMR": (np.asarray(prev["body"]["position"]), q),
                    "/World/Actors/Magnet/Body": (rp[:3], rp[3:7]),
                    "/World/Actors/Magnet/Rotor": (rp[:3], rp[3:7])})
            if not np.all(np.isfinite(np.asarray(prev["body"]["position"]))):
                print(f"  NaN at step {step} (lvl {lvl:.0f} Hz)", flush=True); blew = True; break
            now = time.perf_counter()
            if now - last > 10.0:
                print(f"  lvl {lvl:.0f}/{a.hz1:.0f} Hz, step {step} ({now-t0:.0f}s)",
                      flush=True); last = now
            step += 1
        if blew: break
    drv, spn, bet = np.array(drv), np.array(spn), np.array(bet)
    np.savez(a.out + ".npz", drive=drv, spin=spn, beta=bet, model=a.magnetic_model)
    print(f"  saved {a.out}.npz", flush=True)
    try:
        write_usdc(frames, a.out + ".usdc")
        print(f"  saved {a.out}.usdc ({len(frames)} frames @ 30fps)", flush=True)
    except Exception as e:
        print(f"  USDC write failed (npz is safe): {type(e).__name__}: {e}", flush=True)
    trip = None; run = 0
    for i in range(len(drv)):
        if drv[i] < 8: continue
        if spn[i] < 0.5 * drv[i]:
            run += 1
            if run > 40: trip = drv[i]; break
        else: run = 0
    print(f"  [{a.magnetic_model}] self-trip: {('~%.0f Hz' % trip) if trip else 'none to %.0f' % a.hz1}"
          f"  max β={bet.max():.0f}°", flush=True)


if __name__ == "__main__":
    main()
