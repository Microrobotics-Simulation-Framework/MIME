#!/usr/bin/env python3
"""Render the EMERGENT (inertial) schwarz_vessel_helix body to a .usdc, with the
EP-1 drive ramp + step-out perturbation. Self-contained (does not depend on the
shared schwarz_record_usdc.py, which is owned by the MICROROBOTICA recorder/viewer
work). Authors the clip at a fixed playback fps (default 30) so it is watchable and
decoupled from the sim dt.

Usage (GPU):
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
    python scripts/svh_render_emergent.py --sim-s 2.0 --dt 1e-4 --sample-every 20 \
      --drive-hz-start 5 --drive-hz-end 45 --drive-ramp-s 1.8 \
      --perturb-torque-nm 1e-7 --perturb-hz 13 \
      --out experiments/schwarz_vessel_helix/output/recording_emergent_ep1.usdc
"""
from __future__ import annotations
import argparse, importlib.util, os, sys, time
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
import numpy as np
from pxr import Usd, UsdGeom, Gf

EXP = REPO / "experiments" / "schwarz_vessel_helix"
WORLD_USDA = EXP / "scene" / "world.usda"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def _load_params():
    ns = {}
    with open(EXP / "physics" / "params.py") as fh:
        exec(fh.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}


def _tag_getters():
    g = {}
    for i in range(6):
        g[f"arm_link_{i}"] = (lambda s, i=i: (
            np.asarray(s["arm"]["link_poses_world"][i][:3]),
            np.asarray(s["arm"]["link_poses_world"][i][3:7])))
    g["motor_rotor"] = lambda s: (np.asarray(s["motor"]["rotor_pose_world"][:3]),
                                   np.asarray(s["motor"]["rotor_pose_world"][3:7]))
    g["magnet"] = lambda s: (np.asarray(s["motor"]["rotor_pose_world"][:3]),
                             np.asarray(s["motor"]["rotor_pose_world"][3:7]))
    g["body"] = lambda s: (np.asarray(s["body"]["position"]),
                           np.asarray(s["body"]["orientation"]))
    return g


def _discover_prims():
    stage = Usd.Stage.Open(str(WORLD_USDA)); out = {}
    for prim in stage.Traverse():
        tag = prim.GetCustomData().get("mimeNodeName")
        if tag:
            out[tag] = prim.GetPath().pathString
    return out


def run(args):
    params = _load_params()
    params.update({
        "BODY_MODEL": "inertial", "DT": args.dt,
        "DRIVE_HZ_START": args.drive_hz_start, "DRIVE_HZ_END": args.drive_hz_end,
        "DRIVE_RAMP_S": args.drive_ramp_s,
        "PERTURB_TORQUE_NM": args.perturb_torque_nm, "PERTURB_HZ": args.perturb_hz,
    })
    if args.mag_dipole is not None:
        params["MAG_DIPOLE"] = args.mag_dipole   # field calibration (1.2 mT → ~40.5)
    print(f"  overrides: BODY_MODEL=inertial DT={args.dt} ramp "
          f"{args.drive_hz_start}->{args.drive_hz_end}Hz/{args.drive_ramp_s}s "
          f"perturb={args.perturb_torque_nm}N·m@{args.perturb_hz}Hz", flush=True)
    setup = _load_module("svh_setup", EXP / "physics" / "setup.py")
    ctrl = _load_module("svh_ctrl", EXP / "control" / "controller.py")
    ctrl._controller_instance = None
    print("Building graph...", flush=True)
    gm = setup.build_graph(params)
    n_steps = int(args.sim_s / args.dt)
    getters = _tag_getters(); tag2path = _discover_prims()
    specs = [(tag2path[t], getters[t]) for t in getters if t in tag2path]
    print(f"  {n_steps} steps, sample every {args.sample_every} → "
          f"{n_steps//args.sample_every} frames; prims {[p for p,_ in specs]}", flush=True)
    prev = {n: gm.get_node_state(n) for n in gm._nodes}
    frames = []; t0 = time.perf_counter(); last = t0
    for step in range(n_steps):
        ext = ctrl.get_external_inputs(params, step, state=prev)
        gm.step(ext)
        prev = {n: gm.get_node_state(n) for n in gm._nodes}
        if step % args.sample_every == 0:
            frames.append({p: tuple(np.copy(x) for x in g(prev)) for p, g in specs})
        if not np.all(np.isfinite(np.asarray(prev["body"]["position"]))):
            print(f"  NaN at step {step} — aborting", flush=True); break
        now = time.perf_counter()
        if now - last > 10.0:
            print(f"  step {step+1}/{n_steps} ({now-t0:.0f}s wall, {(step+1)*args.dt:.2f}s sim)",
                  flush=True); last = now
    print(f"  total {time.perf_counter()-t0:.0f}s wall", flush=True)
    return frames


def write_usdc(frames, out, fps):
    out = Path(out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.OverridePrim("/World").GetReferences().AddReference(
        os.path.relpath(WORLD_USDA.resolve(), out.parent))
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(len(frames) - 1)
    stage.SetTimeCodesPerSecond(fps); stage.SetFramesPerSecond(fps)
    op = {}
    for prim_path in (frames[0] if frames else {}):
        xf = UsdGeom.Xformable(stage.OverridePrim(prim_path)); xf.ClearXformOpOrder()
        op[prim_path] = (xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, ""),
                         xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, ""))
    for fi, frame in enumerate(frames):
        tc = Usd.TimeCode(fi)
        for prim_path, (t, q) in frame.items():
            tr, oo = op[prim_path]
            tr.Set(Gf.Vec3d(*[float(v) for v in t]), tc)
            oo.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])), tc)
    stage.GetRootLayer().Save()
    flat = Usd.Stage.Open(str(out)).Flatten()
    def _strip(s, root):
        if not root: s.ClearReferenceList(); s.ClearPayloadList()
        for c in s.nameChildren: _strip(c, False)
    _strip(flat.GetPrimAtPath("/"), True); flat.Export(str(out))
    print(f"  wrote {len(frames)} frames @ {fps}fps = {len(frames)/fps:.1f}s → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-s", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--drive-hz-start", type=float, default=5.0)
    ap.add_argument("--drive-hz-end", type=float, default=45.0)
    ap.add_argument("--drive-ramp-s", type=float, default=1.8)
    ap.add_argument("--perturb-torque-nm", type=float, default=1e-7)
    ap.add_argument("--perturb-hz", type=float, default=13.0)
    ap.add_argument("--mag-dipole", type=float, default=None,
                    help="override MAG_DIPOLE [A·m²] (field calibration; ~40.5 → 1.2 mT)")
    ap.add_argument("--playback-fps", type=float, default=30.0)
    ap.add_argument("--out", type=str,
                    default=str(EXP / "output" / "recording_emergent_ep1.usdc"))
    args = ap.parse_args()
    frames = run(args)
    if frames:
        write_usdc(frames, args.out, args.playback_fps)
    print("Done.")


if __name__ == "__main__":
    main()
