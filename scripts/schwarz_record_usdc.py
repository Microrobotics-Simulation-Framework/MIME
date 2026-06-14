#!/usr/bin/env python3
"""Record the schwarz_vessel_helix experiment to a .usdc file (MICROROBOTICA-viewable).

Mirrors scripts/ar4_record_usdc.py, but for the two-scale coupled experiment and
with the animated prims discovered by their ``customData.mimeNodeName`` tag rather
than hard-coded paths (the scene is the ar4 lab, copied). Runs the full graph +
controller for --sim-s seconds, samples each tagged actor's pose every
--sample-every steps, and writes a self-contained .usdc referencing
``scene/world.usda``.

Tags handled: arm_link_0..5 (arm), motor_rotor + magnet (rotor pose), body (UMR).

Usage:
    .venv/bin/python scripts/schwarz_record_usdc.py --sim-s 3.0
    .venv/bin/python scripts/schwarz_record_usdc.py --sim-s 3.0 --out experiments/schwarz_vessel_helix/output/recording.usdc
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from pxr import Usd, UsdGeom, Gf  # noqa: E402

EXP_DIR = REPO_ROOT / "experiments" / "schwarz_vessel_helix"
PARAMS_PATH = EXP_DIR / "physics" / "params.py"
SETUP_PATH = EXP_DIR / "physics" / "setup.py"
CONTROLLER_PATH = EXP_DIR / "control" / "controller.py"
WORLD_USDA = EXP_DIR / "scene" / "world.usda"


# Pose getters keyed by mimeNodeName tag: state dict → (translate xyz, quat wxyz).
def _tag_getters():
    g = {}
    for i in range(6):
        g[f"arm_link_{i}"] = (lambda s, i=i: (
            np.asarray(s["arm"]["link_poses_world"][i][:3]),
            np.asarray(s["arm"]["link_poses_world"][i][3:7])))
    g["motor_rotor"] = lambda s: (
        np.asarray(s["motor"]["rotor_pose_world"][:3]),
        np.asarray(s["motor"]["rotor_pose_world"][3:7]))
    g["magnet"] = lambda s: (
        np.asarray(s["motor"]["rotor_pose_world"][:3]),
        np.asarray(s["motor"]["rotor_pose_world"][3:7]))
    g["body"] = lambda s: (
        np.asarray(s["body"]["position"]),
        np.asarray(s["body"]["orientation"]))
    return g


def _discover_prims(world_usda: Path):
    """{mimeNodeName: prim_path} from the scene's customData tags."""
    stage = Usd.Stage.Open(str(world_usda))
    out = {}
    for prim in stage.Traverse():
        cd = prim.GetCustomData()
        tag = cd.get("mimeNodeName")
        if tag:
            out[tag] = prim.GetPath().pathString
    return out


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _load_params():
    ns = {}
    with open(PARAMS_PATH) as fh:
        exec(fh.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}


def run_simulation(sim_s, sample_every):
    params = _load_params()
    setup = _load_module("svh_setup", SETUP_PATH)
    ctrl = _load_module("svh_ctrl", CONTROLLER_PATH)
    ctrl._controller_instance = None

    print("Building graph (two-scale + arm + off-center)...", flush=True)
    gm = setup.build_graph(params)
    dt = float(params.get("DT", 5e-4))
    n_steps = int(sim_s / dt)
    print(f"  {n_steps} steps × {dt*1e3:.2f} ms = {sim_s:.2f} s, "
          f"sample every {sample_every} → {n_steps//sample_every} frames", flush=True)

    getters = _tag_getters()
    tag2path = _discover_prims(WORLD_USDA)
    specs = [(tag2path[tag], getters[tag]) for tag in getters if tag in tag2path]
    print(f"  animated prims: {[p for p,_ in specs]}", flush=True)

    prev = {n: gm.get_node_state(n) for n in gm._nodes}
    frames = []
    t0 = time.perf_counter(); last = t0
    for step in range(n_steps):
        ext = ctrl.get_external_inputs(params, step, state=prev)
        gm.step(ext)
        prev = {n: gm.get_node_state(n) for n in gm._nodes}
        if step % sample_every == 0:
            frames.append({p: tuple(np.copy(x) for x in g(prev)) for p, g in specs})
        now = time.perf_counter()
        if now - last > 10.0:
            print(f"  step {step+1}/{n_steps} ({now-t0:.0f}s wall, "
                  f"{(step+1)*dt:.2f}s sim)", flush=True); last = now
    print(f"  total {time.perf_counter()-t0:.0f}s wall", flush=True)
    return frames, dt * sample_every


def write_usdc(frames, frame_dt, out):
    print(f"Writing {out}...", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    rel = os.path.relpath(WORLD_USDA.resolve(), out.parent)
    stage.OverridePrim("/World").GetReferences().AddReference(rel)
    fps = 1.0 / frame_dt
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(len(frames) - 1)
    stage.SetTimeCodesPerSecond(fps); stage.SetFramesPerSecond(fps)

    op = {}
    for prim_path in (frames[0] if frames else {}):
        prim = stage.OverridePrim(prim_path)
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
        op[prim_path] = (xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, ""),
                         xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, ""))
    for fi, frame in enumerate(frames):
        tc = Usd.TimeCode(fi)
        for prim_path, (t, q) in frame.items():
            tr, oo = op[prim_path]
            tr.Set(Gf.Vec3d(*[float(v) for v in t]), tc)
            oo.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])), tc)
    stage.GetRootLayer().Save()

    # Flatten + strip refs for portability (same as ar4 recorder).
    flat = Usd.Stage.Open(str(out)).Flatten()

    def _strip(spec, is_root):
        if not is_root:
            spec.ClearReferenceList(); spec.ClearPayloadList()
        for c in spec.nameChildren:
            _strip(c, False)
    _strip(flat.GetPrimAtPath("/"), True)
    flat.Export(str(out))
    mb = out.stat().st_size / 1024 / 1024
    print(f"  wrote {len(frames)} frames @ {fps:.1f} fps "
          f"({len(frames)*frame_dt:.2f}s) → {out} ({mb:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-s", type=float, default=3.0)
    ap.add_argument("--sample-every", type=int, default=100)
    ap.add_argument("--out", type=str,
                    default=str(EXP_DIR / "output" / "recording.usdc"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    frames, frame_dt = run_simulation(args.sim_s, args.sample_every)
    write_usdc(frames, frame_dt, out)
    print(f"\nDone. Open in MICROROBOTICA recording mode / usdview:\n  {out}")


if __name__ == "__main__":
    main()
