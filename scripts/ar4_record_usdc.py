#!/usr/bin/env python3
"""Record the AR4 helical-drive experiment to a .usdc file.

Runs the full graph + closed-loop controller for SIM_S seconds,
captures the pose of every animated actor (arm_link_0..5,
motor_rotor, magnet, body) every SAMPLE_EVERY physics steps, and
writes a self-contained .usdc that:

  * `references` the experiment's existing ``scene/world.usda`` for
    the static lab-furniture and mesh assets, so the recording uses
    the same visual scene.
  * Layers time-sampled ``xformOp:translate`` and ``xformOp:orient``
    on the animated Xform prims, indexed by the prims'
    ``customData.mimeNodeName`` tags.
  * Stores stage time metadata so the viewer plays at real-time.

Usage:
    .venv/bin/python scripts/ar4_record_usdc.py
    .venv/bin/python scripts/ar4_record_usdc.py --sim-s 5.0 --out demo.usdc

The output is openable in usdview / Omniverse / MICROROBOTICA's
"open recording" mode (any USD-aware viewer with playback).

Mesh geometry is flattened in from ``scene/world.usda`` →
``assets/ar4_meshes.usdc``. That asset is kept viewer-friendly by
``scene/_bake_meshes.py``, which decimates each mesh on bake — raw
CAD-density meshes (~1M triangles) shimmer with geometric aliasing in the
viewport. If you point this recorder at a different or re-baked mesh asset,
run ``scripts/decimate_ar4_meshes.py`` on the result. See
``docs/experiment_recordings.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

# Force CPU for reproducibility — recording is offline.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
from pxr import Usd, UsdGeom, Gf, Sdf

EXP_DIR = REPO_ROOT / "experiments" / "ar4_helical_drive"
PARAMS_PATH = EXP_DIR / "physics" / "params.py"
SETUP_PATH = EXP_DIR / "physics" / "setup.py"
CONTROLLER_PATH = EXP_DIR / "control" / "controller.py"
WORLD_USDA = EXP_DIR / "scene" / "world.usda"

# Each entry: (USD prim path, source-of-pose callable)
# The pose callable takes the per-step `state` dict and returns
# (translate xyz [m], orient quat [wxyz]).
_arm_link_n = lambda i: (
    f"/World/Actors/Arm/L{i}",
    lambda s, i=i: (
        np.asarray(s["arm"]["link_poses_world"][i][:3]),
        np.asarray(s["arm"]["link_poses_world"][i][3:7]),
    ),
)
_motor_rotor = (
    "/World/Actors/Magnet/Rotor",
    lambda s: (
        np.asarray(s["motor"]["rotor_pose_world"][:3]),
        np.asarray(s["motor"]["rotor_pose_world"][3:7]),
    ),
)
_magnet_body = (
    "/World/Actors/Magnet/Body",
    lambda s: (
        np.asarray(s["motor"]["rotor_pose_world"][:3]),
        np.asarray(s["motor"]["rotor_pose_world"][3:7]),
    ),
)
_helix = (
    "/World/Actors/UMR",
    lambda s: (
        np.asarray(s["body"]["position"]),
        np.asarray(s["body"]["orientation"]),
    ),
)


def animated_prim_specs():
    """List of (prim_path, getter_fn) for everything that moves."""
    specs = [_arm_link_n(i) for i in range(6)]
    specs.append(_motor_rotor)
    specs.append(_magnet_body)
    specs.append(_helix)
    return specs


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _load_params() -> dict:
    ns: dict = {}
    with open(PARAMS_PATH) as fh:
        exec(fh.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}


def run_simulation(sim_s: float, sample_every: int):
    """Run the full graph and return a list of frame dicts."""
    params = _load_params()
    setup = _load_module("ar4_setup", SETUP_PATH)
    ctrl = _load_module("ar4_ctrl", CONTROLLER_PATH)
    ctrl._controller_instance = None

    print(f"Building graph...", flush=True)
    gm = setup.build_graph(params)
    dt = float(params["DT_PHYS"])
    n_steps = int(sim_s / dt)
    n_frames = n_steps // sample_every
    print(f"  {n_steps} steps × {dt*1000:.2f} ms = {sim_s:.2f} s, "
          f"sampling every {sample_every} steps → {n_frames} frames",
          flush=True)

    prev_state = {n: gm.get_node_state(n) for n in gm._nodes}
    specs = animated_prim_specs()

    frames = []
    t_start = time.perf_counter()
    last_report = t_start
    for step in range(n_steps):
        ext = ctrl.get_external_inputs(params, step, state=prev_state)
        gm.step(ext)
        prev_state = {n: gm.get_node_state(n) for n in gm._nodes}

        if step % sample_every == 0:
            frame = {}
            for prim_path, getter in specs:
                t, q = getter(prev_state)
                frame[prim_path] = (t.copy(), q.copy())
            frames.append(frame)

        now = time.perf_counter()
        if now - last_report > 5.0:
            elapsed = now - t_start
            sim_elapsed = (step + 1) * dt
            print(f"  step {step+1}/{n_steps} "
                  f"({elapsed:.0f}s wall, {sim_elapsed:.2f}s sim, "
                  f"{elapsed/max(sim_elapsed, 1e-6):.1f}× slower than real-time)",
                  flush=True)
            last_report = now

    print(f"  total {time.perf_counter() - t_start:.1f}s wall", flush=True)
    return frames, dt * sample_every


def write_usdc(frames, frame_dt: float, output_path: Path):
    """Build the .usdc with time-sampled overrides on animated prims."""
    print(f"Writing {output_path}...", flush=True)
    n_frames = len(frames)

    # Create the stage on disk first so USD knows the layer's
    # directory; relative reference paths are then resolved against
    # *the saved file's* location, not the build machine's CWD.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Reference the experiment's world.usda by RELATIVE path so the
    # recording is portable: any machine with the experiment tree
    # can open it (the .usdc lives at
    # experiments/ar4_helical_drive/recordings/ and the scene at
    # experiments/ar4_helical_drive/scene/world.usda — relative
    # link `../scene/world.usda`).
    rel_to_world = os.path.relpath(WORLD_USDA.resolve(), output_path.parent)
    root = stage.OverridePrim("/World")
    root.GetReferences().AddReference(rel_to_world)

    # Playback FPS: 1/frame_dt (so playback runs in real time).
    fps = 1.0 / frame_dt
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n_frames - 1)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)

    # Pre-create the overridden prims and the xformOp definitions.
    specs = animated_prim_specs()
    op_cache = {}
    for prim_path, _ in specs:
        # Ensure the prim exists as an override on the referenced
        # world.usda's prims.
        prim = stage.OverridePrim(prim_path)
        xf = UsdGeom.Xformable(prim)
        # Wipe any existing op order from the reference layer and
        # set our two ops in the order our recording uses.
        xf.ClearXformOpOrder()
        translate = xf.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble, ""
        )
        orient = xf.AddOrientOp(
            UsdGeom.XformOp.PrecisionDouble, ""
        )
        op_cache[prim_path] = (translate, orient)

    # Stamp time-sampled values.
    for frame_idx, frame in enumerate(frames):
        tc = Usd.TimeCode(frame_idx)
        for prim_path, (t, q) in frame.items():
            translate, orient = op_cache[prim_path]
            translate.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])), tc)
            # USD quat is (real, i, j, k) = (w, x, y, z) — same as ours.
            orient.Set(
                Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
                tc,
            )

    stage.GetRootLayer().Save()

    # Flatten + strip references for portability. The recorder writes
    # the .usdc with a `references = @../scene/world.usda@` declaration
    # so the on-disk file is small (refs into the experiment tree),
    # but on a different machine USD's reference resolver can't find
    # the world.usda and rendering fails.  Flattening inlines every
    # composed prim's data; stripping references then removes the
    # now-dangling reference metadata so loaders don't warn.
    stage_to_flatten = Usd.Stage.Open(str(output_path))
    flat = stage_to_flatten.Flatten()

    def _strip(spec, is_root):
        if not is_root:
            spec.ClearReferenceList()
            spec.ClearPayloadList()
        for child in spec.nameChildren:
            _strip(child, False)

    _strip(flat.GetPrimAtPath("/"), True)
    flat.Export(str(output_path))

    final_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {n_frames} frames at {fps:.1f} fps "
          f"({n_frames * frame_dt:.2f}s playback) → {output_path} "
          f"({final_size_mb:.1f} MB, self-contained)",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-s", type=float, default=5.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--sample-every", type=int, default=100,
                        help="Sample every N physics steps (dt=5e-4 → "
                             "100 = 20 fps, matching MICROROBOTICA's "
                             "QTimer playback rate; values much "
                             "higher than 60 fps are silently clamped "
                             "and only ~40% of frames render)")
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "experiments"
                                    / "ar4_helical_drive" / "recordings"
                                    / "ar4_demo.usdc"),
                        help="Output .usdc path")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    frames, frame_dt = run_simulation(args.sim_s, args.sample_every)
    write_usdc(frames, frame_dt, out)
    print(f"\nDone. Open in usdview or MICROROBOTICA recording mode:")
    print(f"  {out}")


if __name__ == "__main__":
    main()
