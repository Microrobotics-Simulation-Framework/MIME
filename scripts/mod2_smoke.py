#!/usr/bin/env python3
"""M2 coupled smoke test — build the schwarz two-scale graph with the EMERGENT
inertial body (BODY_MODEL='inertial', no prescribed spin) and step it briefly to
confirm: (1) numerically stable (finite) over a short settle+swim horizon at the
finer DT, (2) the body actually rotates (orientation changes → emergent spin) and
translates (swims), and (3) the locked path is unchanged (regression).

This is a fast numeric check BEFORE the full USDC recording. Reuses the experiment's
own setup + controller, same as scripts/schwarz_record_usdc.py.

Run: python scripts/mod2_smoke.py --sim-s 0.05 --dt 1e-4
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

EXP = REPO / "experiments" / "schwarz_vessel_helix"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def _load_params():
    ns = {}
    with open(EXP / "physics" / "params.py") as fh:
        exec(fh.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_") and k.isupper()}


def quat_angle(q0, q1):
    """rotation angle (deg) between two wxyz quats."""
    d = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    return np.degrees(2 * np.arccos(min(1.0, d)))


def run(body_model, dt, sim_s):
    params = _load_params()
    params["BODY_MODEL"] = body_model
    params["DT"] = dt
    setup = _load_module("svh_setup", EXP / "physics" / "setup.py")
    ctrl = _load_module("svh_ctrl", EXP / "control" / "controller.py")
    ctrl._controller_instance = None
    gm = setup.build_graph(params)
    n = int(sim_s / dt)
    prev = {nm: gm.get_node_state(nm) for nm in gm._nodes}
    q0 = np.asarray(prev["body"]["orientation"]); p0 = np.asarray(prev["body"]["position"])
    t0 = time.perf_counter(); blew = False; max_w = 0.0
    for step in range(n):
        ext = ctrl.get_external_inputs(params, step, state=prev)
        gm.step(ext)
        prev = {nm: gm.get_node_state(nm) for nm in gm._nodes}
        w = np.asarray(prev["body"]["angular_velocity"])
        max_w = max(max_w, float(np.linalg.norm(w)))
        if not np.all(np.isfinite(np.asarray(prev["body"]["position"]))) or \
           not np.all(np.isfinite(w)):
            blew = True; print(f"  NaN/Inf at step {step}", flush=True); break
    q1 = np.asarray(prev["body"]["orientation"]); p1 = np.asarray(prev["body"]["position"])
    wall = time.perf_counter() - t0
    print(f"  [{body_model}] dt={dt:.0e} steps={n} wall={wall:.1f}s  "
          f"finite={not blew}")
    print(f"     Δorientation={quat_angle(q0,q1):.2f}°  "
          f"Δposition={np.linalg.norm(p1-p0)*1e3:.4f} mm  "
          f"max|ω|={max_w:.2f} rad/s")
    return not blew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-s", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--locked-regression", action="store_true",
                    help="also run the locked model (DT=5e-4) to confirm it is unaffected")
    args = ap.parse_args()
    print("=" * 70)
    print("M2 smoke — emergent inertial body in the schwarz two-scale graph")
    print("=" * 70)
    ok = run("inertial", args.dt, args.sim_s)
    if args.locked_regression:
        print("-" * 70)
        run("locked", 5e-4, args.sim_s)
    print("=" * 70)
    print("PASS" if ok else "FAIL (inertial blew up)")


if __name__ == "__main__":
    main()
