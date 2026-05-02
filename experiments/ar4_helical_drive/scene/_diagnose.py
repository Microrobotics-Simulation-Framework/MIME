#!/usr/bin/env python3
"""Diagnostics for the AR4 + helical-UMR scene.

Three independent checks, each printable on its own:

* ``--scene``     — walk world.usda, print every prim's type, the
                    customData.mimeNodeName tag (if any), the result
                    of resolving any ``references`` arc, and the
                    composed local-to-world translate at composition
                    time. Useful when "things are at the wrong place
                    on the screen" — confirms whether a prim's static
                    USD transform is what you think it is *before*
                    ResultsApplicator stamps a runtime pose on it.
* ``--actors``    — load experiment.json, print every actor, its
                    declared prim_path, prim_type, and state_fields.
                    Cross-references against world.usda to flag
                    actors whose prim_path isn't found in the scene.
* ``--frame``     — run one physics step, print the resulting frame
                    payload (per-actor world poses) so you can see
                    exactly what the runner emits to MICROROBOTICA
                    each tick.

Default is ``--actors`` + ``--scene`` (cheap, no JAX init).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent.parent
WORLD_USDA = EXP_DIR / "scene" / "world.usda"
EXPERIMENT_YAML = EXP_DIR / "experiment.yaml"
EXPERIMENT_JSON = EXP_DIR / "experiment.json"


def diag_scene() -> int:
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        print("pxr (OpenUSD) not in venv — skipping --scene diagnostic.")
        return 0

    print(f"\n=== USD scene: {WORLD_USDA} ===")
    stage = Usd.Stage.Open(str(WORLD_USDA))
    if stage is None:
        print("FAIL: stage did not open")
        return 1

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()
        # mimeNodeName custom-data tag
        mime_name = ""
        cd = prim.GetCustomData()
        if cd and "mimeNodeName" in cd:
            mime_name = f"  ← actor '{cd['mimeNodeName']}'"
        # references arc (composition source)
        ref_str = ""
        try:
            refs = prim.GetMetadata("references")
            if refs:
                items = refs.GetAddedOrExplicitItems()
                if items:
                    ref_str = f"  refs={[str(r.assetPath) + r.primPath.pathString for r in items]}"
        except Exception:
            pass
        # composed translate (USD-static, before runtime override)
        translate_str = ""
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    val = op.Get()
                    if val:
                        translate_str = f"  static_translate={tuple(round(float(v), 3) for v in val)}"
                        break

        print(f"  {path:42s}  {str(type_name):10s}{translate_str}{ref_str}{mime_name}")

    return 0


def diag_actors() -> int:
    if not EXPERIMENT_JSON.exists():
        print(f"FAIL: {EXPERIMENT_JSON} missing — regenerate with"
              f"\n  python MICROROBOTICA/scripts/yaml_to_json.py {EXP_DIR}")
        return 1

    print(f"\n=== Actor declarations: {EXPERIMENT_JSON} ===")
    with EXPERIMENT_JSON.open() as f:
        cfg = json.load(f)
    actors = cfg.get("scene", {}).get("actors", {})
    print(f"  {len(actors)} actor(s)")

    # Load USD scene to cross-reference prim paths.
    scene_paths = set()
    try:
        from pxr import Usd
        stage = Usd.Stage.Open(str(WORLD_USDA))
        if stage is not None:
            scene_paths = {str(p.GetPath()) for p in stage.Traverse()}
    except ImportError:
        scene_paths = None  # cannot cross-check

    for name, spec in actors.items():
        prim_path = spec.get("prim_path", "<missing>")
        prim_type = spec.get("prim_type", "<missing>")
        fields = spec.get("state_fields", [])
        ok = (
            "✓" if scene_paths is None or prim_path in scene_paths
            else "✗ (prim path NOT in world.usda)"
        )
        print(f"  {name:14s}  {prim_path:36s}  {prim_type:8s}  {fields}  {ok}")

    return 0


def diag_frame() -> int:
    """Build the experiment's graph, run one step, print the frame."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    sys.path.insert(0, str(EXP_DIR / "physics"))

    print(f"\n=== Frame payload (one physics step) ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_setup", str(EXP_DIR / "physics" / "setup.py"),
    )
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    # Load + strip params (mimic what server.py does).
    params_ns = {}
    exec((EXP_DIR / "physics" / "params.py").read_text(), params_ns)
    params = {
        k: v for k, v in params_ns.items()
        if not k.startswith("_")
        and isinstance(v, (int, float, str, bool, list, tuple, dict))
    }

    gm = setup.build_graph(params)
    state = gm.step(external_inputs={})

    # Mimic _state_to_result_frame's composite-actor resolution.
    print("  per-actor world translate (m):")
    arm = state.get("arm", {})
    motor = state.get("motor", {})
    body = state.get("body", {})
    if "link_poses_world" in arm:
        import numpy as np
        link_poses = np.asarray(arm["link_poses_world"])
        for i, pose in enumerate(link_poses):
            print(f"    arm_link_{i}    ({float(pose[0]):+.3f}, "
                  f"{float(pose[1]):+.3f}, {float(pose[2]):+.3f})")
    if "rotor_pose_world" in motor:
        import numpy as np
        p = np.asarray(motor["rotor_pose_world"])
        print(f"    motor_rotor   ({float(p[0]):+.3f}, "
              f"{float(p[1]):+.3f}, {float(p[2]):+.3f})")
        print(f"    magnet        ({float(p[0]):+.3f}, "
              f"{float(p[1]):+.3f}, {float(p[2]):+.3f})")
    if "position" in body:
        import numpy as np
        p = np.asarray(body["position"])
        print(f"    body          ({float(p[0]):+.3f}, "
              f"{float(p[1]):+.3f}, {float(p[2]):+.3f})")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", action="store_true",
                    help="Walk world.usda and print every prim.")
    ap.add_argument("--actors", action="store_true",
                    help="Cross-check experiment.yaml/scene.actors against world.usda.")
    ap.add_argument("--frame", action="store_true",
                    help="Run one physics step, print emitted poses.")
    args = ap.parse_args()
    if not (args.scene or args.actors or args.frame):
        args.scene = True
        args.actors = True

    rc = 0
    if args.actors:
        rc |= diag_actors()
    if args.scene:
        rc |= diag_scene()
    if args.frame:
        rc |= diag_frame()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
