"""M4 — the schwarz_vessel_helix scene loads and its actors map to the graph.

Pins the legibility/visualisation contract: the experiment reuses the full ar4 lab
(AR4 arm, rotor/magnet, desks, floor, transparent vessel, FL-9 UMR mesh, palette),
its USD references resolve, and every ``scene.actors`` entry's ``pose_from`` recipe
resolves against the live (built + stepped) graph — so the runner can drive each
prim. (Visual legibility itself is a manual viewport check — see M5 notes.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = [pytest.mark.x64, pytest.mark.slow]

_DIR = Path("experiments/schwarz_vessel_helix")
_TABLE = Path("data/dejongh_benchmark/wall_tables/wall_R2.500.npz")


def _yaml():
    return yaml.safe_load((_DIR / "experiment.yaml").read_text())


def test_scene_yaml_actors_present():
    cfg = _yaml()
    actors = cfg["scene"]["actors"]
    # SI scene = the full ar4 lab: 6 arm links + rotor + magnet + body.
    assert {f"arm_link_{i}" for i in range(6)} <= set(actors)
    assert {"motor_rotor", "magnet", "body"} <= set(actors)
    # vessel is parametric at PHYSICAL SI scale (de Jongh 1/4" tube ~3 mm radius).
    r = float(cfg["scene"]["environment"]["vessel"]["geometry"]["radius"])
    assert 1e-3 < r < 1e-2


def test_scene_usd_loads_self_contained_at_si_scale():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    import numpy as np
    stage = Usd.Stage.Open(str(_DIR / "scene" / "world.usda"))
    # self-contained: no layer outside the experiment dir.
    ext = [l.realPath for l in stage.GetUsedLayers()
           if l.realPath and "schwarz_vessel_helix" not in l.realPath]
    assert ext == [], ext
    prims = [p.GetPath().pathString for p in stage.Traverse()]
    assert sum("/Actors/Arm/L" in p for p in prims) == 6          # ar4 arm links
    assert any("LabFurniture" in p for p in prims)                # ar4 lab
    # the UMR screw mesh resolves at PHYSICAL SI scale (de Jongh ~mm, not nondim ~1).
    umr = next((p for p in stage.Traverse()
                if p.GetName() == "UMR" and p.IsA(UsdGeom.Mesh)), None)
    assert umr is not None
    pts = np.array([[v[0], v[1], v[2]] for v in UsdGeom.Mesh(umr).GetPointsAttr().Get()])
    assert 1e-3 < float(np.hypot(pts[:, 0], pts[:, 1]).max()) < 5e-3


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_actor_pose_recipes_resolve_against_graph():
    import importlib.util

    def _load(name):
        path = _DIR / name
        spec = importlib.util.spec_from_file_location(name.replace("/", "_"), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    pmod = _load("physics/params.py")
    params = {k: getattr(pmod, k) for k in dir(pmod) if k.isupper()}
    params.update(N_THETA=12, N_ZETA=16)               # coarse for speed
    setup = _load("physics/setup.py")
    ctrl = _load("control/controller.py")

    gm = setup.build_graph(params)
    state = None
    for i in range(2):
        state = gm.step(ctrl.get_external_inputs(params, i, state))

    for name, spec in _yaml()["scene"]["actors"].items():
        pf = spec.get("pose_from")
        node, field = (pf["node"], pf["field"]) if pf else (name, "position")
        val = state.get(node, {}).get(field)
        assert val is not None, f"actor {name}: {node}.{field} missing"
        if pf and "index" in pf:
            assert pf["index"] < np.asarray(val).shape[0]
