"""M3 — the schwarz_vessel_helix experiment composes effects-first and runs.

Pins that ``build_experiment`` assembles the coupled gate experiment **only via
``Experiment.attach``** (no hand-wired graph) and that the built graph **steps
stably** across both ``SWIM_MODE`` (free | held) and both ``FLOW_PROFILE``
(poiseuille | womersley) values.

The FVM coupling geometry (``sample_points``/``forcing_points``) is supplied each
step — constant for a held body, pose-tracked for a free one — via
``default_external_inputs(..., body_points_ref=, state=)``. (Omitting it makes the
``TwoScale`` coupling sample at the origin and diverge — see the effect docstring.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mime.experiments.schwarz_vessel_helix import (
    build_experiment, default_external_inputs, screw_points, _seed_body_orientation)

pytestmark = [pytest.mark.x64, pytest.mark.slow]

_DIR = Path("experiments/schwarz_vessel_helix")
_TABLE = Path("data/dejongh_benchmark/wall_tables/wall_R2.500.npz")
# coarse screw for a headless build/step smoke (feasibility, not resolution).
_BASE = {"N_THETA": 12, "N_ZETA": 16}

# the nodes each effect contributes — the composition contract.
_EXPECTED_NODES = {"body", "fvm", "bem", "motor", "ext_magnet", "magnet", "gravity"}


def _run(params, n=8):
    ref = screw_points(params)
    gm, handles = build_experiment(params).build()
    # attach-only composition: exactly the effect-contributed nodes, nothing else.
    assert set(gm.node_names) == _EXPECTED_NODES, set(gm.node_names)
    _seed_body_orientation(gm, params)             # body-z → world-x (vessel axis)
    last = None
    state = None
    for _ in range(n):
        ext = default_external_inputs(params, body_points_ref=ref, state=state)
        state = gm.step(ext)
        last = float(state["bem"]["drag_force"][2])
    return last, handles


def test_params_exec_like_the_runner():
    """The runner ``exec``s params.py in a **bare namespace** (no ``__file__``) and
    keeps only scalars (server.py::_execute_params). Guard that params.py honours
    that — a regression for the ``__file__``-in-params crash. (importlib-based
    loading sets ``__file__`` and would hide it, so mirror the runner exactly.)"""
    ns: dict = {}
    exec((_DIR / "physics" / "params.py").read_text(), ns)        # no __file__
    params = {k: v for k, v in ns.items()
              if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, tuple, dict))}
    assert "INCLUDE_ARM" in params and "SWIM_MODE" in params
    # every asset-path default must be ABSOLUTE and exist — the runner can launch
    # from any CWD (container workdir), so a relative default would FileNotFound.
    from mime.experiments.schwarz_vessel_helix import _DEFAULTS
    for key in ("ARM_URDF", "WALL_TABLE"):
        p = Path(_DEFAULTS[key])
        assert p.is_absolute(), f"{key} default is not absolute: {p}"
        assert p.exists(), f"{key} default missing: {p}"


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_builds_from_foreign_cwd(tmp_path, monkeypatch):
    """The runner launches from an arbitrary CWD; building must not depend on it
    (regression for the relative WALL_TABLE / ARM_URDF FileNotFound)."""
    monkeypatch.chdir(tmp_path)                        # NOT the repo root
    params = {**_BASE, "SWIM_MODE": "held", "FLOW_PROFILE": "poiseuille"}
    gm, _ = build_experiment(params).build()
    assert set(gm.node_names) == _EXPECTED_NODES


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
@pytest.mark.parametrize("swim", ["held", "free"])
@pytest.mark.parametrize("flow", ["poiseuille", "womersley"])
def test_builds_and_steps(swim, flow):
    params = {**_BASE, "SWIM_MODE": swim, "FLOW_PROFILE": flow}
    fz, handles = _run(params)
    print(f"\n[M3 {swim}/{flow}] Fz={fz:.3e}  handles={[h.node_names for h in handles]}")
    assert np.isfinite(fz)
    # effects-first: the four attached effects (TwoScale, PointDipole, Gravity[, Arm])
    # each returned a handle naming the nodes they added.
    assert len(handles) >= 3


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_arm_variant_builds():
    """INCLUDE_ARM wires the AR4 arm → motor parent pose as an effect (scene path)."""
    arm_urdf = Path("experiments/schwarz_vessel_helix/assets/ar4.urdf")
    if not arm_urdf.exists():
        pytest.skip("arm urdf absent")
    params = {**_BASE, "SWIM_MODE": "held", "FLOW_PROFILE": "poiseuille",
              "INCLUDE_ARM": True}
    gm, handles = build_experiment(params).build()
    assert "arm" in gm.node_names
    # the arm drives the motor's parent pose (carrier wiring).
    assert any("arm" in h.node_names for h in handles)
