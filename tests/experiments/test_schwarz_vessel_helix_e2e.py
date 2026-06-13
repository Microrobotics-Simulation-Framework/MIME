"""M5 — schwarz_vessel_helix end-to-end gate: run, record, differentiate.

The gate deliverable. Pins that the composed coupled experiment:

  1. **runs end-to-end** in both ``SWIM_MODE`` values over a short horizon, finite,
     and writes a trajectory recording artifact (body pose + confined drag vs step)
     — ``free`` shows the screw swimming, ``held`` the reaction drag;
  2. **is differentiable** end-to-end through the coupled two-scale graph — the
     gradient of the confined drag w.r.t. a control param (the held body's prescribed
     velocity) matches a central finite difference (the sensitivity the dual-RPM
     controller / inverse design needs; Phase-D differentiability applied to *this*
     graph).

Caveats (in the experiment metadata, restated): coupled drag is **differential**
(self-wake subtraction is a known-divergent blocker, OFF) — absolute magnitude is
not physical; primary validity Re ≲ 25, λ ≤ 0.8; non-dimensional wall-table units.
The USD ``recording.usdc`` proper is the MICROROBOTICA runner's output; here we
write a headless trajectory ``.npz`` as the recording artifact.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.experiments.schwarz_vessel_helix import (
    build_experiment, default_external_inputs, screw_points, _seed_body_orientation)

pytestmark = [pytest.mark.x64, pytest.mark.slow]

_TABLE = Path("data/dejongh_benchmark/wall_tables/wall_R2.500.npz")
_OUT = Path("experiments/schwarz_vessel_helix/output")
_BASE = {"N_THETA": 14, "N_ZETA": 20}


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
@pytest.mark.parametrize("swim", ["held", "free"])
def test_run_and_record(swim):
    params = {**_BASE, "SWIM_MODE": swim, "FLOW_PROFILE": "poiseuille",
              "INCLUDE_ARM": False}
    ref = screw_points(params)
    gm, _ = build_experiment(params).build()
    _seed_body_orientation(gm, params)             # body-z → world-x (the pipe axis)

    n = 25
    pos = np.empty((n, 3)); orient = np.empty((n, 4)); drag = np.empty(n)
    state = None
    for k in range(n):
        ext = default_external_inputs(params, body_points_ref=ref, state=state)
        state = gm.step(ext)
        pos[k] = np.asarray(state["body"]["position"])
        orient[k] = np.asarray(state["body"]["orientation"])
        drag[k] = float(state["bem"]["drag_force"][0])     # axial = x (pipe axis)

    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(drag))
    # held (kinematic, V=Ω=0): pose frozen. free (dynamic): state evolves (the screw
    # responds to drive + flow). NB: the calibrated corkscrew swim is an open debug
    # item — here we only pin run-stability + the held/free distinction.
    moved = float(np.linalg.norm(pos[-1] - pos[0]) + np.linalg.norm(orient[-1] - orient[0]))
    if swim == "free":
        assert moved > 1e-9                            # not frozen
    else:
        assert np.allclose(pos, pos[0], atol=1e-9)     # held fixed

    _OUT.mkdir(exist_ok=True)
    rec = _OUT / f"trajectory_{swim}.npz"
    np.savez(rec, position=pos, orientation=orient, drag_x=drag,
             swim_mode=swim, note="SI units (m, N); ar4 axes (pipe=x); differential drag")
    assert rec.exists()
    print(f"\n[M5 {swim}] {n} steps finite; drag_x[-1]={drag[-1]:.3e} N; wrote {rec.name}")


@pytest.mark.skipif(not _TABLE.exists(), reason=f"wall table absent: {_TABLE}")
def test_coupled_experiment_is_differentiable():
    """∂(confined drag_z)/∂(prescribed body Vz) through the full coupled graph,
    FD-consistent — Phase-D differentiability on the schwarz_vessel_helix graph."""
    params = {**_BASE, "N_THETA": 12, "N_ZETA": 16,
              "SWIM_MODE": "held", "FLOW_PROFILE": "poiseuille", "INCLUDE_ARM": False}
    ref = screw_points(params)
    gm, _ = build_experiment(params).build()
    gm.compile()
    step, s0 = gm._compiled_step, gm._state

    base_ext = default_external_inputs(params, body_points_ref=ref)

    def drag_after(Vz, nsteps=4):
        ext = {**base_ext}
        ext["body"] = {"external_velocity": jnp.array([0.0, 0.0, Vz]),
                       "external_angular_velocity": jnp.zeros(3)}
        s = s0
        for _ in range(nsteps):
            s = step(s, ext)
        return s["bem"]["drag_force"][2]

    V = 0.5
    grad = float(jax.grad(drag_after)(V))
    eps = 1e-4
    fd = float((drag_after(V + eps) - drag_after(V - eps)) / (2 * eps))
    print(f"\n[M5 grad] d(drag_z)/dVz: analytic={grad:.5e} fd={fd:.5e} "
          f"rel.err={abs(grad - fd) / abs(fd):.2e}")
    assert np.isfinite(grad)
    assert abs(grad - fd) / abs(fd) < 1e-4
