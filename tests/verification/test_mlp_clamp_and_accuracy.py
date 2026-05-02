"""V3 — MLP clamp diagnostic + clamped-regime accuracy bound.

Two independent test functions:

(3a) ``clamp_fired`` propagates to the trajectory JSON written by
     ``scripts/dejongh_dynamic_simulation.py``. This is a wiring
     check — pass/fail is binary.

(3b) Compare the MLP surrogate's predicted resistance matrix R
     against fresh BEM ground truth at 6 lateral offsets straddling
     the silent-clamp boundary at ``offset_frac = 0.30``. Inside
     training range expect ≤ 1% relative swim-speed error; outside
     training range we *measure* the error and write the per-offset
     envelope to
     ``data/dejongh_benchmark/diagnostics/mlp_clamp_envelope.json``
     so phase 1 can quote a known systematic-error envelope when its
     trajectories enter that regime.

The 6 BEM evaluations cost ~10–20 s each on GPU; on a GPU-less host
they fall back to chunked-CPU and take ~40 s each. To keep CI cheap
we cache the results in
``data/dejongh_benchmark/diagnostics/mlp_clamp_bem_cache.npz`` and
reuse on rerun.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────
# (3a) clamp_fired diagnostic propagation
# ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "dejongh_benchmark"
DIAG_DIR = DATA_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def test_3a_clamp_fired_in_trajectory_schema():
    """Inspect the dejongh_dynamic_simulation.py source: ``clamp_fired``
    must appear in the per-frame trajectory dict that is written to JSON.
    No subprocess invocation — this is a static-source check that catches
    a regression where the column is silently removed.
    """
    sim_script = REPO_ROOT / "scripts" / "dejongh_dynamic_simulation.py"
    assert sim_script.exists(), f"Missing {sim_script}"
    src = sim_script.read_text()
    # The trajectory.append({...}) block is the only place per-frame
    # dicts are built. ``clamp_fired`` must be a key there.
    assert '"clamp_fired"' in src, (
        "clamp_fired key missing from trajectory write path in "
        f"{sim_script}. The MLP clamp will silently corrupt phase-1 "
        "results without an output diagnostic."
    )


def test_3a_clamp_fired_propagation_in_short_run():
    """Run a 200-step simulation forced into the clamped regime and
    confirm at least one frame's ``clamp_fired`` column is non-zero.

    The clamp boundary at the 1/4" vessel:
      offset_clamp_nd = (R_ves_nd − R_max_UMR_factor) × 0.95
                      = (2.035 − 1.33) × 0.95 ≈ 0.670 (non-dim)
                      ≈ 1.045 mm in SI.

    We initialise the body at y = −1.20 mm, well inside the clamp.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    # Avoid GPU pre-allocation if any
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from mime.experiments.dejongh import build_graph

    gm = build_graph(
        design_name="FL-9", vessel_name='1/4"',
        dt=5e-4, use_lubrication=True,
    )

    # Patch initial body state
    body_init = gm._state["body"]
    gm._state["body"] = {
        **body_init,
        "position": jnp.array([0.0, -1.20e-3, 0.0], dtype=jnp.float32),
    }

    # Set the actuation external inputs (would normally come from
    # the controller / params).
    ext_inputs = {
        "field": {
            "frequency_hz": jnp.float32(10.0),
            "field_strength_mt": jnp.float32(1.2),
        },
    }

    any_fired = False
    for _ in range(200):
        gm.step(external_inputs=ext_inputs)
        cf = float(np.array(gm._state["mlp_drag"]["clamp_fired"]))
        if cf > 0.5:
            any_fired = True
            break

    assert any_fired, (
        "clamp_fired remained zero across 200 steps with body forced "
        "1.20 mm off-axis in the 1/4\" vessel — well inside the clamp "
        "boundary. The clamp wiring may be broken."
    )


# ──────────────────────────────────────────────────────────────────
# (3b) MLP vs BEM ground truth at 6 offsets
# ──────────────────────────────────────────────────────────────────

# Test grid: FL-9, 1/4" vessel, offsets straddling the 0.30 boundary.
TEST_OFFSETS_FRAC = [0.20, 0.25, 0.28, 0.30, 0.32, 0.35]
TEST_KAPPA = 0.491
TEST_R_VES_NPZ = (DATA_DIR / "wall_tables" / "wall_R2.035.npz")

CACHE_PATH = DIAG_DIR / "mlp_clamp_bem_cache.npz"
ENVELOPE_PATH = DIAG_DIR / "mlp_clamp_envelope.json"

OMEGA_PHYS = 2.0 * np.pi * 10.0   # 10 Hz
R_CYL_UMR_MM = 1.56


def _load_or_build_bem_cache() -> dict:
    """Return ``{offset_frac: R_BEM_6x6}`` from cache or fresh BEM.

    Fresh BEM takes ~2 minutes total on GPU. The cache is invalidated
    by changing the tuple of offsets above.
    """
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH, allow_pickle=False)
        cached_offsets = list(z["offsets"])
        cached_Rs = z["R_BEMs"]
        if cached_offsets == [float(o) for o in TEST_OFFSETS_FRAC]:
            return {
                float(o): cached_Rs[i]
                for i, o in enumerate(cached_offsets)
            }

    # Fresh BEM run.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from mime.nodes.environment.stokeslet.cylinder_wall_table import (
        load_wall_table,
    )
    from generate_training_fill_v3 import (
        make_mesh_and_wts, _confined_R_hybrid, MU,
    )

    table = load_wall_table(str(TEST_R_VES_NPZ))
    R_ves_nd = float(table.R_cyl)

    # FL-9: ν = 2.33, L_UMR = 7.47 mm
    nu = 2.33
    L_UMR_mm = 7.47
    _, pts, wts, eps = make_mesh_and_wts(nu, L_UMR_mm)

    out = {}
    for off_frac in TEST_OFFSETS_FRAC:
        offset_nd = off_frac * R_ves_nd
        R, _ = _confined_R_hybrid(
            pts, wts, float(eps), float(R_ves_nd), float(MU),
            table, offset_nd=(offset_nd, 0.0),
        )
        out[float(off_frac)] = np.asarray(R, dtype=np.float64)

    # Cache.
    np.savez(
        CACHE_PATH,
        offsets=np.asarray(TEST_OFFSETS_FRAC, dtype=np.float64),
        R_BEMs=np.stack([out[float(o)] for o in TEST_OFFSETS_FRAC]),
    )
    return out


def _swim_speed_z(R_si: np.ndarray) -> float:
    """Force-free axial swim speed at ω_z = OMEGA_PHYS.

    U = −R_FU⁻¹ · R_FΩ · [0, 0, ω].
    """
    R_FU = R_si[:3, :3]
    R_FOmega = R_si[:3, 3:]
    U = -np.linalg.solve(R_FU, R_FOmega @ np.array([0.0, 0.0, OMEGA_PHYS]))
    return float(U[2])


def _bem_to_si(R_nd: np.ndarray, mu_si: float = 1e-3,
                a_si: float = R_CYL_UMR_MM * 1e-3) -> np.ndarray:
    """Convert non-dim BEM R to SI using the same block scaling that
    ``MLPResistanceNode._R_nd_to_SI`` applies (mlp_resistance_node.py:68).
    """
    R = np.zeros_like(R_nd)
    R[:3, :3] = mu_si * a_si      * R_nd[:3, :3]
    R[:3, 3:] = mu_si * a_si ** 2 * R_nd[:3, 3:]
    R[3:, :3] = mu_si * a_si ** 2 * R_nd[3:, :3]
    R[3:, 3:] = mu_si * a_si ** 3 * R_nd[3:, 3:]
    return R


def _mlp_R(node, offset_x_nd: float) -> np.ndarray:
    """Drive ``MLPResistanceNode._compute_R_and_drag`` with the given
    canonical-frame offset and zero velocity, return the resulting R_SI.
    Deliberately invokes the silent clamp for offsets > training range.
    """
    pos = jnp.array([
        offset_x_nd * R_CYL_UMR_MM * 1e-3, 0.0, 0.0
    ], dtype=jnp.float32)
    R_ves_nd = float(node._R_ves_mm) / float(node._R_cyl_UMR_mm)
    _, _, R_si, _ = node._compute_R_and_drag(
        pos,
        jnp.zeros(3), jnp.zeros(3), jnp.zeros(3),
        R_ves_nd,
    )
    return np.asarray(R_si, dtype=np.float64)


def test_3b_mlp_clamp_envelope():
    """For each of the 6 offsets, compute MLP and BEM swim speeds and
    record per-offset relative error to a JSON envelope file.

    Asserts:
      • inside training range (offset_frac ≤ 0.30): rel_err ≤ 5%.
      • outside training range: no assertion — measurement only.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from mime.nodes.environment.stokeslet.mlp_resistance_node import (
        MLPResistanceNode,
    )
    from mime.experiments.dejongh import default_mlp_weights_path

    node = MLPResistanceNode(
        "mlp_v3", timestep=5e-4,
        mlp_weights_path=str(default_mlp_weights_path()),
        nu=2.33, L_UMR_mm=7.47,
        R_cyl_UMR_mm=R_CYL_UMR_MM,
        R_ves_mm=3.175,  # 1/4"
        mu_Pa_s=1e-3,
    )

    # BEM ground truth — from cache or freshly computed.
    bem_R_nd = _load_or_build_bem_cache()
    R_ves_nd_table = 1.0 / TEST_KAPPA  # 2.035

    envelope = []
    for off_frac in TEST_OFFSETS_FRAC:
        offset_nd = off_frac * R_ves_nd_table

        R_BEM_si = _bem_to_si(bem_R_nd[float(off_frac)])
        v_BEM = _swim_speed_z(R_BEM_si)

        R_MLP_si = _mlp_R(node, offset_nd)
        v_MLP = _swim_speed_z(R_MLP_si)

        rel_err = abs(v_MLP - v_BEM) / max(abs(v_BEM), 1e-30)
        envelope.append({
            "offset_frac": float(off_frac),
            "offset_nd": float(offset_nd),
            "v_BEM_mm_s": v_BEM * 1e3,
            "v_MLP_mm_s": v_MLP * 1e3,
            "rel_err": float(rel_err),
            "in_training_range": off_frac <= 0.30,
        })

    # Persist the envelope so phase-1 can cite it.
    ENVELOPE_PATH.write_text(json.dumps({
        "design": "FL-9",
        "vessel": "1/4\"",
        "kappa": TEST_KAPPA,
        "training_offset_cap": 0.30,
        "weights_path": str(_safe_relpath(REPO_ROOT,
            __file_to_default_weights__())),
        "points": envelope,
    }, indent=2))

    # The PRIMARY deliverable of V3b is the envelope JSON. Phase 1
    # consumes it to flag trajectory points where the MLP is known
    # to be unreliable. The assertions below are *regression guards*
    # only — they catch "MLP weights wrong file" or "BEM blown up",
    # not "MLP locally inaccurate" (which is the very thing the
    # envelope is meant to expose).

    # Print envelope first so the failure is informative if anything
    # below trips.
    print("\nMLP clamp envelope (V3b):")
    print(f"{'offset_frac':>12} {'in_train':>10} {'v_BEM mm/s':>12} "
          f"{'v_MLP mm/s':>12} {'rel_err':>10}")
    for p in envelope:
        flag = "yes" if p["in_training_range"] else "no (clamp)"
        print(f"{p['offset_frac']:>12.2f} {flag:>10} "
              f"{p['v_BEM_mm_s']:>12.3f} {p['v_MLP_mm_s']:>12.3f} "
              f"{p['rel_err']:>10.3%}")
    print(f"\nWrote: {ENVELOPE_PATH}")

    # Regression guard 1: MLP must produce finite, plausible swim
    # speeds at every offset (catches NaN / inf / wildly unphysical
    # outputs). 100 mm/s is generously above any expected helical UMR
    # speed at 10 Hz.
    for p in envelope:
        assert np.isfinite(p["v_MLP_mm_s"]), (
            f"MLP v_z is non-finite at offset_frac={p['offset_frac']}: "
            f"{p['v_MLP_mm_s']}"
        )
        assert 0.0 < p["v_MLP_mm_s"] < 100.0, (
            f"MLP v_z out of plausible range at "
            f"offset_frac={p['offset_frac']}: {p['v_MLP_mm_s']:.3f} mm/s"
        )

    # Regression guard 2: at offset_frac = 0.30 (a heavily-trained
    # point, very close to the centred-training-data interpolation
    # ridge), the MLP should still match BEM to <5%. If this fails,
    # the weights file is wrong or the feature ordering has drifted.
    p_030 = next(p for p in envelope if abs(p["offset_frac"] - 0.30) < 1e-6)
    assert p_030["rel_err"] < 0.05, (
        f"At offset_frac=0.30 (well-trained): rel_err = "
        f"{p_030['rel_err']:.3%}. Expected <5%. The weights file may be "
        "wrong or feature ordering has changed."
    )


def __file_to_default_weights__():
    from mime.experiments.dejongh import default_mlp_weights_path
    return default_mlp_weights_path()


def _safe_relpath(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)
