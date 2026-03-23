"""MIME-VER-013: Step-out frequency verification for d2.8 UMR configurations.

Verifies that the averaged ODE model's step-out frequency (detected from
the speed-vs-frequency curve peak) matches the paper values within 5%.

Each configuration is calibrated independently (C_rot from f_step), so the
analytical step-out is exact by construction. The test verifies that the
*dynamical* step-out — detected from the Adler equation rolloff in the
speed curve — agrees with the analytical/paper value.

d2.1 configs deferred to MIME-VER-014 (no validated geometry scaling law).
"""

import math
import os

import pytest
import numpy as np
import jax.numpy as jnp

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)
from mime.nodes.robot.umr_ode import (
    fit_drag_coefficients,
    compute_step_out_frequency,
    _mean_angular_velocity,
)


DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "docs", "validation", "umr_deboer2025", "deboer_fig12_digitised",
)


def _load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    return np.loadtxt(path, delimiter=",", skiprows=1)


M_SINGLE = 1.07e-3
B_FIELD = 3e-3


def _detect_step_out_from_sweep(params, f_max, df=1.0):
    """Sweep frequencies and detect step-out as the peak of U_ss(f).

    Uses the analytical steady state: U_ss = (C_prop/C_trans) * <Omega>(f).
    """
    freqs = np.arange(5.0, f_max, df)
    omega_arr = jnp.array(2.0 * math.pi * freqs)
    omega_so = params["n_mag"] * params["m_single"] * params["B"] / params["C_rot"]

    Omega_avg = np.array([
        float(_mean_angular_velocity(jnp.array(w), jnp.array(omega_so)))
        for w in omega_arr
    ])

    speeds = (params["C_prop"] / params["C_trans"]) * Omega_avg
    idx_peak = np.argmax(speeds)
    return freqs[idx_peak]


D28_CONFIGS = [
    {"name": "d2.8_1mag", "n_mag": 1, "f_step": 128.0, "csv": "d2.8_1mag.csv"},
    {"name": "d2.8_2mag", "n_mag": 2, "f_step": 181.0, "csv": "d2.8_2mag.csv"},
    {"name": "d2.8_3mag", "n_mag": 3, "f_step": 222.0, "csv": "d2.8_3mag.csv"},
]


@verification_benchmark(
    benchmark_id="MIME-VER-013",
    description="Step-out frequencies for d2.8 configs within 5% of paper values",
    node_type="umr_ode",
    benchmark_type=BenchmarkType.REGRESSION,
    acceptance_criteria="Detected step-out within 5% of paper f_step for each config",
    references=("deBoer2025",),
)
def test_step_out_frequencies():
    """Verify step-out detection from speed curve matches paper values.

    Each configuration is independently calibrated (C_rot = n*m*B / (2*pi*f_step)),
    so the Adler equation step-out exactly equals the input f_step. This test
    verifies that the speed-curve peak detection recovers that frequency, confirming
    the sweep and detection pipeline is self-consistent.
    """
    for cfg in D28_CONFIGS:
        data = _load_csv(cfg["csv"])
        params = fit_drag_coefficients(
            data,
            n_mag=cfg["n_mag"],
            m_single=M_SINGLE,
            B=B_FIELD,
            f_step=cfg["f_step"],
        )

        # Analytical step-out (should be exact by construction)
        f_analytical = compute_step_out_frequency(
            cfg["n_mag"], M_SINGLE, B_FIELD, params["C_rot"],
        )

        # Dynamical step-out from speed curve peak
        f_detected = _detect_step_out_from_sweep(
            params, f_max=1.5 * cfg["f_step"], df=1.0,
        )

        # Analytical should match paper exactly (by construction)
        rel_err_analytical = abs(f_analytical - cfg["f_step"]) / cfg["f_step"]
        assert rel_err_analytical < 0.01, (
            f"{cfg['name']}: analytical f_step={f_analytical:.1f} Hz "
            f"vs paper {cfg['f_step']:.0f} Hz ({rel_err_analytical:.1%} error)"
        )

        # Detected should match paper within 5% (1 Hz sweep resolution)
        rel_err_detected = abs(f_detected - cfg["f_step"]) / cfg["f_step"]
        assert rel_err_detected < 0.05, (
            f"{cfg['name']}: detected f_step={f_detected:.1f} Hz "
            f"vs paper {cfg['f_step']:.0f} Hz ({rel_err_detected:.1%} error)"
        )
