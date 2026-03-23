"""UMR ODE validation — verification benchmarks against de Boer 2025 Fig. 12.

MIME-VER-010: Calibration baseline (d2.8, 1 magnet) — RMSE < 5% of peak
MIME-VER-011: Prediction (d2.8, 2 and 3 magnets) — RMSE < 15% of peak

Fitting strategy:
    - Calibrate C_rot, C_prop, C_trans from 1-magnet data
    - Predict 2- and 3-magnet curves with same drag coefficients
      (only n_mag and omega_field change)
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
    umr_averaged_speed_curve,
    params_dict_to_array,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "docs", "validation", "umr_deboer2025", "deboer_fig12_digitised",
)


def _load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _compute_rmse(t_data, U_data, t_sim, U_sim):
    """Interpolate simulation to data time points, compute RMSE."""
    U_interp = np.interp(t_data, np.array(t_sim), np.array(U_sim))
    return float(np.sqrt(np.mean((U_interp - U_data) ** 2)))


# ---------------------------------------------------------------------------
# MIME-VER-010: Calibration baseline (1 magnet)
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-010",
    description="UMR ODE speed curve RMSE < 5% of peak for d2.8_1mag (calibration)",
    node_type="umr_ode",
    benchmark_type=BenchmarkType.REGRESSION,
    acceptance_criteria="RMSE / peak_speed < 0.05",
    references=("deBoer2025",),
)
def test_umr_1mag_calibration():
    """Calibrate model on 1-magnet data; verify fit quality."""
    data = _load_csv("d2.8_1mag.csv")
    t_data = data[:, 0]
    U_data = data[:, 1]
    peak_speed = float(np.max(U_data))

    # Fit
    params = fit_drag_coefficients(
        data, n_mag=1, m_single=1.07e-3, B=3e-3, f_step=128.0,
    )
    p_arr = params_dict_to_array(params)

    # Simulate
    dt = 1e-5
    t_final = float(t_data[-1])
    t_sim, U_sim = umr_averaged_speed_curve(p_arr, dt, t_final)

    # RMSE
    rmse = _compute_rmse(t_data, U_data, t_sim, U_sim)
    rel_rmse = rmse / peak_speed

    assert rel_rmse < 0.05, (
        f"1-mag RMSE/peak = {rel_rmse:.4f} (RMSE={rmse:.4f}, peak={peak_speed:.4f})"
    )


# ---------------------------------------------------------------------------
# MIME-VER-011: Prediction (2 and 3 magnets)
# ---------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MIME-VER-011",
    description="UMR ODE prediction RMSE < 15% for d2.8_2mag and d2.8_3mag",
    node_type="umr_ode",
    benchmark_type=BenchmarkType.REGRESSION,
    acceptance_criteria="2mag RMSE/peak < 0.15; 3mag RMSE/peak < 0.15",
    references=("deBoer2025",),
)
def test_umr_multimag_prediction():
    """Use 1-mag calibration to predict 2- and 3-magnet speed curves."""
    # Calibrate on 1-mag
    data_1 = _load_csv("d2.8_1mag.csv")
    base_params = fit_drag_coefficients(
        data_1, n_mag=1, m_single=1.07e-3, B=3e-3, f_step=128.0,
    )

    dt = 1e-5

    # --- 2-magnet prediction ---
    data_2 = _load_csv("d2.8_2mag.csv")
    t_data_2 = data_2[:, 0]
    U_data_2 = data_2[:, 1]
    peak_2 = float(np.max(U_data_2))

    params_2 = dict(base_params)
    params_2["n_mag"] = 2.0
    params_2["omega_field"] = 2.0 * math.pi * 181.0
    p2_arr = params_dict_to_array(params_2)

    t_sim_2, U_sim_2 = umr_averaged_speed_curve(p2_arr, dt, float(t_data_2[-1]))
    rmse_2 = _compute_rmse(t_data_2, U_data_2, t_sim_2, U_sim_2)
    rel_2 = rmse_2 / peak_2

    assert rel_2 < 0.15, (
        f"2-mag RMSE/peak = {rel_2:.4f} (RMSE={rmse_2:.4f}, peak={peak_2:.4f})"
    )

    # --- 3-magnet prediction ---
    data_3 = _load_csv("d2.8_3mag.csv")
    t_data_3 = data_3[:, 0]
    U_data_3 = data_3[:, 1]
    peak_3 = float(np.max(U_data_3))

    params_3 = dict(base_params)
    params_3["n_mag"] = 3.0
    params_3["omega_field"] = 2.0 * math.pi * 222.0
    p3_arr = params_dict_to_array(params_3)

    t_sim_3, U_sim_3 = umr_averaged_speed_curve(p3_arr, dt, float(t_data_3[-1]))
    rmse_3 = _compute_rmse(t_data_3, U_data_3, t_sim_3, U_sim_3)
    rel_3 = rmse_3 / peak_3

    assert rel_3 < 0.15, (
        f"3-mag RMSE/peak = {rel_3:.4f} (RMSE={rmse_3:.4f}, peak={peak_3:.4f})"
    )
