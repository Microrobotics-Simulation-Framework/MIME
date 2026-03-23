#!/usr/bin/env python3
"""Replicate de Boer et al. (2025) Figure 12 speed-vs-frequency curves.

For each d2.8 UMR configuration (1, 2, 3 magnets):
1. Calibrate drag coefficients from digitised speed-vs-time data
2. Sweep actuation frequency from 10 Hz to 1.2 * f_step
3. Extract steady-state swimming speed at each frequency
4. Detect step-out from the speed curve peak

The d2.1 configurations (75% geometry scaling) are deferred — the drag
coefficient scaling law is not established. See MIME-VER-012.

Reference: de Boer, M.C.J. et al. (2025). Appl. Phys. Rev. 12, 011416.
"""

import os
import math

import numpy as np
import jax
import jax.numpy as jnp

from mime.nodes.robot.umr_ode import (
    fit_drag_coefficients,
    umr_averaged_speed_curve,
    _mean_angular_velocity,
    params_dict_to_array,
    unpack_params,
)


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "docs", "validation", "umr_deboer2025", "deboer_fig12_digitised",
)


def _load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    return np.loadtxt(path, delimiter=",", skiprows=1)


# ---------------------------------------------------------------------------
# Configuration table — d2.8 only
# ---------------------------------------------------------------------------

D28_CONFIGS = [
    {"name": "d2.8_1mag", "n_mag": 1, "f_step": 128.0, "csv": "d2.8_1mag.csv"},
    {"name": "d2.8_2mag", "n_mag": 2, "f_step": 181.0, "csv": "d2.8_2mag.csv"},
    {"name": "d2.8_3mag", "n_mag": 3, "f_step": 222.0, "csv": "d2.8_3mag.csv"},
]

M_SINGLE = 1.07e-3  # A*m^2
B_FIELD = 3e-3       # T


# ---------------------------------------------------------------------------
# Steady-state speed at a given frequency (averaged model)
# ---------------------------------------------------------------------------

def steady_state_speed(params_dict, freq):
    """Compute U_ss at a given frequency using the averaged model.

    Below step-out: U_ss = (C_prop/C_trans) * omega_field
    Above step-out: U_ss = (C_prop/C_trans) * <Omega>(omega_field)
    where <Omega> is from the Adler equation.
    """
    omega = 2.0 * math.pi * freq
    n_mag = params_dict["n_mag"]
    m_single = params_dict["m_single"]
    B = params_dict["B"]
    C_rot = params_dict["C_rot"]
    C_prop = params_dict["C_prop"]
    C_trans = params_dict["C_trans"]

    omega_so = n_mag * m_single * B / C_rot
    Omega_avg = float(_mean_angular_velocity(
        jnp.array(omega), jnp.array(omega_so),
    ))
    return (C_prop / C_trans) * Omega_avg


def sweep_steady_state(params_dict, freqs):
    """Sweep frequencies and return steady-state speeds.

    Uses the analytical steady-state from the averaged model (no ODE
    integration needed — the translational ODE always converges to
    U_ss = C_prop/C_trans * <Omega>).
    """
    omega_arr = jnp.array(2.0 * math.pi * np.array(freqs))
    n_mag = params_dict["n_mag"]
    omega_so = n_mag * params_dict["m_single"] * params_dict["B"] / params_dict["C_rot"]

    Omega_avg = jax.vmap(
        lambda w: _mean_angular_velocity(w, jnp.array(omega_so))
    )(omega_arr)

    ratio = params_dict["C_prop"] / params_dict["C_trans"]
    return np.array(ratio * Omega_avg)


def detect_step_out_from_curve(freqs, speeds):
    """Detect step-out frequency as the frequency of peak speed."""
    idx = np.argmax(speeds)
    return freqs[idx]


# ---------------------------------------------------------------------------
# Main replication
# ---------------------------------------------------------------------------

def replicate_figure_12():
    """Run the full replication and return results dict."""
    results = {}

    for cfg in D28_CONFIGS:
        data = _load_csv(cfg["csv"])

        # Calibrate per-configuration (each has its own step-out freq)
        params = fit_drag_coefficients(
            data,
            n_mag=cfg["n_mag"],
            m_single=M_SINGLE,
            B=B_FIELD,
            f_step=cfg["f_step"],
        )

        # Frequency sweep: 10 Hz to 1.2 * f_step, ~2 Hz steps
        f_max = 1.2 * cfg["f_step"]
        freqs = np.arange(10.0, f_max, 2.0)

        speeds = sweep_steady_state(params, freqs)
        f_step_detected = detect_step_out_from_curve(freqs, speeds)

        results[cfg["name"]] = {
            "params": params,
            "freqs": freqs,
            "speeds": speeds,
            "f_step_paper": cfg["f_step"],
            "f_step_detected": f_step_detected,
            "peak_speed": float(np.max(speeds)),
            "data": data,
        }

        print(
            f"{cfg['name']}: f_step(paper)={cfg['f_step']:.0f} Hz, "
            f"f_step(detected)={f_step_detected:.0f} Hz, "
            f"U_peak={np.max(speeds):.3f} m/s"
        )

    return results


def plot_figure_12(results, save_path=None):
    """Plot speed-vs-frequency curves for all d2.8 configurations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = {"d2.8_1mag": "#1f77b4", "d2.8_2mag": "#ff7f0e", "d2.8_3mag": "#2ca02c"}
    labels = {"d2.8_1mag": "d2.8, 1 mag", "d2.8_2mag": "d2.8, 2 mag", "d2.8_3mag": "d2.8, 3 mag"}

    for name in ["d2.8_1mag", "d2.8_2mag", "d2.8_3mag"]:
        r = results[name]
        ax.plot(r["freqs"], r["speeds"], color=colors[name], label=labels[name], linewidth=2)
        # Step-out marker
        ax.axvline(r["f_step_detected"], color=colors[name], linestyle="--", alpha=0.5)
        ax.plot(r["f_step_detected"], r["peak_speed"], "o",
                color=colors[name], markersize=8, zorder=5)

    ax.set_xlabel("Actuation frequency [Hz]")
    ax.set_ylabel("Steady-state swimming speed [m/s]")
    ax.set_title("UMR swimming speed vs. frequency — de Boer et al. (2025) replication")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    # Note on d2.1 deferral
    ax.text(
        0.98, 0.02,
        "d2.1 configs deferred (MIME-VER-012):\nno validated geometry scaling law",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        style="italic", alpha=0.6,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_transient_curves(results, save_path=None):
    """Plot speed-vs-time transient curves with digitised data overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {"d2.8_1mag": "#1f77b4", "d2.8_2mag": "#ff7f0e", "d2.8_3mag": "#2ca02c"}

    for i, name in enumerate(["d2.8_1mag", "d2.8_2mag", "d2.8_3mag"]):
        r = results[name]
        ax = axes[i]

        # Simulate transient at step-out frequency
        p_arr = params_dict_to_array(r["params"])
        dt = 1e-5
        t_final = 0.5
        t_sim, U_sim = umr_averaged_speed_curve(p_arr, dt, t_final)

        ax.plot(np.array(t_sim) * 1000, np.array(U_sim),
                color=colors[name], linewidth=2, label="Model")
        ax.plot(r["data"][:, 0] * 1000, r["data"][:, 1],
                "o", color="black", markersize=4, label="Digitised data")
        ax.set_xlabel("Time [ms]")
        if i == 0:
            ax.set_ylabel("Swimming speed [m/s]")
        ax.set_title(name.replace("_", ", "))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Transient speed curves — de Boer et al. (2025) Fig. 12 replication",
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    results = replicate_figure_12()

    fig_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "docs", "validation", "umr_deboer2025", "figures",
    )
    os.makedirs(fig_dir, exist_ok=True)

    plot_figure_12(results, os.path.join(fig_dir, "deboer_fig12_replicated.png"))
    plot_transient_curves(results, os.path.join(fig_dir, "deboer_fig12_transients.png"))
