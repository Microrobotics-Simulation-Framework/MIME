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

    # Per-annotation offsets: position each label in clear space
    annot_offsets = {
        "d2.8_1mag": (15, -20),   # right and below — avoids ascending curves above
        "d2.8_2mag": (15, -25),   # right and below — avoids green curve above
        "d2.8_3mag": (15, -25),   # right and below — avoids title/top border
    }
    bbox_style = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

    for name in ["d2.8_1mag", "d2.8_2mag", "d2.8_3mag"]:
        r = results[name]
        ax.plot(r["freqs"], r["speeds"], color=colors[name], label=labels[name], linewidth=2)
        # Step-out marker with annotation
        ax.axvline(r["f_step_detected"], color=colors[name], linestyle="--", alpha=0.5)
        ax.plot(r["f_step_detected"], r["peak_speed"], "o",
                color=colors[name], markersize=8, zorder=5)
        ox, oy = annot_offsets[name]
        ax.annotate(
            f"$f_{{step}}$ = {r['f_step_detected']:.0f} Hz",
            xy=(r["f_step_detected"], r["peak_speed"]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=7.5, color=colors[name],
            bbox=bbox_style, zorder=6,
        )

    ax.set_xlabel("Actuation frequency [Hz]")
    ax.set_ylabel("Steady-state swimming speed [m/s]")
    ax.set_title("UMR swimming speed vs. frequency — de Boer et al. (2025) replication")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300)
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    colors = {"d2.8_1mag": "#1f77b4", "d2.8_2mag": "#ff7f0e", "d2.8_3mag": "#2ca02c"}
    mag_labels = {"d2.8_1mag": "1 magnet", "d2.8_2mag": "2 magnets", "d2.8_3mag": "3 magnets"}

    for i, name in enumerate(["d2.8_1mag", "d2.8_2mag", "d2.8_3mag"]):
        r = results[name]
        ax = axes[i]

        # Simulate transient at step-out frequency
        p_arr = params_dict_to_array(r["params"])
        dt = 1e-5
        t_final = 0.5
        t_sim, U_sim = umr_averaged_speed_curve(p_arr, dt, t_final)

        ax.plot(np.array(t_sim) * 1000, np.array(U_sim),
                color=colors[name], linewidth=2,
                label="Model" if i == 0 else None)
        ax.plot(r["data"][:, 0] * 1000, r["data"][:, 1],
                "o", color="black", markersize=4,
                label="Digitised data" if i == 0 else None)
        ax.set_xlabel("Time [ms]")
        if i == 0:
            ax.set_ylabel("Swimming speed [m/s]")
            ax.legend(fontsize=8)
        ax.set_title(mag_labels[name])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(0, None)
    plt.suptitle("d2.8 transient speed curves — de Boer et al. (2025) Fig. 12 replication",
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T1.4 — JAX autodiff: drag sensitivity for T2 confinement comparison
# ---------------------------------------------------------------------------

def compute_drag_sensitivity(results):
    """Autodiff gradients and drag-scaled speed curves.

    For each config:
    1. jax.grad of final speed through ODE integration (jax.lax.scan)
    2. Speed-vs-frequency at drag factors 0.8–1.3 (T2 prediction bands)
    """
    @jax.jit
    def _ode_final_speed(p_arr):
        _, U = umr_averaged_speed_curve(p_arr, 1e-5, 0.5)
        return U[-1]

    _ode_grad = jax.jit(jax.grad(_ode_final_speed))
    drag_factors = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    param_keys = ("omega_field", "n_mag", "m_single", "B",
                  "I_eff", "m_eff", "C_rot", "C_prop", "C_trans")

    sensitivity = {}
    for name, r in results.items():
        p = r["params"]
        p_arr = params_dict_to_array(p)
        grad_vec = np.array(_ode_grad(p_arr))

        curves = {}
        for alpha in drag_factors:
            p_scaled = dict(p)
            p_scaled["C_rot"] = p["C_rot"] * float(alpha)
            curves[float(alpha)] = sweep_steady_state(p_scaled, r["freqs"])

        sensitivity[name] = {
            "freqs": r["freqs"],
            "curves": curves,
            "f_step_baseline": r["f_step_detected"],
            "ode_grad": {k: float(grad_vec[i]) for i, k in enumerate(param_keys)},
        }

        f0 = r["f_step_detected"]
        print(f"\n{name} — T1.4 autodiff:")
        print(f"  ODE gradient (through jax.lax.scan):")
        for i, k in enumerate(param_keys):
            if abs(grad_vec[i]) > 1e-15:
                print(f"    dU/d{k:12s} = {grad_vec[i]:+.4e}")
        print(f"  Confinement prediction (C_rot * alpha):")
        for a in [1.1, 1.2, 1.3]:
            print(f"    alpha={a}: f_step {f0:.0f} -> {f0/a:.1f} Hz "
                  f"(delta = {f0/a - f0:+.1f} Hz)")

    return sensitivity


def plot_drag_sensitivity(results, sensitivity, save_path=None):
    """Plot speed-vs-frequency with confinement prediction bands."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    colors = {"d2.8_1mag": "#1f77b4", "d2.8_2mag": "#ff7f0e", "d2.8_3mag": "#2ca02c"}
    labels = {"d2.8_1mag": "1 magnet", "d2.8_2mag": "2 magnets", "d2.8_3mag": "3 magnets"}

    for i, name in enumerate(["d2.8_1mag", "d2.8_2mag", "d2.8_3mag"]):
        ax = axes[i]
        s = sensitivity[name]
        freqs = s["freqs"]
        c = colors[name]

        ax.fill_between(freqs, s["curves"][0.8], s["curves"][1.2],
                         alpha=0.15, color=c, label=r"$\pm$20% $C_{rot}$")
        ax.fill_between(freqs, s["curves"][0.9], s["curves"][1.1],
                         alpha=0.25, color=c, label=r"$\pm$10% $C_{rot}$")
        ax.plot(freqs, s["curves"][1.0], color=c, linewidth=2,
                label=r"Unconfined ($\alpha$=1)")

        f0 = s["f_step_baseline"]
        for alpha, ls, lbl in [
            (1.0, "-", r"$f_{step}$"),
            (1.1, "--", r"$f_{step}$ at $\alpha$=1.1"),
            (1.2, ":", r"$f_{step}$ at $\alpha$=1.2"),
        ]:
            ax.axvline(f0 / alpha, color=c, linestyle=ls, alpha=0.4, linewidth=1,
                       label=lbl if i == 0 else None)

        ax.set_xlabel("Actuation frequency [Hz]")
        if i == 0:
            ax.set_ylabel("Swimming speed [m/s]")
        ax.set_title(f"d2.8, {labels[name]}")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(True, alpha=0.3)

    # Shared y-limit: analytical peak at lowest drag factor (α=0.8)
    # U_peak = (C_prop/C_trans) * omega_so, omega_so = n*m*B / (C_rot*α)
    y_max = max(
        r["peak_speed"] / 0.8 for r in results.values()  # peak scales as 1/α
    )
    axes[0].set_xlim(0, 280)
    axes[0].set_ylim(0, y_max * 1.05)

    fig.suptitle(
        r"T1.4: Drag sensitivity — unconfined baseline with prediction bands for T2"
        "\n"
        r"(bands show effect of rotational drag scaling $\alpha$ on step-out and post-step-out speed)",
        fontsize=10,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T1.5 — vmap Pareto surface over parameter space
# ---------------------------------------------------------------------------

def _make_speed_fn(n_mag, m_single, B, C_rot, C_prop, C_trans):
    """Factory for a JIT-able speed function with fixed physical params."""
    def _speed(freq, alpha):
        omega = 2.0 * jnp.pi * freq
        omega_so = n_mag * m_single * B / (C_rot * alpha)
        return (C_prop / C_trans) * _mean_angular_velocity(omega, omega_so)
    return _speed


def compute_pareto_surface(results):
    """Sweep (n_mag, freq) and (freq, drag_factor) via jax.vmap."""
    base_p = results["d2.8_1mag"]["params"]

    # Continuous n_mag sweep (1-mag base drag coefficients)
    n_mags = jnp.linspace(0.5, 4.0, 50)
    freqs_p = jnp.linspace(10.0, 300.0, 200)

    def _speed_nmag(n_mag, freq):
        omega = 2.0 * jnp.pi * freq
        omega_so = n_mag * base_p["m_single"] * base_p["B"] / base_p["C_rot"]
        return (base_p["C_prop"] / base_p["C_trans"]) * _mean_angular_velocity(
            omega, omega_so)

    nmag_surface = jax.jit(jax.vmap(
        jax.vmap(_speed_nmag, in_axes=(None, 0)),
        in_axes=(0, None),
    ))(n_mags, freqs_p)

    # 2D speed(freq, drag_factor) per config
    freqs_2d = jnp.linspace(10.0, 280.0, 200)
    alphas_2d = jnp.linspace(0.7, 1.5, 80)

    speed_grids = {}
    for name, r in results.items():
        p = r["params"]
        fn = _make_speed_fn(
            p["n_mag"], p["m_single"], p["B"],
            p["C_rot"], p["C_prop"], p["C_trans"],
        )
        grid = jax.jit(jax.vmap(
            jax.vmap(fn, in_axes=(0, None)),
            in_axes=(None, 0),
        ))(freqs_2d, alphas_2d)
        speed_grids[name] = np.array(grid)

    print("\nT1.5: vmap parameter sweeps computed")
    print(f"  n_mag surface: {len(n_mags)} x {len(freqs_p)} = "
          f"{len(n_mags)*len(freqs_p)} points")
    print(f"  Per-config grids: {len(alphas_2d)} x {len(freqs_2d)} = "
          f"{len(alphas_2d)*len(freqs_2d)} points each")

    return {
        "n_mags": np.array(n_mags),
        "freqs_p": np.array(freqs_p),
        "nmag_surface": np.array(nmag_surface),
        "freqs_2d": np.array(freqs_2d),
        "alphas_2d": np.array(alphas_2d),
        "speed_grids": speed_grids,
        "paper_configs": {
            name: {"n_mag": r["params"]["n_mag"],
                   "f_step": r["f_step_detected"],
                   "U_peak": r["peak_speed"]}
            for name, r in results.items()
        },
    }


def plot_pareto_surface(pareto, save_path=None):
    """Plot parameter space surfaces."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: speed surface over (freq, n_mag) — unconfined baseline
    ax = axes[0]
    cf = ax.contourf(pareto["freqs_p"], pareto["n_mags"],
                     pareto["nmag_surface"], levels=10, cmap="viridis")
    ax.contour(pareto["freqs_p"], pareto["n_mags"],
               pareto["nmag_surface"], levels=10,
               colors="white", linewidths=0.3, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="Speed [m/s]")

    markers = {"d2.8_1mag": "o", "d2.8_2mag": "s", "d2.8_3mag": "D"}
    for name, cfg in pareto["paper_configs"].items():
        ax.plot(cfg["f_step"], cfg["n_mag"], markers[name],
                color="white", markersize=10, markeredgecolor="black",
                markeredgewidth=1.5, zorder=5,
                label=name.replace("_", " "))

    ax.set_xlabel("Actuation frequency [Hz]")
    ax.set_ylabel("Magnet count (continuous)")
    ax.set_title("Speed(freq, n_mag) — unconfined baseline")
    ax.legend(fontsize=7)

    # Right: speed(freq, drag_factor) for 3-mag config
    ax = axes[1]
    grid = pareto["speed_grids"]["d2.8_3mag"]
    cf2 = ax.contourf(pareto["freqs_2d"], pareto["alphas_2d"],
                      grid, levels=20, cmap="viridis")
    ax.contour(pareto["freqs_2d"], pareto["alphas_2d"],
               grid, levels=20, colors="white", linewidths=0.3, alpha=0.5)
    plt.colorbar(cf2, ax=ax, label="Speed [m/s]")

    cfg = pareto["paper_configs"]["d2.8_3mag"]
    ax.plot(cfg["f_step"], 1.0, "w*", markersize=22, zorder=5,
            markeredgecolor="black", markeredgewidth=1.5)
    ax.annotate(
        f"Unconfined step-out\n({cfg['f_step']:.0f} Hz, "
        r"$\alpha$=1)",
        xy=(cfg["f_step"], 1.0), xytext=(-80, 40),
        textcoords="offset points", fontsize=7, color="white",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
    )
    ax.axhline(1.0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)

    # Confinement ratio on right y-axis
    # Mapping: Couette correction alpha = 1/(1 - r^2), so r = sqrt(1 - 1/alpha)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    aticks = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
    rlabels = [f"{np.sqrt(max(1 - 1/a, 0)):.2f}" for a in aticks]
    ax2.set_yticks(aticks)
    ax2.set_yticklabels(rlabels, fontsize=7)
    ax2.set_ylabel(r"$R_{umr}/R_{vessel}$")

    ax.set_xlabel("Actuation frequency [Hz]")
    ax.set_ylabel(r"Drag factor $\alpha$ (rotational)")
    ax.set_title("d2.8, 3mag: speed(freq, confinement)")
    ax.text(0.02, 0.02, r"Couette: $\alpha = 1/(1 - r^2)$",
            transform=ax.transAxes, fontsize=7, color="white", alpha=0.8)

    fig.suptitle("T1.5: Parameter space sweep via jax.vmap", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    results = replicate_figure_12()

    fig_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "docs", "validation", "umr_deboer2025", "figures",
    )
    os.makedirs(fig_dir, exist_ok=True)

    # T1.3 — reproduce speed curves
    plot_figure_12(results, os.path.join(fig_dir, "deboer_fig12_replicated.png"))
    plot_transient_curves(results, os.path.join(fig_dir, "deboer_fig12_transients.png"))

    # T1.4 — autodiff drag sensitivity
    sensitivity = compute_drag_sensitivity(results)
    plot_drag_sensitivity(results, sensitivity,
                          os.path.join(fig_dir, "deboer_fig12_drag_sensitivity.png"))

    # T1.5 — vmap parameter space
    pareto = compute_pareto_surface(results)
    plot_pareto_surface(pareto, os.path.join(fig_dir, "deboer_fig12_pareto.png"))
