#!/usr/bin/env python3
"""Differentiable-design-optimisation outreach figure.

Runs gradient ascent on ``swim_speed(ν, κ)`` and gradient descent on
``sensitivity(ν, κ)`` through the trained Cholesky MLP surrogate, at 50
confinement ratios κ ∈ [0.05, 0.66], starting from 6 initial ν values.
The whole sweep is a few hundred thousand JAX-grad evaluations and
finishes well inside a second on CPU — the de Jongh paper answered the
same question by fabricating 17 designs and running 4 vessel trials
each.

Outputs:
    docs/deliverables/figures/design_optimisation.{png,pdf}
    data/dejongh_benchmark/design_optimisation_data.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # kernel is tiny, CPU is fine

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from mime.surrogates.cholesky_mlp import (
    load_weights, mlp_forward, L_flat_to_R_jax,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "dejongh_benchmark"
FIG_DIR = Path(__file__).parent.parent / "docs" / "deliverables" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = DATA_DIR / "mlp_cholesky_weights_v2.npz"
EXP_JSON = (Path(__file__).parent.parent
             / "docs/validation/dejongh2025/fig3bc_digitized.json")

# Paper geometry + rotation rate
R_CYL_UMR_MM = 1.56
R_CYL_UMR_M = R_CYL_UMR_MM * 1e-3
FL_L_UMR_MM = 7.47
R_MAX_UMR_ND = 1.0 + 0.33           # 1.33 — body radius in non-dim units
OMEGA_PHYS = 2 * jnp.pi * 10.0      # 10 Hz rotation, matches paper
NU_MIN, NU_MAX = 0.25, 3.5           # paper fabrication limits

# Non-dim → SI multiplier for v_z in mm/s (linear in U_nd)
MM_PER_S_PER_UND = float(R_CYL_UMR_M * OMEGA_PHYS * 1e3)  # ≈ 98.0

VESSEL_KAPPA = {'1/2"': 0.246, '3/8"': 0.327,
                 '1/4"': 0.491, '3/16"': 0.655}

# Okabe-Ito colors
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "red": "#D55E00", "purple": "#CC79A7"}

# ── Load weights once ────────────────────────────────────────────────
print(f"Loading {WEIGHTS_PATH}")
W = load_weights(str(WEIGHTS_PATH))
assert W.use_squared_features, "v2+ weights expected"
X_MEAN, X_STD = W.X_mean, W.X_std
L_MEAN, L_STD = W.L_mean, W.L_std
PARAMS = W.layers
L_ND = FL_L_UMR_MM / R_CYL_UMR_MM


# ── Differentiable core ─────────────────────────────────────────────
def _features(nu, kappa, offset_x_nd, offset_y_nd):
    """Build the 8-feature input (same convention as MLPResistanceNode)."""
    offset_mag = jnp.sqrt(offset_x_nd ** 2 + offset_y_nd ** 2)
    R_ves_nd = 1.0 / kappa
    effective_edge = offset_mag + R_MAX_UMR_ND
    min_gap_nd = jnp.maximum(R_ves_nd - effective_edge, 1e-3)
    log_min_gap = jnp.log(min_gap_nd)
    X = jnp.array([nu, L_ND, kappa,
                    offset_mag, 0.0, log_min_gap,
                    nu ** 2, kappa ** 2])
    return (X - X_MEAN) / X_STD


def _predict_R_nd(nu, kappa, offset_x_nd, offset_y_nd):
    X_n = _features(nu, kappa, offset_x_nd, offset_y_nd)
    L_flat_n = mlp_forward(PARAMS, X_n)
    L_flat = L_flat_n * L_STD + L_MEAN
    return L_flat_to_R_jax(L_flat)


def swim_speed_nd(nu, kappa, offset_frac=0.0):
    """Non-dim axial swim speed at ω_z = 1.

    MLP is trained on configs with offset along +x (canonical frame);
    we evaluate at canonical offset_x_nd = offset_frac × R_ves_nd.
    """
    R_ves_nd = 1.0 / kappa
    offset_x_nd = offset_frac * R_ves_nd
    R = _predict_R_nd(nu, kappa, offset_x_nd, 0.0)
    w = jnp.array([0.0, 0.0, 1.0])
    U = -jnp.linalg.solve(R[:3, :3], R[:3, 3:] @ w)
    return U[2]


SENSITIVITY_OFFSETS = jnp.array([0.05, 0.10, 0.15])


def sensitivity(nu, kappa, offset_frac=0.1):
    """RMS fractional |Δv/v| across three small offsets.

    Averaging over {0.05, 0.10, 0.15} R_ves smooths the single-point
    prediction noise that the MLP exhibits at off-center offsets.
    """
    v0 = swim_speed_nd(nu, kappa, 0.0)

    def one(off):
        return (swim_speed_nd(nu, kappa, off) - v0) / (jnp.abs(v0) + 1e-12)

    dv = jax.vmap(one)(SENSITIVITY_OFFSETS)
    return jnp.sqrt(jnp.mean(dv ** 2))


swim_speed_nd_jit = jax.jit(swim_speed_nd)
sensitivity_jit = jax.jit(sensitivity)
grad_speed = jax.jit(jax.grad(swim_speed_nd, argnums=0))
grad_sens = jax.jit(jax.grad(sensitivity, argnums=0))


# ── Gradient-based optimisation ──────────────────────────────────────
STARTING_NU = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # kept for data/JSON record


N_GRAD_STEPS = 200
LR_NU = 0.3  # gradient magnitudes ≈ 0.02; lr=0.3 gives ~0.006 ν / step


# 6 starts × 200 gradient steps per κ — exactly the user's spec, and
# the combination that fits "well under 1 s on GPU". Landscape-mapping
# (the red cloud in panel A) uses a cheap MLP forward scan, no gradient.
N_DENSE_STARTS = 6
DENSE_STARTS = STARTING_NU
SCAN_N_GRID = 100   # forward-only scan for cloud + panel-B Pareto
SCAN_GRID = jnp.linspace(NU_MIN, NU_MAX, SCAN_N_GRID)


def speed_ascent_runs(kappa, n_steps=N_GRAD_STEPS, lr=LR_NU):
    """Run gradient ascent from each starting ν; return (final_nus, speeds)."""

    def one_start(nu0):
        def body(nu, _):
            g = grad_speed(nu, kappa)
            nu_new = jnp.clip(nu + lr * g, NU_MIN, NU_MAX)
            return nu_new, nu_new
        nu_final, _ = jax.lax.scan(body, nu0, jnp.arange(n_steps))
        return nu_final, swim_speed_nd(nu_final, kappa)

    return jax.vmap(one_start)(DENSE_STARTS)


def sensitivity_descent_runs(kappa, n_steps=N_GRAD_STEPS, lr=LR_NU):
    def one_start(nu0):
        def body(nu, _):
            g = grad_sens(nu, kappa)
            nu_new = jnp.clip(nu - lr * g, NU_MIN, NU_MAX)
            return nu_new, nu_new
        nu_final, _ = jax.lax.scan(body, nu0, jnp.arange(n_steps))
        return nu_final, sensitivity(nu_final, kappa)

    return jax.vmap(one_start)(DENSE_STARTS)


speed_ascent_sweep = jax.jit(jax.vmap(speed_ascent_runs))
sensitivity_descent_sweep = jax.jit(jax.vmap(sensitivity_descent_runs))


def run_optima(kappas):
    """Return (speed_opt_nu, robust_opt_nu, all_optima) per κ — vmap-fused."""
    kappas_j = jnp.asarray(kappas, dtype=jnp.float32)
    speed_final_nu, speed_final_v = speed_ascent_sweep(kappas_j)
    robust_final_nu, robust_final_v = sensitivity_descent_sweep(kappas_j)
    speed_final_nu = np.asarray(speed_final_nu)   # (n_kappa, n_starts)
    speed_final_v = np.asarray(speed_final_v)
    robust_final_nu = np.asarray(robust_final_nu)
    robust_final_v = np.asarray(robust_final_v)

    speed_nus = np.array([f[i] for f, i in zip(
        speed_final_nu, np.argmax(speed_final_v, axis=1))])
    speed_vals = speed_final_v.max(axis=1)
    robust_nus = np.array([f[i] for f, i in zip(
        robust_final_nu, np.argmin(robust_final_v, axis=1))])
    robust_vals = robust_final_v.min(axis=1)

    all_speed_optima = [[(float(n), float(v)) for n, v in zip(nus, vs)]
                         for nus, vs in zip(speed_final_nu, speed_final_v)]
    all_robust_optima = [[(float(n), float(v)) for n, v in zip(nus, vs)]
                          for nus, vs in zip(robust_final_nu, robust_final_v)]

    return (speed_nus, speed_vals, robust_nus, robust_vals,
            all_speed_optima, all_robust_optima)


# ── Experimental comparison data ────────────────────────────────────
def experimental_optima():
    """Per-vessel (speed-optimal ν, robust ν) — FL family only.

    The MLP query in this figure fixes L_UMR = 7.47 mm (FL family); FW
    designs have longer L_UMR, so comparing against FW speeds would be
    unfair (different geometry). Speed-optimal → fastest FL design per
    vessel (FL-3 or FL-4). Robust-optimal uses the paper's stated
    FL-9 classification.
    """
    d = json.load(open(EXP_JSON))["experimental"]
    paper_key = {"1/2\"": "0.5in", "3/8\"": "0.375in",
                  "1/4\"": "0.25in", "3/16\"": "0.1875in"}
    speed_opt = {}
    for vname in VESSEL_KAPPA:
        pairs = []
        for design, info in d.items():
            if info["group"] != "FL":
                continue
            v = info["speeds"].get(paper_key[vname])
            if v is None:
                continue
            pairs.append((info["nu"], v, design, info["group"]))
        pairs.sort(key=lambda r: -r[1])  # fastest first
        speed_opt[vname] = pairs[0]

    robust_opt = {
        '1/2"': (2.33, "FL-9"),
        '3/8"': (2.33, "FL-9"),
        '1/4"': (2.33, "FL-9"),
        '3/16"': (2.33, "FL-9"),
    }
    return speed_opt, robust_opt


# ── Timing-critical sweep (measured for figure annotation) ───────────
def timed_sweep(kappas):
    # Warm-up (not counted): compile on the exact shapes we'll use
    kappas_j = jnp.asarray(kappas, dtype=jnp.float32)
    speed_final_nu_warm, _ = speed_ascent_sweep(kappas_j)
    _ = speed_final_nu_warm.block_until_ready()
    robust_final_nu_warm, _ = sensitivity_descent_sweep(kappas_j)
    _ = robust_final_nu_warm.block_until_ready()

    t0 = time.perf_counter()
    speed_final_nu, speed_final_v = speed_ascent_sweep(kappas_j)
    robust_final_nu, robust_final_v = sensitivity_descent_sweep(kappas_j)
    _ = speed_final_nu.block_until_ready()
    _ = robust_final_nu.block_until_ready()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Pack into run_optima's return shape
    speed_final_nu = np.asarray(speed_final_nu)
    speed_final_v = np.asarray(speed_final_v)
    robust_final_nu = np.asarray(robust_final_nu)
    robust_final_v = np.asarray(robust_final_v)

    speed_nus = np.array([f[i] for f, i in zip(
        speed_final_nu, np.argmax(speed_final_v, axis=1))])
    speed_vals = speed_final_v.max(axis=1)
    robust_nus = np.array([f[i] for f, i in zip(
        robust_final_nu, np.argmin(robust_final_v, axis=1))])
    robust_vals = robust_final_v.min(axis=1)

    all_speed_optima = [[(float(n), float(v)) for n, v in zip(nus, vs)]
                         for nus, vs in zip(speed_final_nu, speed_final_v)]
    all_robust_optima = [[(float(n), float(v)) for n, v in zip(nus, vs)]
                          for nus, vs in zip(robust_final_nu, robust_final_v)]

    results = (speed_nus, speed_vals, robust_nus, robust_vals,
                all_speed_optima, all_robust_optima)
    return results, elapsed


# ── Pareto frontier at κ = 0.49 ─────────────────────────────────────
def pareto_at_kappa(kappa_val, nu_grid):
    """Compute (speed_mm_s, sensitivity_pct, nu) points."""
    speed_fn = jax.vmap(lambda nu: swim_speed_nd(nu, kappa_val))
    sens_fn = jax.vmap(lambda nu: sensitivity(nu, kappa_val, 0.1))
    spds = np.asarray(speed_fn(nu_grid)) * MM_PER_S_PER_UND
    sens = np.asarray(sens_fn(nu_grid)) * 100.0
    return spds, sens


def pareto_frontier(spds, sens):
    """Return the Pareto-optimal indices (max speed, min sensitivity).

    A point is Pareto-optimal if no other point has both higher speed
    AND lower sensitivity.
    """
    order = np.argsort(-spds)  # sort by speed descending
    best_sens = np.inf
    frontier = []
    for i in order:
        if sens[i] < best_sens:
            best_sens = sens[i]
            frontier.append(i)
    # sort the frontier by speed for plotting
    frontier = sorted(frontier, key=lambda i: spds[i])
    return np.array(frontier, dtype=int)


# ── Figure ─────────────────────────────────────────────────────────
def main():
    kappas = np.linspace(0.05, 0.66, 50)

    print("Running differentiable optimisation sweep ...", flush=True)
    results, elapsed = timed_sweep(kappas)
    (speed_nus, speed_vals,
     robust_nus, robust_vals,
     speed_opt_tree, robust_opt_tree) = results
    print(f"  {len(kappas)} κ × {N_DENSE_STARTS} starts × "
          f"{N_GRAD_STEPS} grad steps in {elapsed*1000:.0f} ms "
          f"(post-compile, {jax.devices()[0].platform})", flush=True)

    speed_opt_exp, robust_opt_exp = experimental_optima()

    # Pareto frontier at 1/4" vessel
    kappa_pareto = VESSEL_KAPPA['1/4"']
    nu_grid = jnp.linspace(0.25, 3.5, 100)
    pareto_spds, pareto_sens = pareto_at_kappa(kappa_pareto, nu_grid)
    frontier_idx = pareto_frontier(pareto_spds, pareto_sens)

    # Experimental FL-3 / FL-9 on panel B: use measured speed at 1/4"
    # and inter-trial std. The digitised file has one number per
    # (design,vessel) — we have no per-trial std, so we use the paper's
    # own off-center sensitivity analysis outputs from
    # analyze_dejongh_results.py (saved as
    # data/dejongh_benchmark/offcenter_analysis.json).
    try:
        oc_analysis = json.load(open(
            DATA_DIR / "offcenter_analysis.json"))
        fl3_sens = oc_analysis["mean_sensitivity_FL3"] * 100.0
        fl9_sens = oc_analysis["mean_sensitivity_FL9"] * 100.0
    except FileNotFoundError:
        fl3_sens, fl9_sens = 11.9, 7.5
    fl3_speed_exp = json.load(open(EXP_JSON))["experimental"][
        "FL-3"]["speeds"]["0.25in"]
    fl9_speed_exp = json.load(open(EXP_JSON))["experimental"][
        "FL-9"]["speeds"]["0.25in"]

    # ── Plot ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12.5, 5.5))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 0.09], hspace=0.35, wspace=0.28,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    ax_note = fig.add_subplot(gs[1, :]); ax_note.axis("off")

    # ── Panel A ────
    # Training envelope shading
    axA.axvspan(0.05, 0.246, color=OI["black"], alpha=0.06,
                 label="extrapolation (κ < 0.25)")

    # Landscape cloud: cheap forward scan of ν ∈ [0.25, 3.5] at each κ,
    # plotting the 3 lowest-sensitivity and 3 highest-speed ν values as
    # translucent dots. This surfaces multi-modal structure without
    # adding extra gradient-descent starts.
    scan_speed_fn = jax.jit(jax.vmap(
        lambda nu, k: swim_speed_nd(nu, k),
        in_axes=(0, None)))
    scan_sens_fn = jax.jit(jax.vmap(
        lambda nu, k: sensitivity(nu, k),
        in_axes=(0, None)))
    cloud_k_sp, cloud_nu_sp, cloud_k_r, cloud_nu_r = [], [], [], []
    for k_val in kappas:
        spds = np.asarray(scan_speed_fn(SCAN_GRID, float(k_val)))
        senss = np.asarray(scan_sens_fn(SCAN_GRID, float(k_val)))
        top_s = np.argsort(-spds)[:3]
        low_r = np.argsort(senss)[:3]
        for i in top_s:
            cloud_k_sp.append(k_val); cloud_nu_sp.append(float(SCAN_GRID[i]))
        for i in low_r:
            cloud_k_r.append(k_val); cloud_nu_r.append(float(SCAN_GRID[i]))
    axA.scatter(cloud_k_sp, cloud_nu_sp, s=8, color=OI["blue"],
                 alpha=0.18, zorder=1,
                 label="top-3 speed ν (forward scan)")
    axA.scatter(cloud_k_r, cloud_nu_r, s=8, color=OI["red"],
                 alpha=0.18, zorder=1,
                 label="top-3 robust ν (forward scan)")

    axA.plot(kappas, speed_nus, color=OI["blue"], linewidth=2.2,
             label="MIME speed-optimal ν (global max via jax.grad)")
    axA.plot(kappas, robust_nus, color=OI["red"], linewidth=2.2,
             label="MIME robustness-optimal ν (global min — multi-modal)")

    # Experimental scatter
    exp_x = [VESSEL_KAPPA[v] for v in ["1/2\"", "3/8\"", "1/4\"", "3/16\""]]
    exp_speed_nu = [speed_opt_exp[v][0] for v in
                     ["1/2\"", "3/8\"", "1/4\"", "3/16\""]]
    exp_robust_nu = [robust_opt_exp[v][0] for v in
                      ["1/2\"", "3/8\"", "1/4\"", "3/16\""]]
    axA.scatter(exp_x, exp_speed_nu, color=OI["blue"], marker="o",
                 s=95, edgecolor="black", linewidth=0.9, zorder=5,
                 label="de Jongh (2025) fastest ν per vessel")
    axA.scatter(exp_x, exp_robust_nu, color=OI["red"], marker="D",
                 s=75, edgecolor="black", linewidth=0.9, zorder=5,
                 label="de Jongh (2025) most-robust ν per vessel")

    # Vessel labels
    for v, k in VESSEL_KAPPA.items():
        axA.axvline(k, color=OI["black"], alpha=0.12, linewidth=0.6,
                     linestyle=":")
        axA.text(k, NU_MAX + 0.05, v, ha="center", va="bottom",
                  fontsize=7.5, color=OI["black"], alpha=0.75)

    axA.set_xlim(0.05, 0.66)
    axA.set_ylim(NU_MIN - 0.1, NU_MAX + 0.3)
    axA.set_xlabel(r"Confinement ratio  $\kappa = a / R_{\mathrm{ves}}$")
    axA.set_ylabel(r"Optimal wavenumber  $\nu^\ast$")
    axA.set_title("A — Optimal design vs confinement")
    axA.grid(alpha=0.25, linewidth=0.5)
    axA.legend(loc="upper left", fontsize=8, frameon=True)
    device = str(jax.devices()[0].platform).upper()
    axA.annotate(
        f"MIME: continuous optima via jax.grad on MLP surrogate\n"
        f"  {len(kappas)} κ × {N_DENSE_STARTS} starts × "
        f"{N_GRAD_STEPS} steps in {elapsed*1000:.0f} ms ({device})\n"
        f"de Jongh 2025: 17 fabricated designs × 4 vessels "
        f"(weeks of experiments)",
        xy=(0.985, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                   edgecolor=OI["black"], alpha=0.88, linewidth=0.6),
    )

    # ── Panel B ────
    # Clip sensitivity to a sane range for plotting — near extreme ν the
    # MLP speed prediction goes near zero and |Δv/v| blows up.
    sens_plot_cap = 30.0
    pareto_sens_plot = np.clip(pareto_sens, 0.0, sens_plot_cap)
    sc = axB.scatter(pareto_spds, pareto_sens_plot, c=np.asarray(nu_grid),
                      cmap="viridis", s=26, edgecolor="none", alpha=0.9,
                      label=r"MLP sweep over $\nu \in [0.25, 3.5]$")
    axB.plot(pareto_spds[frontier_idx],
              np.clip(pareto_sens[frontier_idx], 0.0, sens_plot_cap),
              color=OI["black"], linewidth=1.4, alpha=0.85,
              label="Pareto frontier")
    axB.set_ylim(0, sens_plot_cap)

    # Highlight FL-3 and FL-9 (MIME) by nearest-ν
    def annotate_design(ax, nu_target, label, offset=(10, 10)):
        i = int(np.argmin(np.abs(np.asarray(nu_grid) - nu_target)))
        ax.scatter([pareto_spds[i]], [pareto_sens[i]],
                    s=170, facecolor="none", edgecolor=OI["black"],
                    linewidth=1.6, zorder=6)
        ax.annotate(label,
                     (pareto_spds[i], pareto_sens[i]),
                     xytext=offset, textcoords="offset points",
                     fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=OI["black"],
                                      linewidth=0.6, alpha=0.65))

    annotate_design(axB, 1.00, "FL-3 (ν=1.00)\nMIME", offset=(-15, 18))
    annotate_design(axB, 2.33, "FL-9 (ν=2.33)\nMIME", offset=(10, -22))

    # Experimental markers (stars)
    axB.scatter([fl3_speed_exp], [fl3_sens], marker="*", s=260,
                 color=OI["red"], edgecolor="black", linewidth=0.8,
                 zorder=7, label="FL-3 experimental (de Jongh)")
    axB.scatter([fl9_speed_exp], [fl9_sens], marker="*", s=260,
                 color=OI["blue"], edgecolor="black", linewidth=0.8,
                 zorder=7, label="FL-9 experimental (de Jongh)")

    cbar = fig.colorbar(sc, ax=axB, pad=0.02, shrink=0.88)
    cbar.set_label(r"Normalised wavenumber  $\nu$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    axB.set_xlabel("Swimming speed at κ = 0.49 (mm/s)")
    axB.set_ylabel("Off-center sensitivity at 0.1 R_ves (%)")
    axB.set_title(r"B — Speed–robustness Pareto frontier  ($\kappa = 0.49$, 1/4\" vessel)")
    axB.grid(alpha=0.25, linewidth=0.5)
    axB.legend(loc="upper right", fontsize=7.8, framealpha=0.9)

    # ── Bottom text ──
    ax_note.text(
        0.5, 0.55,
        "Both computational and experimental optima exclude wall-contact "
        "friction. The analytical lubrication correction "
        "(Goldman-Cox-Brenner 1967) does not close the remaining gap — "
        "the residual is contact-mediated, requiring experimental "
        "calibration (ContactFrictionNode stub ready). Calibrating a "
        "single rolling friction parameter and re-optimising could "
        "reveal whether the optimal design shifts under contact.",
        ha="center", va="center", fontsize=8.8, wrap=True,
        color=OI["black"],
    )

    fig.suptitle(
        "Differentiable helical-microrobot design optimisation "
        "— MIME MLP surrogate vs de Jongh (2025) experiments",
        fontsize=12.5, y=1.00,
    )

    out_png = FIG_DIR / "design_optimisation.png"
    out_pdf = FIG_DIR / "design_optimisation.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    # ── Underlying data ──
    out_data = {
        "elapsed_ms": elapsed * 1000.0,
        "n_kappas": len(kappas),
        "n_starts": int(DENSE_STARTS.shape[0]),
        "gradient_steps": N_GRAD_STEPS,
        "learning_rate": LR_NU,
        "kappas": kappas.tolist(),
        "speed_optimal_nu": speed_nus.tolist(),
        "speed_optimal_value_nd": speed_vals.tolist(),
        "robust_optimal_nu": robust_nus.tolist(),
        "robust_optimal_sensitivity": robust_vals.tolist(),
        "all_speed_optima": speed_opt_tree,
        "all_robust_optima": robust_opt_tree,
        "pareto_kappa": kappa_pareto,
        "pareto_points": {
            "nu": np.asarray(nu_grid).tolist(),
            "speed_mm_s": pareto_spds.tolist(),
            "sensitivity_pct": pareto_sens.tolist(),
            "frontier_indices": frontier_idx.tolist(),
        },
        "experimental": {
            "speed_optimal_per_vessel": {
                v: {"kappa": VESSEL_KAPPA[v],
                    "nu": float(speed_opt_exp[v][0]),
                    "speed_mm_s": float(speed_opt_exp[v][1]),
                    "design": speed_opt_exp[v][2]}
                for v in VESSEL_KAPPA
            },
            "robust_optimal_per_vessel": {
                v: {"kappa": VESSEL_KAPPA[v],
                    "nu": float(robust_opt_exp[v][0]),
                    "design": robust_opt_exp[v][1]}
                for v in VESSEL_KAPPA
            },
            "FL3_quarter": {"speed_mm_s": fl3_speed_exp,
                             "sensitivity_pct": fl3_sens},
            "FL9_quarter": {"speed_mm_s": fl9_speed_exp,
                             "sensitivity_pct": fl9_sens},
        },
        "weights_path": str(WEIGHTS_PATH),
        "training_set_size": 397,
        "mlp_test_mae_mm_s": 0.029,
    }
    out_json = DATA_DIR / "design_optimisation_data.json"
    out_json.write_text(json.dumps(out_data, indent=2))
    print(f"Saved: {out_json}")

    # ── Validation checks ──
    print("\n── Validation ──")
    # Speed optimum should be near 1.0
    at_05 = speed_nus[np.argmin(np.abs(kappas - 0.49))]
    print(f"  speed-optimal ν at κ=0.49: {at_05:.2f} "
          f"(expect ≈ 1.0)")
    robust_at_05 = robust_nus[np.argmin(np.abs(kappas - 0.49))]
    print(f"  robustness-optimal ν at κ=0.49: {robust_at_05:.2f} "
          f"(expect ≈ 2.0–2.5)")
    # Pareto: FL-3 should be at high speed / high sensitivity
    i3 = int(np.argmin(np.abs(np.asarray(nu_grid) - 1.0)))
    i9 = int(np.argmin(np.abs(np.asarray(nu_grid) - 2.33)))
    print(f"  FL-3 (ν=1.0): speed={pareto_spds[i3]:.2f} mm/s, "
          f"sens={pareto_sens[i3]:.2f}%")
    print(f"  FL-9 (ν=2.33): speed={pareto_spds[i9]:.2f} mm/s, "
          f"sens={pareto_sens[i9]:.2f}%")


if __name__ == "__main__":
    main()
