#!/usr/bin/env python3
"""Figures for the step-out damping-ratio study — Physical Review Applied style, TRUE LaTeX.

Uses the matplotlib *pgf* backend with pdflatex (real LaTeX; no dvipng needed) → emits a vector
**.pdf** (primary, for submission) and a raster **.png** (preview) per figure. All maths render in
LaTeX (Computer Modern), multi-letter subscripts upright via \\mathrm. Okabe–Ito colour-blind-safe
palette (blue = ramp-up, vermilion = ramp-down, consistent across C/D and the supplementary loops).
Annotations are placed in empty regions/corners so no text overlaps a curve, marker or legend.

Fig A designs · Fig B scaling · Fig C coupled loops · Fig D regime map · Fig E wall ratio.
Individual per-design coupled loops live in supplementary/ (superseded by Fig C).

Run:  .venv/bin/python scripts/mod1_sweep_figures.py [--which A B C D E]
"""
from __future__ import annotations
import os, sys, json, math, argparse
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "schwarz_vessel_helix" / "output" / "damping_sweep"
import numpy as np
import matplotlib; matplotlib.use("pgf"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "text.usetex": True, "pgf.rcfonts": False,
    "font.family": "serif",
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 14.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10.5,
    "axes.linewidth": 1.0, "lines.linewidth": 2.0, "legend.framealpha": 0.92,
})

ZC = math.pi / 8.0
C_UP = "#0072B2"    # blue      — ramp-up  / pull-out
C_DN = "#D55E00"    # vermilion — ramp-down / pull-in
C_FL = "#009E73"    # bluish green — FL group
C_FW = "#CC79A7"    # reddish purple — FW group
C_BAND = "#56B4E9"  # sky blue — hysteresis band shading


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def save(fig, stem):
    p_pdf = OUT / f"{stem}.pdf"; p_png = OUT / f"{stem}.png"
    fig.savefig(p_pdf); fig.savefig(p_png, dpi=220); plt.close(fig)
    print("saved", p_pdf.name, "+", p_png.name)


def _design_floor():
    rows = load_jsonl(OUT / "mod1_design_sweep.jsonl")
    key = {f"{r['group']}{r['id']}": r for r in rows}
    return {d: key[d]["f_si_analytic"] for d in ("FL9", "FW1", "FW6")}


# ─────────────────────────────────────────────────────────────────────────────
def fig_A():
    rows = load_jsonl(OUT / "mod1_design_sweep.jsonl")
    fl = [r for r in rows if r["group"] == "FL"]; fw = [r for r in rows if r["group"] == "FW"]
    names = [f"{r['group']}{r['id']}" for r in fl + fw]; x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhspan(0.008, ZC, color=C_BAND, alpha=0.10)
    ax.plot(x[:len(fl)], [r["zeta"] for r in fl], "o", color=C_FL, ms=9, label="FL group")
    ax.plot(x[len(fl):], [r["zeta"] for r in fw], "s", color=C_FW, ms=9, label="FW group")
    ax.axhline(ZC, color="k", ls="--", lw=1.6)
    ax.text(len(names) - 0.4, ZC * 1.07, r"hysteresis closes  $\zeta_c=\pi/8\approx0.39$",
            ha="right", va="bottom", fontsize=12)
    ax.text(0.5, 0.11, "underdamped\n(hysteretic)", fontsize=12, color="#1b4a6b", va="center")
    ax.set_yscale("log"); ax.set_ylim(0.008, 0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("de Jongh design")
    ax.set_ylabel(r"damping ratio  $\zeta = C/2\sqrt{I\,mB}$")
    ax.set_title("All 17 mm-scale de Jongh devices are deeply underdamped")
    ax.legend(loc="center right"); ax.grid(alpha=0.3, which="both", axis="y")
    fig.tight_layout(); save(fig, "mod1_figA_designs")


def fig_B():
    rows = load_jsonl(OUT / "mod1_scaling_sweep.jsonl"); des = load_jsonl(OUT / "mod1_design_sweep.jsonl")
    z = np.array([r["zeta"] for r in rows]); wn = rows[0]["omega_n_hz"]; o = np.argsort(z); z = z[o]
    so = np.array([np.nan if rows[i]["f_so_sim_norm"] is None else rows[i]["f_so_sim_norm"] for i in o])
    si = np.array([np.nan if rows[i]["f_si_sim_norm"] is None else rows[i]["f_si_sim_norm"] for i in o])
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.3))
    # ── (a)
    zz = np.geomspace(z.min(), z.max(), 300)
    ax[0].plot(zz, 1.0 / (2.0 * zz), "k-", lw=2.0, label=r"$f_\mathrm{so}/\omega_n=1/2\zeta$")
    ax[0].axhline(4 / math.pi, color="0.35", lw=2.0, label=r"$f_\mathrm{si}/\omega_n=4/\pi$")
    ax[0].plot(z, so, "o", color=C_UP, ms=5.5, label=r"sim $f_\mathrm{so}\!\uparrow$")
    ax[0].plot(z, si, "s", color=C_DN, ms=5.5, label=r"sim $f_\mathrm{si}\!\downarrow$")
    ax[0].axvline(ZC, color="green", ls="--", lw=1.8)
    ax[0].text(ZC * 1.12, 55, r"$\zeta_c=\pi/8$", color="green", fontsize=12, rotation=90, va="top")
    zfl = [r["zeta"] for r in des]
    ax[0].axvspan(min(zfl), max(zfl), color="orange", alpha=0.18)
    ax[0].text(math.sqrt(min(zfl) * max(zfl)), 7.5, "17 de Jongh", color="#8a5a00",
               fontsize=10, ha="center", va="center", rotation=90)
    ax[0].annotate("Fazeli 2023:\noverdamped " r"$f_\mathrm{so}\!\ll\!f_n$",
                   xy=(0.75, 0.70), xytext=(0.03, 3.3), fontsize=9.5, color="#333333",
                   ha="left", va="center",
                   arrowprops=dict(arrowstyle="->", color="#333333", lw=1.3))
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(0.33, 150)
    ax[0].set_xlabel(r"damping ratio  $\zeta$"); ax[0].set_ylabel(r"$f/\omega_n$")
    ax[0].set_title(r"(a) Universal step-out window vs $\zeta$")
    ax[0].legend(loc="lower left", fontsize=8, ncol=2, columnspacing=1.0, handletextpad=0.4,
                 labelspacing=0.3, borderpad=0.4)
    ax[0].grid(alpha=0.3, which="both")
    secx = ax[0].secondary_xaxis("top", functions=(lambda t: 1 / (2 * t), lambda q: 1 / (2 * q)))
    secx.set_xlabel(r"quality factor  $Q = 1/2\zeta$", fontsize=13)
    # ── (b)
    L = np.array([r["body_len_mm"] for r in rows]); ob = np.argsort(L); L = L[ob]
    fso_ab = np.array([rows[i]["f_so_abs"] for i in ob]); fsi_ab = np.array([rows[i]["f_si_abs"] for i in ob])
    Lc = 7.47 * (0.0159 / ZC); band = L >= Lc
    ax[1].fill_between(L[band], fsi_ab[band], fso_ab[band], color=C_BAND, alpha=0.22)
    ax[1].plot(L, fso_ab, "-", color="k", lw=2.2, label=r"$f_\mathrm{so}=mB/C$ (scale-invariant)")
    ax[1].plot(L, fsi_ab, "-", color="0.35", lw=2.2, label=r"$f_\mathrm{si}=(4/\pi)\omega_n\propto 1/\lambda$")
    ax[1].axvspan(L.min(), Lc, color="0.85", alpha=0.5)
    ax[1].text(Lc * 0.68, 430, "overdamped\n(single\nthreshold)", fontsize=9.5, ha="right", color="0.3")
    ax[1].text(Lc * 3, 40, "underdamped\nhysteresis band", fontsize=10.5, color="#1b4a6b")
    ax[1].annotate("window closes\n" r"$\approx$\,302\,$\mu$m", xy=(Lc, 622), xytext=(Lc * 2.3, 235),
                   fontsize=10.5, ha="left", bbox=dict(boxstyle="round", fc="white", ec="green"),
                   arrowprops=dict(arrowstyle="->", color="green", lw=1.6))
    for r in des:
        c = C_FL if r["group"] == "FL" else C_FW
        ax[1].plot(r["L_mm"], r["f_so_analytic"], "^", color=c, ms=8, alpha=0.85)
        ax[1].plot(r["L_mm"], r["f_si_analytic"], "v", color=c, ms=8, alpha=0.85)
    # FW group varies length (fixed radius) → faint guides trace the length trend: shorter
    # screws have less inertia and drag (I, C ∝ L), so BOTH f_so and f_si rise (∝~1/L, 1/√L).
    fw = sorted((r for r in des if r["group"] == "FW"), key=lambda r: r["L_mm"])
    Lfw = [r["L_mm"] for r in fw]
    ax[1].plot(Lfw, [r["f_so_analytic"] for r in fw], ls=(0, (4, 2)), color=C_FW, lw=1.3, alpha=0.8, zorder=1)
    ax[1].plot(Lfw, [r["f_si_analytic"] for r in fw], ls=(0, (4, 2)), color=C_FW, lw=1.3, alpha=0.8, zorder=1)
    ax[1].annotate(r"FW: vary $L$" "\n" r"(shorter $\to$ higher $f$)", xy=(Lfw[0], fw[0]["f_so_analytic"]),
                   xytext=(3.7, 1500), fontsize=9, color=C_FW, ha="left", va="bottom",
                   arrowprops=dict(arrowstyle="->", color=C_FW, lw=1.2))
    # FL group: 11 designs all at L=7.47 mm (vary only thread pitch) → they overlap into one stack
    nfl = sum(1 for r in des if r["group"] == "FL")
    fl_so = next(r["f_so_analytic"] for r in des if r["group"] == "FL")
    ax[1].annotate(rf"FL: {nfl} designs" "\n" r"(overlap at $L=7.47$ mm)", xy=(7.47, fl_so),
                   xytext=(9.0, 170), fontsize=9, color=C_FL, ha="left", va="center",
                   arrowprops=dict(arrowstyle="->", color=C_FL, lw=1.2))
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlim(L.min(), L.max() * 1.03)      # pin left edge so the overdamped band fills to it
    ax[1].set_xlabel(r"body length  $L$ [mm]"); ax[1].set_ylabel(r"frequency  $f$ [Hz]")
    ax[1].set_title("(b) Absolute window vs size")
    # 4-entry legend matching (a): two analytic lines + two group-colour marker pairs
    # (up-triangle = f_so, down-triangle = f_si; green = FL group, purple = FW group)
    def _pair(c):
        return (Line2D([], [], marker="^", ls="", mfc=c, mec=c, ms=8),
                Line2D([], [], marker="v", ls="", mfc=c, mec=c, ms=8))
    handles = [Line2D([], [], color="k", lw=2.2), Line2D([], [], color="0.35", lw=2.2),
               _pair(C_FL), _pair(C_FW)]
    labels = [r"$f_\mathrm{so}=mB/C$", r"$f_\mathrm{si}=(4/\pi)\omega_n$",
              r"FL ($\blacktriangle f_\mathrm{so}$, $\blacktriangledown f_\mathrm{si}$)",
              r"FW ($\blacktriangle f_\mathrm{so}$, $\blacktriangledown f_\mathrm{si}$)"]
    ax[1].legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                 loc="lower left", fontsize=8, ncol=2, columnspacing=1.0, handletextpad=0.5,
                 labelspacing=0.3, borderpad=0.4)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); save(fig, "mod1_figB_scaling")


def fig_C():
    sys.path.insert(0, str(REPO / "scripts")); from mod2_hysteresis_coupled import selftrip
    floor = _design_floor()
    files = [OUT / f"mod1_coupled_loop_{d}.npz" for d in ("FL9", "FW1", "FW6")]
    files += [OUT / "supplementary" / f"mod1_coupled_loop_{d}.npz" for d in ("FL9", "FW1", "FW6")]
    seen = {}
    for f in files:
        if f.exists():
            d = np.load(f, allow_pickle=True); seen.setdefault(str(d["design"]), f)
    order = [d for d in ("FL9", "FW1", "FW6") if d in seen]
    if not order:
        print("Fig C: no coupled loop npz — skipping"); return
    fig, ax = plt.subplots(1, len(order), figsize=(5.5 * len(order), 5.0), squeeze=False)
    for j, design in enumerate(order):
        d = np.load(seen[design], allow_pickle=True)
        dU, sU, bU = d["drive_up"], d["spin_up"], d["beta_up"]; dD, sD = d["drive_dn"], d["spin_dn"]
        fso = selftrip(dU, sU, bU); fsi = floor.get(design)
        a = ax[0][j]; dmax = max(dU.max() if len(dU) else 1, dD.max() if len(dD) else 1)
        if fso is not None and fsi is not None:
            a.axvspan(fsi, fso, color=C_BAND, alpha=0.22, label="bistable band")
        if fsi is not None:
            a.axvline(fsi, color=C_DN, ls="--", lw=1.8,
                      label=r"predicted $f_\mathrm{si}\approx$ " + f"{fsi:.0f} Hz")
        a.plot(dU, sU, ".", ms=4, alpha=0.55, color=C_UP, label="ramp up")
        a.plot(dD, sD, ".", ms=4, alpha=0.55, color=C_DN, label="ramp down")
        a.plot([0, dmax], [0, dmax], "k--", lw=1.0, alpha=0.6, label="synchronous")
        if fso is not None:
            a.axvline(fso, color=C_UP, ls=":", lw=2.0,
                      label=r"$f_\mathrm{so}\!\uparrow\approx$ " + f"{fso:.0f} Hz")
        else:
            a.text(0.5, 0.05, r"no step-out $\leq$ 260 Hz" "\n" "(tracks through)",
                   transform=a.transAxes, ha="center", fontsize=11, color=C_UP)
        a.set_xlabel(r"drive frequency  $f_\mathrm{d}$ [Hz]")
        if j == 0:
            a.set_ylabel(r"body spin  $f_\mathrm{b}$ [Hz]")
        a.set_title(design); a.legend(fontsize=9.5, loc="upper left"); a.grid(alpha=0.3)
    fig.suptitle("Coupled step-out hysteresis loops (elongated FL9/FW1 vs stubby FW6)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "mod1_figC_coupled")


def fig_D():
    rows = load_jsonl(OUT / "mod1_scaling_sweep.jsonl"); des = load_jsonl(OUT / "mod1_design_sweep.jsonl")
    z = np.array([r["zeta"] for r in rows]); o = np.argsort(z); z = z[o]
    so = np.array([np.nan if rows[i]["f_so_sim_norm"] is None else rows[i]["f_so_sim_norm"] for i in o])
    si = np.array([np.nan if rows[i]["f_si_sim_norm"] is None else rows[i]["f_si_sim_norm"] for i in o])
    zz = np.geomspace(0.005, 2.0, 400); f_so = 1.0 / (2.0 * zz); f_in = np.minimum(f_so, 4.0 / math.pi)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.fill_between(zz, f_in, f_so, where=f_so > 4 / math.pi, color=C_BAND, alpha=0.16, label="hysteresis band")
    ax.plot(zz, f_so, "k-", lw=2.4, label=r"$f_\mathrm{so}=\omega_n/2\zeta=mB/C$ (pull-out)")
    ax.plot(zz, f_in, color=C_DN, lw=2.4,
            label=r"$f_\mathrm{si}=\omega_n\min(\tfrac{1}{2\zeta},\tfrac{4}{\pi})$ (pull-in)")
    ax.plot(z, so, "o", color="k", ms=5, alpha=0.75, label=r"sim $f_\mathrm{so}\!\uparrow$")
    ax.plot(z, si, "s", color=C_DN, ms=5, alpha=0.75, label=r"sim $f_\mathrm{si}\!\downarrow$")
    ax.axvline(ZC, color="green", ls="--", lw=1.8)
    zfl = [r["zeta"] for r in des]; ax.axvspan(min(zfl), max(zfl), color="orange", alpha=0.20, label="17 de Jongh")
    ax.text(0.0108, 0.5, r"underdamped" "\n" r"hysteretic" "\n" r"($\zeta<\pi/8$)", fontsize=12, color="#1b4a6b")
    ax.text(0.55, 2.6, r"overdamped" "\n" r"single Adler" "\n" r"($\zeta\geq\pi/8$)", fontsize=12, color="0.35")
    ax.annotate(r"hysteresis closes" "\n" r"$\zeta_c=\pi/8\approx0.39$", xy=(ZC, 1.25), xytext=(0.115, 0.46),
                fontsize=11, color="green", ha="left", bbox=dict(boxstyle="round", fc="white", ec="green"),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.6))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.3, 120)
    ax.set_xlabel(r"damping ratio  $\zeta = C/2\sqrt{I\,mB}$"); ax.set_ylabel(r"$f/\omega_n$")
    ax.set_title("Step-out regime map and equation validation")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(alpha=0.3, which="both")
    secx = ax.secondary_xaxis("top", functions=(lambda t: 1 / (2 * t), lambda q: 1 / (2 * q)))
    secx.set_xlabel(r"quality factor  $Q = 1/2\zeta$  (Josephson $\beta_c=Q^2$)", fontsize=13)
    fig.tight_layout(); save(fig, "mod1_figD_regime_map")


def fig_E():
    p_in = OUT / "mod1_wall_ratio_sweep.jsonl"
    if not p_in.exists():
        print("Fig E: run mod1_design_sweep.py --mode wall first — skipping"); return
    rows = sorted(load_jsonl(p_in), key=lambda r: r["ratio"])
    ratio = np.array([r["ratio"] for r in rows]); fso = np.array([r["f_so_analytic"] for r in rows])
    fsi = np.array([r["f_si_analytic"] for r in rows]); C = np.array([r["C"] for r in rows])
    xo = ratio[[r["is_ours"] for r in rows].index(True)]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.fill_between(ratio, fsi, fso, color=C_BAND, alpha=0.15, label="hysteresis band")
    ax.plot(ratio, fso, "o-", color="k", lw=2.2, ms=8, label=r"$f_\mathrm{so}=mB/C$ (wall-sensitive)")
    ax.plot(ratio, fsi, "s-", color=C_DN, lw=2.2, ms=8, label=r"$f_\mathrm{si}=(4/\pi)\omega_n$ (wall-drag-free)")
    ax.axvline(xo, color="green", ls="--", lw=1.8)
    ax.text(xo * 0.98, 155, "our 1/4-inch tube\n" + f"(ratio {xo:.3f})", color="green",
            fontsize=10, ha="right", va="center")
    ax.text(ratio.min() * 1.02, fso[0] - 40, "tighter", fontsize=10.5, ha="left", va="top", color="0.35")
    ax.text(ratio.max(), fso[-1] + 12, r"looser $\to$ free space", fontsize=10.5, ha="right", va="bottom", color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel(r"confinement ratio  $R_\mathrm{ves}/R_\mathrm{cyl}$"); ax.set_ylabel(r"frequency  $f$ [Hz]")
    ax.set_title(r"Wall shifts the pull-out $f_\mathrm{so}$ (via drag $C$); the $f_\mathrm{si}$ floor is wall-free")
    ax.set_ylim(0, 900); ax.grid(alpha=0.3, which="both")
    axc = ax.twinx()
    axc.plot(ratio, C * 1e10, "^:", color="#7030A0", lw=1.8, ms=7, label=r"drag $C$ ($\propto\zeta$)")
    axc.set_ylabel(r"rotational drag  $C$  [$10^{-10}\,\mathrm{N\,m\,s}$]", color="#7030A0")
    axc.tick_params(axis="y", colors="#7030A0")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axc.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="center right")
    fig.tight_layout(); save(fig, "mod1_figE_wall_ratio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["A", "B", "C", "D", "E"])
    a = ap.parse_args()
    fns = {"A": fig_A, "B": fig_B, "C": fig_C, "D": fig_D, "E": fig_E}
    for k in a.which:
        fns[k]()


if __name__ == "__main__":
    main()
