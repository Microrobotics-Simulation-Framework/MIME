#!/usr/bin/env python3
"""MOD-2 null-result figure: single-RPM (gradient) vs dual-RPM (gradient-cancelled) at matched
|B|=1.2 mT. Reads the two coupled step-out traces and shows that the RPM field gradient does NOT
shift the step-out frequency (identical transition band), its only signature being extra wobble.

Two panels: (a) body spin vs drive (both track then desync over the same band); (b) wobble angle β
vs drive (single reaches a larger β and wobbles slightly earlier). Physical-Review-Applied style
(true LaTeX via pgf/pdflatex), Okabe–Ito colour-blind-safe palette. Emits .pdf + .png.

Run:  .venv/bin/python scripts/mod2_null_figure.py \
        --single <single.npz> --dual <dual.npz> --out <dir>/mod2_single_vs_dual
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("pgf"); import matplotlib.pyplot as plt

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "text.usetex": True, "pgf.rcfonts": False, "font.family": "serif",
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 14.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10.5,
    "axes.linewidth": 1.0, "lines.linewidth": 2.0, "legend.framealpha": 0.92,
})
C_SINGLE = "#D55E00"   # vermilion — single RPM (has field gradient)
C_DUAL = "#0072B2"     # blue      — dual RPM (gradient cancelled)
BAND = (165.0, 202.0)  # step-out transition band (onset -> 50% desync), same for both


def load(p):
    d = np.load(p, allow_pickle=True)
    return np.asarray(d["drive"]), np.asarray(d["spin"]), np.asarray(d["beta"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True)
    ap.add_argument("--dual", required=True)
    ap.add_argument("--out", required=True, help="output path prefix (.pdf/.png appended)")
    a = ap.parse_args()
    dS, sS, bS = load(a.single)
    dD, sD, bD = load(a.dual)
    dmax = max(dS.max(), dD.max())

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.1))
    # (a) spin vs drive
    ax[0].axvspan(*BAND, color="0.85", alpha=0.6)
    ax[0].plot([0, dmax], [0, dmax], "k--", lw=1.0, alpha=0.6, label="synchronous")
    ax[0].plot(dS, sS, ".", ms=4, alpha=0.5, color=C_SINGLE, label="single RPM (gradient)")
    ax[0].plot(dD, sD, ".", ms=4, alpha=0.5, color=C_DUAL, label="dual RPM (gradient-cancelled)")
    ax[0].text(np.mean(BAND), 6, "step-out\n" r"$\approx$165–202 Hz" "\n(same for both)",
               ha="center", va="bottom", fontsize=9.5, color="0.3")
    ax[0].set_xlabel(r"drive frequency  $f_\mathrm{d}$ [Hz]")
    ax[0].set_ylabel(r"body spin  $f_\mathrm{b}$ [Hz]")
    ax[0].set_title("(a) Step-out is unchanged by the gradient")
    ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)
    # (b) wobble vs drive
    ax[1].axvspan(*BAND, color="0.85", alpha=0.6)
    ax[1].plot(dS, bS, ".", ms=4, alpha=0.5, color=C_SINGLE, label="single RPM (gradient)")
    ax[1].plot(dD, bD, ".", ms=4, alpha=0.5, color=C_DUAL, label="dual RPM (gradient-cancelled)")
    ax[1].axhline(bS.max(), color=C_SINGLE, ls=":", lw=1.5)
    ax[1].axhline(bD.max(), color=C_DUAL, ls=":", lw=1.5)
    ax[1].text(120, bS.max() - 1.5, rf"single $\beta_\mathrm{{max}}\approx{bS.max():.0f}^\circ$",
               color=C_SINGLE, fontsize=10, va="top", ha="left")
    ax[1].text(120, bD.max() + 1.5, rf"dual $\beta_\mathrm{{max}}\approx{bD.max():.0f}^\circ$",
               color=C_DUAL, fontsize=10, va="bottom", ha="left")
    ax[1].set_xlabel(r"drive frequency  $f_\mathrm{d}$ [Hz]")
    ax[1].set_ylabel(r"wobble angle  $\beta$ [deg]")
    ax[1].set_title("(b) The gradient's only signature is extra wobble")
    ax[1].legend(loc="upper left"); ax[1].grid(alpha=0.3)

    out = Path(a.out)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print(f"saved {out.with_suffix('.pdf').name} + .png")
    print(f"  single: beta_max={bS.max():.0f} deg   dual: beta_max={bD.max():.0f} deg")


if __name__ == "__main__":
    main()
