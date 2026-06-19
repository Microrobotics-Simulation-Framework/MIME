#!/usr/bin/env python3
"""Plot the MOD-2 / EP-1 gradient comparison: single-RPM (gradient) vs dual-RPM
(gradient-cancelled) step-out at matched |B|. Reads the two trace .npz files written by
mod2_stepout_compare.py and reports each self-trip + the gradient depression.

Usage: python scripts/mod2_compare_plot.py --single <single.npz> --dual <dual.npz> --out fig.png
(--single optional; if missing, plots dual alone.)
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def selftrip(drive, spin):
    """First sustained desync AFTER the body has locked (skips the spin-up transient)."""
    locked_once = False; run = 0
    for i in range(len(drive)):
        if drive[i] < 8: continue
        if abs(spin[i] / drive[i] - 1.0) < 0.3:     # tracking
            locked_once = True; run = 0; continue
        if locked_once and spin[i] < 0.5 * drive[i]:
            run += 1
            if run > 2: return float(drive[i])
    return None


def load(path):
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["drive"]), np.asarray(d["spin"]), np.asarray(d["beta"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", default=None)
    ap.add_argument("--dual", required=True)
    ap.add_argument("--out", default="/tmp/mod2_compare.png")
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    summary = []
    series = [("dual (gradient-cancelled)", a.dual, "tab:blue")]
    if a.single:
        series.insert(0, ("single (gradient)", a.single, "tab:red"))
    dmax = 0
    for label, path, col in series:
        try:
            drive, spin, beta = load(path)
        except FileNotFoundError:
            print(f"  {label}: {path} not found — skipping"); continue
        dmax = max(dmax, drive.max())
        tr = selftrip(drive, spin)
        ax[0].plot(drive, spin, ".", ms=3, color=col, label=f"{label} (trip {tr})")
        ax[1].plot(drive, beta, ".", ms=3, color=col, label=label)
        summary.append((label, tr, float(beta.max())))
    ax[0].plot([0, dmax], [0, dmax], "k--", lw=0.7, alpha=0.5, label="synchronous")
    ax[0].set_xlabel("drive freq (Hz)"); ax[0].set_ylabel("body spin (Hz)")
    ax[0].set_title("Step-out: single vs dual (matched |B|)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("drive freq (Hz)"); ax[1].set_ylabel("wobble β (deg)")
    ax[1].set_title("Wobble vs drive"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(a.out, dpi=130)
    print("=== MOD-2 gradient comparison ===")
    for label, tr, bmax in summary:
        print(f"  {label:28s} self-trip={tr}  max β={bmax:.0f}°")
    if len(summary) == 2:
        s, d = summary[0][1], summary[1][1]
        if s and d:
            print(f"  gradient depression: single trips {s:.0f} Hz vs dual {d:.0f} Hz "
                  f"→ gradient lowers step-out by {d-s:.0f} Hz ({d/s:.2f}×)")
        elif s and not d:
            print(f"  single trips {s:.0f} Hz; dual tracks past {dmax:.0f} Hz (no trip) "
                  f"→ gradient depression ≥ {dmax/s:.2f}×")
    print(f"  saved {a.out}")


if __name__ == "__main__":
    main()
