#!/usr/bin/env python3
"""Analyze a coupled ramp recording: body spin tracking vs drive, wobble β(ω), and
self-trip detection (does the 3D body desynchronise on its own / under perturbation?).

Usage:
  python scripts/analyze_coupled_ramp.py --usdc <path> --hz0 5 --hz1 150 --ramp-s 2.5 \
      [--sample-every 20] [--dt 1e-4] [--out fig.png]
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pxr import Usd, UsdGeom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdc", required=True)
    ap.add_argument("--hz0", type=float, required=True)
    ap.add_argument("--hz1", type=float, required=True)
    ap.add_argument("--ramp-s", type=float, required=True)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--out", default="/tmp/coupled_ramp_analysis.png")
    a = ap.parse_args()
    fdt = a.sample_every * a.dt

    st = Usd.Stage.Open(a.usdc); n = int(st.GetEndTimeCode()) + 1
    A = st.GetPrimAtPath("/World/Actors/UMR").GetAttribute("xformOp:orient")
    Q = [A.Get(f) for f in range(n)]
    def q4(q): r = q.GetReal(); im = q.GetImaginary(); return np.array([r, im[0], im[1], im[2]])
    bx = np.zeros((n, 3)); bz = np.zeros((n, 3))
    for f in range(n):
        w, x, y, z = q4(Q[f])
        bx[f] = [1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)]
        bz[f] = [2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)]
    t = np.arange(n) * fdt
    drive = a.hz0 + (a.hz1 - a.hz0) * np.minimum(1.0, t / a.ramp_s)
    beta = np.degrees(np.arccos(np.clip(bz[:, 0], -1, 1)))
    phi = np.unwrap(np.arctan2(bx[:, 2], bx[:, 1]))
    spin = np.abs(np.gradient(phi, t)) / (2 * np.pi)
    sm = lambda v, k=11: np.convolve(v, np.ones(k)/k, mode="same")
    spin_s = sm(spin)
    # self-trip detection: first drive freq where spin falls persistently <50% of drive
    trip = None; run = 0
    for i in range(n):
        if drive[i] < 8: continue
        if spin_s[i] < 0.5 * drive[i]:
            run += 1
            if run > 40: trip = drive[i]; break
        else: run = 0
    print(f"=== coupled ramp analysis: {a.usdc.split('/')[-1]} ===")
    print(f"  frames={n} sim={t[-1]:.2f}s drive {a.hz0:.0f}->{a.hz1:.0f}Hz")
    print(f"  {'drive':>8} {'meanβ':>7} {'maxβ':>7} {'spin':>7} {'slip%':>7}")
    for lo, hi in [(5,15),(15,25),(25,35),(35,50),(50,80),(80,120),(120,250)]:
        m = (drive >= lo) & (drive < hi)
        if m.sum():
            dm = drive[m].mean(); sp = spin_s[m].mean()
            print(f"  {lo:3d}-{hi:<3d} {beta[m].mean():7.1f} {beta[m].max():7.1f} {sp:7.1f} {(1-sp/dm)*100:7.0f}")
    print(f"  SELF-TRIP: {'at ~%.0f Hz' % trip if trip else 'NONE (tracked to %.0f Hz)' % a.hz1}")
    print(f"  max wobble β = {beta.max():.1f}° @ {drive[np.argmax(beta)]:.0f} Hz")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].plot(t, drive, "k--", label="drive"); ax[0].plot(t, spin_s, "tab:blue", label="body spin")
    if trip: ax[0].axvline(t[np.argmax(drive >= trip)], color="r", ls=":", label=f"trip~{trip:.0f}Hz")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("Hz"); ax[0].set_title("spin tracking"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(drive, beta, "tab:red"); ax[1].set_xlabel("drive (Hz)"); ax[1].set_ylabel("wobble β (deg)")
    ax[1].set_title("wobble vs drive"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(a.out, dpi=130); print(f"  saved {a.out}")


if __name__ == "__main__":
    main()
