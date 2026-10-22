#!/usr/bin/env python3
"""M2/M3 — COUPLED two-scale step-out HYSTERESIS LOOP for the schwarz vessel-helix
experiment. Extends the ramp-up-only mod2_stepout_compare.py with a ramp-DOWN branch
so the emergent inertial body's full bistable loop [f_si↓, f_so↑] is measured in the
coupled solver (FVM far-field ⊗ confined BEM near-field), not just the up-ramp self-trip.

Loop = one continuous run: staircase the drive UP (5→ceiling) to reach a running/
desynchronised state, then staircase DOWN (ceiling→5) to catch the homoclinic re-lock.
The graph state (body ω/orientation, motor phase, FVM/BEM fields) flows continuously —
no state surgery. Commanded velocity is passed as jnp.float32 (traced) so the graph does
NOT recompile per step (a continuous ramp + external pose would; hence the staircase).

Detectors: up self-trip → f_so↑ (spin drops <50% drive, lock-first); down re-lock →
f_si↓ (spin returns within 30% of drive, after being desynced).

Outputs (prefix from --out): <prefix>.npz (both branches), <prefix>.png (loop figure),
<prefix>.usdc (body + driving magnet recording).

Usage (GPU):
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
    scripts/../.venv/bin/python scripts/mod2_hysteresis_coupled.py --direction loop \
      --hz0 5 --hz1 250 --hz-step 12 --hold-steps 1000 --dt 1e-4 --mag-dipole 40.5 \
      --nu 2.33 --l-mm 7.47 --design FL9 --out <prefix>
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
import numpy as np
import jax.numpy as jnp
from mime.core.quaternion import rotate_vector_inverse
from pxr import Usd, UsdGeom, Gf

WORLD_USDA = REPO / "experiments" / "schwarz_vessel_helix" / "scene" / "world.usda"


def write_usdc(frames, out, fps=30.0):
    """frames: list of {prim_path: (translate3, quat_wxyz)}. Animates body + magnet.
    (Copied from mod2_stepout_compare.py to avoid importing/altering that script.)"""
    out = Path(out); stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.OverridePrim("/World").GetReferences().AddReference(
        os.path.relpath(WORLD_USDA.resolve(), out.parent))
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(max(0, len(frames) - 1))
    stage.SetTimeCodesPerSecond(fps); stage.SetFramesPerSecond(fps)
    op = {}
    for pth in (frames[0] if frames else {}):
        xf = UsdGeom.Xformable(stage.OverridePrim(pth)); xf.ClearXformOpOrder()
        op[pth] = (xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, ""),
                   xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, ""))
    for fi, fr in enumerate(frames):
        tc = Usd.TimeCode(fi)
        for pth, (t, q) in fr.items():
            tr, oo = op[pth]
            tr.Set(Gf.Vec3d(*[float(v) for v in t]), tc)
            oo.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])), tc)
    stage.GetRootLayer().Save()
    flat = Usd.Stage.Open(str(out)).Flatten()

    def _strip(s, root):
        if not root:
            s.ClearReferenceList(); s.ClearPayloadList()
        for c in s.nameChildren:
            _strip(c, False)
    _strip(flat.GetPrimAtPath("/"), True); flat.Export(str(out))


def _level_tracking(drive, spin, beta, lv, beta_thr, ratio_lo, ratio_hi):
    """Is the body synchronously tracking at frequency level lv? Uses the level MEDIAN
    (robust to within-level transients): spin/drive near 1 and small wobble β."""
    m = drive == lv
    if not np.any(m):
        return False
    r = float(np.median(spin[m] / lv))
    b = float(np.median(beta[m])) if beta is not None else 0.0
    return (ratio_lo < r < ratio_hi) and (b < beta_thr)


def selftrip(drive, spin, beta=None, beta_thr=25.0, ratio_lo=0.6, ratio_hi=1.5,
             min_drive=30.0):
    """Coupled step-out ≈ the SUSTAINED 3-D wobble instability. Anchored on the ceiling:
    a genuine step-out means the body is desynchronised at the TOP level and stays that way.
    We find the onset of that FINAL sustained desync by scanning levels down from the top —
    which correctly ignores (a) the low-frequency spin-up over/undershoot and (b) transient
    wobble excursions that the body recovers from and re-tracks (as the short FW-6 does).
    Returns None if the body is still tracking at the ceiling (no step-out in range)."""
    levels = sorted({float(x) for x in drive if x >= min_drive})
    if not levels:
        return None
    track = {lv: _level_tracking(drive, spin, beta, lv, beta_thr, ratio_lo, ratio_hi)
             for lv in levels}
    if track[levels[-1]]:
        return None                              # tracking at the top → no step-out in range
    for i in range(len(levels) - 1, -1, -1):     # highest tracking level → step-out just above
        if track[levels[i]]:
            return float(levels[i + 1])
    return float(levels[0])                       # desynced across the whole swept band


def relock(drive, spin, beta=None, thr_lock=0.3, thr_desync=0.5, beta_lo=20.0, run_need=3):
    """First drive freq (scanning the DOWN branch order) where the body RE-LOCKS after
    having been desynchronised: sustained tracking within thr_lock AND small wobble
    (β<beta_lo, i.e. not a transient tumble overshoot). Returns None if the body stays
    desynchronised/tumbled through the whole down sweep (a wide, sticky hysteresis)."""
    desynced_once = False; run = 0
    for i in range(len(drive)):
        if drive[i] < 3:
            continue
        if spin[i] < thr_desync * drive[i]:
            desynced_once = True; run = 0; continue
        small_beta = (beta is None or beta[i] < beta_lo)
        if desynced_once and abs(spin[i] / drive[i] - 1.0) < thr_lock and small_beta:
            run += 1
            if run >= run_need:
                return float(drive[i])
        else:
            run = 0
    return None


def run_pass(gm, prev, levels, hold_steps, make_ext, mkey, sample_every, t0, tag):
    """Staircase the drive through `levels` (continuous state), sampling body spin +
    wobble β. Returns updated prev, arrays (drive, spin, beta), frames, blew(bool)."""
    drv, spn, bet, frames = [], [], [], []
    last = time.perf_counter(); step = 0; blew = False
    for lvl in levels:
        om = 2.0 * math.pi * float(lvl)
        for k in range(hold_steps):
            gm.step(make_ext(om, prev))
            prev = {nm: gm.get_node_state(nm) for nm in gm._nodes}
            if step % sample_every == 0:
                q = np.asarray(prev["body"]["orientation"]); w, x, y, zz = q
                beta = math.degrees(math.acos(max(-1.0, min(1.0, 2 * (x * zz + w * y)))))
                wz = float(rotate_vector_inverse(jnp.asarray(q),
                           jnp.asarray(prev["body"]["angular_velocity"]))[2])
                drv.append(float(lvl)); spn.append(abs(wz) / (2 * math.pi)); bet.append(beta)
                rp = np.asarray(prev[mkey]["rotor_pose_world"])
                frames.append({
                    "/World/Actors/UMR": (np.asarray(prev["body"]["position"]), q),
                    "/World/Actors/Magnet/Body": (rp[:3], rp[3:7]),
                    "/World/Actors/Magnet/Rotor": (rp[:3], rp[3:7])})
            if not np.all(np.isfinite(np.asarray(prev["body"]["position"]))):
                print(f"  [{tag}] NaN at step {step} (lvl {lvl:.0f} Hz)", flush=True)
                blew = True; break
            now = time.perf_counter()
            if now - last > 15.0:
                print(f"  [{tag}] lvl {lvl:.0f} Hz, step {step} ({now-t0:.0f}s)", flush=True)
                last = now
            step += 1
        if blew:
            break
    return prev, np.array(drv), np.array(spn), np.array(bet), frames, blew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["up", "down", "loop"], default="loop")
    ap.add_argument("--magnetic-model", choices=["single", "dual"], default="single")
    ap.add_argument("--hz0", type=float, default=5.0)
    ap.add_argument("--hz1", type=float, default=250.0)
    ap.add_argument("--hz-step", type=float, default=12.0)
    ap.add_argument("--hold-steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--mag-dipole", type=float, default=40.5)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--nu", type=float, default=2.33)
    ap.add_argument("--l-mm", type=float, default=7.47)
    ap.add_argument("--offcenter-cache", default=None)
    ap.add_argument("--design", default="FL9")
    ap.add_argument("--out", required=True, help="path prefix → <prefix>.npz/.png/.usdc")
    a = ap.parse_args()

    from mime.experiments import schwarz_vessel_helix as S
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    p = {"MAGNETIC_MODEL": a.magnetic_model, "INCLUDE_ARM": False, "BODY_MODEL": "inertial",
         "MAG_DIPOLE": a.mag_dipole, "DT": a.dt, "SWIM_MODE": "free",
         "NU_FL": a.nu, "L_UMR_M": a.l_mm * 1e-3}
    if a.offcenter_cache:
        p["OFFCENTER_CACHE"] = a.offcenter_cache
    print(f"[{a.design}] building {a.magnetic_model} graph (no-arm, inertial, "
          f"nu={a.nu}, L={a.l_mm}mm, MAG_DIPOLE={a.mag_dipole})...", flush=True)
    tb = time.perf_counter()
    gm = S.build_graph(p)
    ref = S.screw_points(p)
    print(f"[{a.design}] graph built ({time.perf_counter()-tb:.0f}s)", flush=True)
    nodes = set(gm._nodes)
    mkey = "motor" if "motor" in nodes else "motor0"
    z = 0.15; qm = [0.7071068, 0.0, 0.7071068, 0.0]
    pose0 = jnp.array([0.0, 0.0, z, *qm]); pose1 = jnp.array([0.0, 0.0, -z, *qm])

    def make_ext(omega, prev):
        pts = S.body_world_points(ref, prev)
        e = {"fvm": {"sample_points": pts, "forcing_points": pts}}
        if a.magnetic_model == "dual":
            e["motor0"] = {"commanded_velocity": jnp.float32(omega), "parent_pose_world": pose0}
            e["motor1"] = {"commanded_velocity": jnp.float32(omega), "parent_pose_world": pose1}
        else:
            e["motor"] = {"commanded_velocity": jnp.float32(omega), "parent_pose_world": pose0}
        return e

    up_levels = np.arange(a.hz0, a.hz1 + 1e-6, a.hz_step)
    prev = {nm: gm.get_node_state(nm) for nm in gm._nodes}
    t0 = time.perf_counter()

    # ── UP pass (always — reaches the running/desync state needed for the down branch)
    prev, dU, sU, bU, fU, blew = run_pass(gm, prev, up_levels, a.hold_steps, make_ext,
                                          mkey, a.sample_every, t0, "up")
    f_so_up = selftrip(dU, sU, bU) if len(dU) else None
    print(f"[{a.design}] up pass: f_so↑ = {f_so_up}  (β_max={bU.max():.0f}° if len)"
          if len(dU) else f"[{a.design}] up pass empty", flush=True)

    dD = sD = bD = np.array([]); fD = []; f_si_dn = None
    if a.direction in ("down", "loop") and not blew:
        if f_so_up is None:
            print(f"[{a.design}] WARNING: body did not desync on the up pass (still locked "
                  f"at {a.hz1:.0f} Hz) → down branch has no re-lock to find. Raise --hz1.",
                  flush=True)
        down_levels = up_levels[::-1].copy()
        prev, dD, sD, bD, fD, blew = run_pass(gm, prev, down_levels, a.hold_steps, make_ext,
                                              mkey, a.sample_every, t0, "down")
        f_si_dn = relock(dD, sD, bD) if len(dD) else None
        print(f"[{a.design}] down pass: f_si↓ = {f_si_dn}", flush=True)

    # ── save trace
    np.savez(a.out + ".npz", drive_up=dU, spin_up=sU, beta_up=bU,
             drive_dn=dD, spin_dn=sD, beta_dn=bD,
             f_so_up=(np.nan if f_so_up is None else f_so_up),
             f_si_dn=(np.nan if f_si_dn is None else f_si_dn),
             design=a.design, nu=a.nu, l_mm=a.l_mm, magnetic_model=a.magnetic_model)
    print(f"[{a.design}] saved {a.out}.npz", flush=True)

    # ── figure: hysteresis loop
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        dmax = max(dU.max() if len(dU) else 1, dD.max() if len(dD) else 1)
        if len(dU):
            ax.plot(dU, sU, ".", ms=2, alpha=0.5, color="tab:blue", label="ramp up (f_so↑)")
        if len(dD):
            ax.plot(dD, sD, ".", ms=2, alpha=0.5, color="tab:red", label="ramp down (f_si↓)")
        ax.plot([0, dmax], [0, dmax], "k--", lw=0.8, alpha=0.6, label="synchronous")
        if f_so_up:
            ax.axvline(f_so_up, color="tab:blue", ls=":", label=f"f_so↑≈{f_so_up:.0f}")
        if f_si_dn:
            ax.axvline(f_si_dn, color="tab:red", ls=":", label=f"f_si↓≈{f_si_dn:.0f}")
        ax.set_xlabel("drive frequency (Hz)"); ax.set_ylabel("body spin (Hz)")
        ax.set_title(f"Coupled step-out hysteresis — {a.design} "
                     f"(nu={a.nu}, L={a.l_mm}mm, {a.magnetic_model})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(a.out + ".png", dpi=130); plt.close()
        print(f"[{a.design}] saved {a.out}.png", flush=True)
    except Exception as e:
        print(f"[{a.design}] figure failed: {type(e).__name__}: {e}", flush=True)

    # ── usdc (up frames then down frames)
    try:
        write_usdc(fU + fD, a.out + ".usdc")
        print(f"[{a.design}] saved {a.out}.usdc ({len(fU)+len(fD)} frames)", flush=True)
    except Exception as e:
        print(f"[{a.design}] USDC write failed (npz is safe): {type(e).__name__}: {e}",
              flush=True)

    gap = (None if (f_so_up is None or f_si_dn is None) else f_so_up - f_si_dn)
    print(f"[{a.design}] RESULT: f_so↑={f_so_up}  f_si↓={f_si_dn}  gap={gap}  "
          f"total {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
