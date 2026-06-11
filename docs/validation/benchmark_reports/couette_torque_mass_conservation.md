# Couette Torque Benchmarks — Mass Conservation & TF32 Precision (RESOLVED)

**Investigation:** 2026-05-20 → 2026-05-21
**Benchmarks:** MIME-VER-008, MIME-VER-009,
`TestBouzidiCI::test_torque_accuracy_under_5_percent`,
`test_bouzidi_convergence_order`
**Test file:** `tests/verification/test_ladd_cylinder.py`
**Status:** **RESOLVED.** Two genuine float32 defects in the D3Q19 LBM were
found and fixed. All four Couette benchmarks are un-xfailed and pass; the
three pre-existing `test_d3q19` / `test_d2q9` momentum failures are fixed.

---

## Summary

The rotating-cylinder Couette torque benchmarks failed (torque error ~27 % at
128³, *growing* with resolution; the flow never reaching a steady state). The
cause was **two distinct float32 defects in the LBM core** — neither a
benchmark-design problem, neither a v0.2 regression (both present since the
`v0.1.0` tag).

## Defect 1 — BGK collision mass non-conservation (fix: commit `84016bf`)

The BGK collision `f_out = f − (f − f_eq)/τ` conserves mass exactly in exact
arithmetic (the D3Q19 moment identities make the u² terms cancel). **In float32
that cancellation is not exact** — the equilibrium polynomial does not sum to
exactly the node density, leaving a systematic O(u²) residual. Any non-uniform
flow then loses mass ~1e-6/step; with a continuously-driven rotating wall this
never decays and integrates into an unbounded ∝Ω² drift, so the flow never
reaches a steady state.

**Fix** — in `collide_bgk` (`d3q19.py`), route the per-node residual into the
rest population (`e₀ = (0,0,0)`, so it is momentum-neutral):

```python
f_out = f_out.at[..., 0].add(jnp.sum(f, axis=-1) - jnp.sum(f_out, axis=-1))
```

Verification — relative `Σf` drift per step (float64-measured, `tau = 0.8`):

| case | before | after |
|------|--------|-------|
| Couette simple BB, Ω = 0.005 | −1.79e-6 | +1.2e-10 |
| Couette simple BB, Ω = 0.010 | −7.98e-6 | +7.8e-11 |
| Couette Bouzidi, Ω = 0.005   | −2.02e-6 | −2.3e-7 |

The ∝Ω² leak is gone (round-off). Bouzidi keeps a small ∝Ω residual
(~2e-7/step) — the known mass-conservation error of *interpolated* bounce-back,
from the interpolation formula itself; ~10× smaller and acceptable.

## Defect 2 — GPU TF32 matmul precision (fix: commit `ccc0d53`)

The LBM moments are matmuls (`momentum = f @ E` in `compute_macroscopic`,
`e·u = velocity @ Eᵀ` in `equilibrium`). On GPU, JAX's default float32 matmul
precision is **TF32** (~10-bit mantissa). A moment is a tiny residual of a
near-cancellation of the ~0.05-magnitude populations — far below TF32
granularity.

**Decisive test:** a batched `f @ E` for a near-rest field returns **exactly
`0.0`** at default precision vs the correct **`1.0e-5`** at
`precision="highest"`. TF32 corrupted the velocity → the equilibrium → the
collision relaxed toward a wrong-momentum target → a spurious
velocity-proportional drag that drooped the Couette profile
(`u/u_analytical` 0.99 at the inner wall → 0.67 at the outer, 128³), worsening
with resolution. At very low speed the corruption is total — a forced
Poiseuille flow (~1e-5 velocity) froze completely (the pre-existing
`test_d3q19` / `test_d2q9` failure).

It was masked because `test_fvm_ibm.py` / `test_kinematics.py` used to enable
`jax_enable_x64` at *module scope* (x64 matmuls don't use TF32), which leaked
x64 across the whole session, so the LBM tests passed in the full slow lane
and failed only in genuine float32. (v0.2 removed that leak: x64 is now opt-in
per test via `@pytest.mark.x64` + an autouse `conftest.py` fixture, so this
class of masking can no longer happen — see the
[v0.2 release notes](../../release_notes/v0.2.md).)

**Fix** — `precision="highest"` on every LBM moment matmul:
`compute_macroscopic`, `equilibrium`, `guo_forcing` (d3q19 + d2q9), and the
momentum-exchange / stress-torque sums in `bounce_back.py`.

Result — velocity-profile-fit Couette torque error vs analytical:

| n³ | before fix | after fix |
|----|-----------|-----------|
| 64  | 0.9 % | 1.8 % |
| 96  | 4.0 % | 1.1 % |
| 128 | 7.0 % | 1.3 % |

The resolution-growth is eliminated; the residual ~1–2 % is genuine
wall-position / compressibility accuracy. MEM-torque error at the benchmark
resolutions: 1.6 % / 0.1 % / 0.4 %.

## Outcome

- MIME-VER-008, MIME-VER-009,
  `TestBouzidiCI::test_torque_accuracy_under_5_percent`,
  `test_bouzidi_convergence_order` — **un-xfailed and passing**.
- `test_d3q19` / `test_d2q9` forced-Poiseuille — previously frozen — now
  develop correctly.
- `TestBouzidiRegression` IBLBM baseline re-validated `17.4442 → 17.1138`
  (the `17.4442` value was itself TF32-corrupted).
- LBM suite (`test_ladd_cylinder.py` + `tests/nodes/lbm/`): **194 passed**.

## Production impact — `IBLBMFluidNode`

The production rotating-UMR node uses the same `collide_bgk` (via
`lbm_step_split`) and the same moment matmuls, so it inherits both fixes
directly. Measured with the fixes, an N=24 rotating UMR has `Σf` drift
+5.3e-8/step (simple BB) / −7.5e-8/step (Bouzidi) — down from the ~1e-6/step
∝Ω² collision leak. The small remaining residual is **not** the collision
(static-mask Couette conserves to ~1e-10): it is the rotating-helix **mask
change** ("fresh-node / refilling"), a separate, pre-existing concern beyond
this fix — flagged for follow-up.

## Systemic warning — TF32 beyond the LBM

TF32 silently degrades **any** low-magnitude float32 matmul on GPU. It was
degrading the LBM by up to 100 %+ and went unnoticed for months. Other
precision-sensitive float32 matmul paths — the FVM dense-DCT pressure solver,
the Stokeslet BEM, and the Pallas/Triton LBM kernel (`pallas_lbm.py`, left out
of scope here) — should be audited. Prefer per-physics-path
`precision="highest"` over a global flag (the GNN/MLP surrogates are
TF32-tolerant; a global flag would needlessly slow them).

## Superseded diagnoses (historical record)

The route to the root cause took two wrong turns, recorded here for honesty:

1. **"Ghost-node bounce-back duplication"** (2026-05-20). The original
   benchmark-redesign investigation correctly established that the mass leak is
   real, monotonic, ∝Ω² and resolution-independent, but mis-attributed it to
   the bounce-back. Decisive falsification: a fully periodic box with no walls
   and no bounce-back, carrying a non-uniform flow, leaks at the same ∝u² rate
   — the cause is the collision, not the bounce-back.
2. **"A separate torque-overshoot bug."** An intermediate framing held the mass
   leak and the torque overshoot to be independent defects. The overshoot was
   the TF32 corruption; both symptoms are covered by the two fixes above.

This report supersedes the open questions in `bouzidi_ibb_diagnostics.md`
(2026-03-22), which chased the "30–80 % torque overshoot" through q-values and
the Ladd correction but never checked mass conservation or matmul precision.

## Reproduction

```bash
source /home/nick/MSF/msf/.venv/bin/activate
cd /home/nick/MSF/msf/MIME
python -m pytest -q -m 'slow or not slow' tests/verification/test_ladd_cylinder.py
```

TF32 check: a batched `f @ E` (LBM-field-sized, near-rest distribution) returns
`0.0` at default GPU precision and `1.0e-5` at `precision="highest"`.
