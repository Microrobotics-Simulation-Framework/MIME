# Couette Torque Benchmarks — Mass Conservation (FIXED) + Torque Overshoot (open)

**Original investigation**: 2026-05-20 — **Update / fix**: 2026-05-21
**Benchmarks**: MIME-VER-008, MIME-VER-009,
`TestBouzidiCI::test_torque_accuracy_under_5_percent`,
`test_bouzidi_convergence_order`
**Test file**: `tests/verification/test_ladd_cylinder.py`
**Status**: the moving-wall **mass-conservation bug is FIXED**
(`collide_bgk`, `mime.nodes.environment.lbm.d3q19`). The benchmarks remain
`xfail` — now blocked by a *separate* torque-overshoot bug, not the mass leak.

---

# UPDATE 2026-05-21 — mass leak diagnosed correctly and FIXED

## Corrected root cause — the BGK collision, not the bounce-back

The 2026-05-20 investigation below correctly established that a moving wall
drives a real, monotonic, ~Ω² mass leak. It **mis-attributed the cause** to a
"ghost-node duplication" in the bounce-back. That attribution is **wrong**.

Decisive test: a fully periodic box with **no solid walls and no bounce-back
at all**, carrying a non-uniform flow, leaks mass at the *same* ∝u² rate
(−2.8 × 10⁻⁷/step at |u| = 0.1, float64-measured). The bounce-back is not
involved.

The leak is in the **BGK collision** (`collide_bgk`). The collision is
`f_out = f − (f − f_eq)/τ`. In exact arithmetic `Σ_q f_eq = Σ_q f` (the D3Q19
moment identities `ΣW = 1`, `ΣW e = 0`, `ΣW (e·u)² = cs² u²` make the u²
terms cancel exactly), so the collision conserves mass. **In float32 that
cancellation is not exact** — the equilibrium does not sum to exactly the node
density, with a *systematic* residual of order u². The collision therefore
relaxes `f` toward a slightly-wrong mass every step; for a non-uniform (driven)
flow this integrates into the observed ∝Ω² leak. A static wall sustains no
flow (u ≡ 0), so no leak — which is why the 2026-05-20 "static-wall control"
*appeared* to implicate the bounce-back: it only ever showed "no flow → no
leak", never isolating the operator.

(The "duplication" measurement in **E3** below — `Σ_mm f_pc =
Σ_mm_in f_pre_opp = 154.70` — is arithmetically correct but does **not** prove
a net `Σf` leak: it shows the same mass is in two places at one instant. The
true per-step leak is the collision's, proven by the wall-free periodic box.)

## The fix

`collide_bgk` in `src/mime/nodes/environment/lbm/d3q19.py` — after the BGK
update, route the per-node mass residual into the rest population:

```python
f_out = f_out.at[..., 0].add(jnp.sum(f, axis=-1) - jnp.sum(f_out, axis=-1))
```

`e_0 = (0,0,0)`, so this changes mass only — momentum (`Σ_q f_out e_q`) is
exactly preserved. It makes the collision conserve `Σ_q f` to float32
round-off for every velocity, by construction (not by a global rescale).

## Verification — mass conservation

Relative `Σf` drift per step, float64-measured, D3Q19 `tau = 0.8`:

| case | OLD `collide_bgk` | FIXED `collide_bgk` |
|------|-------------------|---------------------|
| periodic box, no walls, \|u\|=0.1 | −2.8 × 10⁻⁷ | +1.3 × 10⁻¹¹ |
| Couette simple BB, Ω = 0.005 | −1.79 × 10⁻⁶ | +1.2 × 10⁻¹⁰ |
| Couette simple BB, Ω = 0.010 | −7.98 × 10⁻⁶ | +7.8 × 10⁻¹¹ |
| Couette Bouzidi, Ω = 0.005 | −2.02 × 10⁻⁶ | −2.3 × 10⁻⁷ |
| Couette Bouzidi, Ω = 0.010 | −8.51 × 10⁻⁶ | −4.6 × 10⁻⁷ |

- **Simple BB**: the ∝Ω² leak is gone — `Σf` conserved to round-off (~10⁻¹⁰,
  both signs) at every Ω.
- **Bouzidi**: the ∝Ω² leak is gone; a small **∝Ω** residual remains
  (~−2 × 10⁻⁷/step at Ω = 0.005). This is the well-known mass-conservation
  error of *interpolated* bounce-back — it comes from the Bouzidi
  interpolation formula itself (confirmed: present with the wall-velocity
  correction disabled), not from the collision. It is ~10× smaller than the
  fixed leak and acceptable as the interpolation residual.
- No bounce-back change was needed: with the collision fixed, simple-BB
  Couette conserves mass with the solid nodes left exactly as before.
- **IBLBMFluidNode** (the production rotating-UMR node) calls the same
  `collide_bgk` via `lbm_step_split`, so it inherits the fix directly — the
  correction is geometry-agnostic (it enforces the per-node zeroth moment for
  any flow). Measured with the fix, N=24 rotating UMR: `Σf` drift +5.3 × 10⁻⁸
  /step (simple BB), −7.5 × 10⁻⁸/step (Bouzidi) — vs the ~10⁻⁶/step
  ∝Ω² collision leak it had before. The small residual is *not* the
  collision (static-mask Couette simple BB conserves to ~10⁻¹⁰): it is the
  rotating-helix **mask change** ("fresh-node / refilling"), a separate,
  pre-existing concern beyond this fix. `TestBouzidiRegression` (a 200-step
  64³ IBLBM Bouzidi run) passes unchanged.

With mass conserved the rotating-Couette flow now **reaches a genuine steady
state** — the velocity field is flat to <0.1 % across 20 000 / 40 000 /
60 000 steps (it previously drifted without bound).

## The benchmarks remain blocked — by a SEPARATE bug

The 2026-05-20 report concluded the benchmarks were *impossible* because of the
mass leak. With the leak fixed they **still fail**. At the now-genuine steady
state:

- The **momentum-exchange torque overshoots** the analytical Couette torque by
  **~36 % (simple BB)** and **~17 % (Bouzidi)**.
- The converged **velocity profile itself is ~5 % below analytical**
  (`B_fit/B_analytical ≈ 0.95`), and the profile-implied torque
  `4πν·B_fit·n_z` matches analytical to ~5 % — so the momentum-exchange torque
  also disagrees with the profile-implied torque by ~30–40 %, which at a true
  steady state they must not.
- The error **grows with resolution** (simple-BB torque error 3 % at 32³ →
  20 % at 64³), so neither scheme shows its formal convergence order — this is
  why `test_bouzidi_convergence_order` now fails (it previously *passed by
  accident*, the mass-leak drift having contaminated the error ratio).

This is a **separate, pre-existing wall-BC / momentum-exchange defect** — the
"30–80 % torque overshoot" extensively investigated but never resolved in
`bouzidi_ibb_diagnostics.md`. It is independent of mass conservation and out
of scope for the mass-conservation fix.

MIME-VER-008, MIME-VER-009,
`TestBouzidiCI::test_torque_accuracy_under_5_percent` and
`test_bouzidi_convergence_order` therefore remain `xfail` — now blocked by the
torque overshoot, not the mass leak.

---

# UPDATE 2026-05-21 (b) — torque-overshoot root cause: an ill-conditioned MEM

Focused investigation of `compute_momentum_exchange_torque`, the remaining
blocker. All numbers at the converged, mass-conserving steady state.

## The flow is correct; the torque *measurement* is wrong

At the now-genuine steady state the velocity profile matches analytical
Couette to a few percent — the profile-implied torque `T_prof = 4πν·B_fit·n_z`
is −0.5 % at 64³, −5 % at 128³ (simple-BB O(dx) wall accuracy + O(Ma²)
compressibility). The momentum-exchange torque, by contrast, reads +13 % /
+22 % / +40 % high at 64³ / 96³ / 128³. The defect is in the torque
*measurement*, not the flow.

## The MEM torque is a near-cancellation of large opposite terms

Decomposing the MEM sum `T = Σ r × e·(f_pre[opp] + f_bb)` (inner cylinder,
128³, Ω = 0.001):

```
  torque of the incoming populations   2·Σ r×e·f_pre[opp]  ≈ -506
  torque of the Ladd wall correction      Σ r×e·corr       ≈ +520
  ----------------------------------------------------------------
  net MEM torque                                           ≈  +13     (T_prof ≈ +9)
```

The physical torque (~10) is a **~35 : 1 difference of two ~500-magnitude
terms**. This cancellation is intrinsic to the moving-wall MEM: the large
terms scale as `u_wall·ρ·(boundary area)`; the net is the viscous stress
`~ u_wall·ν/gap`. The cancellation ratio therefore scales as `gap/ν` — i.e.
**∝ resolution**.

## Consequence: every geometric error is amplified ∝ resolution

Because the result is a difference amplified by ~`gap/ν`, every sub-percent
modelling error in the large terms is magnified into a large torque error:
the lever arm uses the fluid-node position rather than the wall; the Ladd
correction uses `u_wall` at the fluid node (`∝ r_fluid > R1`); the curved
boundary is a staircase. Direct demonstration of the ill-conditioning:
changing the correction term by 2.5 % (evaluating `u_wall` at the wall vs the
fluid node) swings the 128³ result from ~+13 to ~+1. And because the
cancellation ratio grows with resolution, the amplified error — the
overshoot — **grows with resolution** (12.7 % → 40 %, 64³ → 128³) instead of
converging.

## Corroborating evidence

- **Inner cylinder** (convex, moving): MEM **overshoots +45 %**.
  **Outer wall** (concave, static): MEM **undershoots −18 %**.
  The true (bulk-profile) torque sits between them. At a steady state the
  torque is constant across the annular gap — the MEM giving two different
  answers on the two walls, neither equal to the bulk value, is proof the
  boundary MEM is unreliable here.
- An additive compressibility component `∝ Ma²` (Ω-sweep at fixed geometry:
  overshoot = ~30 % geometric + a term growing with Ω²).
- **NOT** Galilean non-invariance: the Wen et al. (2014) GI correction is
  `−Σ u_w·(f_α − f_α')`, and `f_α − f_α' = −corr` is small — orders of
  magnitude too small to account for the overshoot (a naive
  `Σ u_w·(f_α + f_α')` "correction" *adds* thousands of percent and is wrong).

## Recommendation

The boundary momentum-exchange torque is **ill-conditioned for moving curved
walls**; no small patch to `compute_momentum_exchange_torque` can make it
accurate. A reliable torque needs **viscous-stress integration on a control
surface in the bulk fluid** (well-conditioned — that is what `T_prof` does,
and it agrees with analytical to a few percent). This also affects
production: `IBLBMFluidNode` reports `drag_torque` via the same
`compute_momentum_exchange_torque`, so the UMR drag torque is overestimated
by the same mechanism, increasingly so at higher resolution.

---

# UPDATE 2026-05-21 (c) — torque-overshoot fix: a well-conditioned torque, and a deeper bug

Acting on the (b) investigation: a well-conditioned replacement for the
boundary momentum-exchange torque was implemented and validated, and in the
process a deeper bug surfaced.

## `compute_stress_torque_z` — the well-conditioned torque

New function in `mime/nodes/environment/lbm/bounce_back.py`: the torque about
the z-axis from integrating the r-theta momentum flux over a cylindrical band
of *bulk fluid* (a control surface), instead of the near-cancelling boundary
momentum exchange.

```
T = 2*pi*nz * < r^2 * Pi_rtheta >,   Pi_rtheta = rho u_r u_theta + sigma_rtheta
sigma_ab = -(1 - 1/(2 tau)) * sum_q e_qa e_qb (f_q - f_q^eq)   (viscous stress)
```

`Pi_rtheta` carries no isotropic (pressure) term, so it is an O(stress)
quantity — no catastrophic cancellation. Validated on Couette flow, the
stress torque measured in a band just outside the inner cylinder:

| n³ | MEM torque (old) | stress torque (new) |
|----|------------------|---------------------|
| 64 | +12.7 % | −0.5 % |
| 96 | +22.3 % | −1.1 % |
| 128 | +40.2 % | −0.8 % |

The ~+12–40 % MEM overshoot is gone; the stress torque is ~1 %,
well-conditioned and resolution-stable. `compute_momentum_exchange_torque`
now carries a docstring warning pointing to it. (It remains correct for
forces / static boundaries.)

## …but the fix surfaced a deeper bug: a stress "droop"

Integrating the stress across the *whole* annulus exposed that
`r²·Pi_rtheta` is **not constant across the gap**, which steady-state torque
balance requires. It droops from the inner cylinder outward — 128³:
`r²·Pi_rtheta ≈ −0.57` near the inner wall, `−0.39` near the outer. Both the
MEM and the stress method see the same inner-high / outer-low pattern (the
inner-cylinder MEM overshoots +45 %, the outer-wall MEM undershoots −18 %),
so it is a real flow feature, not a measurement artefact — and it makes the
stress torque band-dependent at large gaps.

A drooping `r²·sigma_rtheta` at a verified steady state means **angular
momentum is not conserved in the bulk fluid**. Confirmed: extending the
collision's mass-conservation correction to also restore the first moment
(momentum) **removes the droop** — `r²·Pi_rtheta` becomes flat and the torque
reads ~+1 % — so the collision's float32 *momentum* residual is the cause.
But that momentum re-injection is **numerically unstable** (the simulation
diverges within ~40 000 steps), so it was reverted: a correct, stable
momentum-conservation fix is a separate, harder problem.

## Status

- The torque-overshoot bug (ill-conditioned MEM) **is addressed** —
  `compute_stress_torque_z` is the well-conditioned method, accurate to <1 %
  where the annular gap is small (e.g. 64³).
- A clean, validated Couette torque benchmark at larger gaps additionally
  needs the collision **angular-momentum non-conservation** fixed (the
  droop). Its naive fix is unstable — an open problem.
- The benchmarks remain `xfail`: the MEM-overshoot half is now solved, but
  the droop still blocks an end-to-end validated torque.

---

# UPDATE 2026-05-21 (d) — the "droop" diagnosed: there is no deeper flow bug

The (c) "stress droop" was investigated as a suspected angular-momentum
non-conservation. A decisive diagnostic (converged 128³ Couette, all sums in
float64) settles it:

- Density `rho(r)` is **uniform** — 1.0000 ± 1e-4 across the annulus.
- The velocity profile is a **clean Couette profile**: fitting
  `u_theta = A·r + B/r` gives `A_fit/A = 0.997`, `B_fit/B = 0.992`, with a
  fit residual of only rms 2e-4 / max 8e-4 on `u_theta ≈ 0.008–0.028`
  (≈ 1–3 %).

So the **flow is correct** — a clean Couette flow ~1 % from analytical, well
within simple-bounce-back LBM accuracy (O(dx) wall + O(Ma²) compressibility).
**There is no deeper flow bug**; the (c) "angular-momentum non-conservation"
framing was a misdiagnosis.

The "droop" (`r²·sigma_rtheta` not constant) is therefore an **artefact of the
stress-tensor torque method**: the viscous stress is a velocity *derivative*,
so it amplifies the flow's small (~1 %) non-Couette imperfection; the stress
readout from `f_neq` additionally carries an r-dependent error (not fully
root-caused) that a clean Couette flow exposes. The MEM's inner-high /
outer-low pattern is its own ill-conditioning. Neither is a flow defect.

The collision's first-moment (momentum) residual IS real — ~1e-5 per node,
the float32 equilibrium-summation analogue of the mass bug — but it is
**benign**: the verified-clean velocity profile shows it does not
meaningfully corrupt the flow. (The (c) momentum-correction "removed the
droop" at 20 k steps only by injecting a summation-bias artefact, then
diverged — it was never a real fix.)

## Consequence for the benchmarks

The LBM **does** produce a correct Couette flow (~1 %). The benchmarks fail
only because they read the torque through the ill-conditioned MEM. The torque
is well-defined and is read robustly from the velocity-profile fit:
`T = 4·pi·nu·B_fit·nz` (`B_fit/B = 0.992` → ~0.8 % at 128³). Switching the
benchmark torque readout to the velocity-profile fit makes MIME-VER-008/009 a
genuine, passing benchmark — the recommended next step. `compute_stress_-
torque_z` remains available (well-conditioned; accurate at small gaps) with a
docstring caveat about the large-gap droop.

---

# UPDATE 2026-05-21 (e) — CORRECTION: (d) was a buggy measurement; the flow IS distorted

Attempting to wire the velocity-profile-fit torque into the benchmarks
(per (d)) exposed that **(d) itself was unreliable**. A convergence study,
then a reproducibility check — four bit-identical runs of the same 128³
config, chunked 4×10000 / 2×20000 / 1×40000 / 4×10000 — return the
**identical** result every time (the LBM is deterministic). Re-measuring the
velocity-profile fit on the converged flow gives a torque error of **7.5 %**,
not the 0.8 % that the (d) diagnostic (`droop_diag.py`) reported. (d)'s
`droop_diag` measurement was simply buggy.

Corrected, reproducible picture — velocity-profile fit (`u_theta = A r + B/r`)
of the converged flow, error vs the analytical Couette torque:

| n³ | velocity-fit torque error |
|----|---------------------------|
| 64 | ~0.9 % |
| 96 | ~4.0 % |
| 128 | ~7.5 % |

This error is reproducible, **grows with resolution**, is essentially the
same for simple BB and Bouzidi, and is margin-dependent (fitting different
radial sub-bands gives 6.8–9.4 % at 128³ — the converged profile is **not** a
single clean Couette curve; it is distorted).

So **(d) was wrong**: there IS a real, resolution-dependent distortion of the
simulated Couette flow (~7.5 % at 128³). Being scheme-independent it is a
bulk effect, not the bounce-back. Its root cause is **not established** —
candidates are compressible-Couette physics integrated over the (growing)
gap, a Reynolds-number effect, or a bulk collision error.

Both earlier framings were unreliable: (c)'s "angular-momentum
non-conservation" rested on an unstable, never-validated momentum
correction; (d)'s "no deeper bug" rested on the buggy `droop_diag`.

## Honest status of this whole investigation

- **Solid / verified**: the mass-conservation fix (`collide_bgk`) and the
  MEM-ill-conditioning diagnosis. The ∝Ω² mass leak is genuinely gone.
- **Not solved**: the Couette torque benchmark. *Every* torque-measurement
  method tried (boundary momentum exchange, bulk stress integration,
  velocity-profile fit) is defeated by a real, resolution-dependent,
  not-root-caused distortion of the simulated Couette flow. The benchmarks
  remain `xfail`. Root-causing that flow distortion — starting with a Mach
  sweep at fixed geometry to separate genuine compressible-Couette physics
  from a numerical defect — is the open task.

---

# UPDATE 2026-05-21 (f) — Mach sweep inconclusive; investigation halted

The Mach sweep recommended in (e) — 128³ fixed geometry, omega from 1e-4 to
1.6e-3 (Ma 0.0067 to 0.106), 40000 steps, velocity-fit torque error — does
**not** give a clean answer:

| omega  | Ma     | err (m=4 band) | density variation |
|--------|--------|----------------|-------------------|
| 0.0001 | 0.0067 |   0.08 % |  4.6e-4 |
| 0.0002 | 0.0133 |   6.15 % |  7.5e-4 |
| 0.0004 | 0.0266 |   5.28 % |  8.7e-4 |
| 0.0008 | 0.0532 |   6.76 % |  1.4e-3 |
| 0.0016 | 0.1064 |  13.26 % |  2.1e-3 |

The error is **not** a clean function of Ma: ~0 at the lowest Ma, then a jump
to ~6 % by Ma = 0.013, non-monotonic through the middle, rising again at the
top. It is **not** simple compressibility — the density stratification is
only ~0.1 % across the whole sweep, far too small to drive a 5–13 % torque
error. A `c0 + c2·Ma²` fit is poor (and would imply a ~3.4 % floor, but the
lowest-Ma point flatly contradicts it).

The one clean point: at the lowest Ma the deep-bulk flow is accurate
(~0.08 %), so the LBM **is** fundamentally capable of the correct Couette
flow. But the error's onset between Ma 0.007 and 0.013, and its messy
behaviour above, are unexplained.

## Investigation halted — honest close-out

Across a long multi-stage investigation (mass leak → collision fix → MEM
ill-conditioning → stress-torque method → the droop → reproducibility check →
Mach sweep) the residual Couette-torque flow distortion has **resisted clean
root-causing**. It is real, reproducible, resolution-dependent,
scheme-independent, and behaves messily. Further progress needs a dedicated
effort with deep LBM-numerics focus, not continued broad experiments.

**What stands, verified:** the mass-conservation fix (`collide_bgk`) and the
MEM-ill-conditioning diagnosis + `compute_stress_torque_z`. **Not resolved:**
the Couette torque benchmarks — they remain `xfail`. Promising lead for a
future effort: the error is near-zero at Ma ≈ 0.007 and turns on sharply by
Ma ≈ 0.013 — understanding that onset is the key.

---

# UPDATE 2026-05-21 (g) — ROOT CAUSE FOUND AND FIXED: GPU matmul precision (TF32)

A systematic hypothesis-and-falsification pass cracked it.

The raw ring-averaged velocity profile (no fit) showed the flow is
genuinely distorted — `u_theta/u_analytical` falls monotonically from 0.99
at the inner wall to **0.67 at the outer wall** (128³); density uniform,
`u_r` ~ 1e-6 (no secondary flow), flow steady. A steady, `u_r=0`,
uniform-density flow that deviates from `A r + B/r` requires a distributed
θ-momentum sink in the bulk; streaming conserves angular momentum exactly,
so the sink is the collision. Measuring the collision's per-node momentum
change (float64) confirmed it — a velocity-proportional drag draining
−6.6 angular momentum per step at 128³.

**ROOT CAUSE — GPU matmul precision (TF32).** The LBM moments are matmuls:
`momentum = f @ E` in `compute_macroscopic`, `e.u = velocity @ E^T` in
`equilibrium`. On GPU the default JAX matmul precision is TF32 (~10-bit
mantissa). A moment is a tiny residual of a near-cancellation of the
~0.05-magnitude populations — far below the TF32 granularity. TF32
corrupted the velocity, hence the equilibrium `f_eq`, so the collision
relaxed toward a wrong-momentum target — a spurious velocity-proportional
drag that drooped the Couette flow, worsening with resolution. Decisive
test: a batched `f @ E` returns **0.0** at default precision but the
correct **1.0e-5** at `precision="highest"`.

At very low speed the corruption is total: a forced Poiseuille flow
(velocity ~1e-5, below the TF32 granularity) freezes completely —
`u_center` never grows. This had been masked — `test_fvm_ibm.py` and
`test_kinematics.py` enable `jax_enable_x64` at module level, and x64
matmuls do not use TF32, so the LBM tests "passed" in the full slow lane
and only failed when run in genuine float32.

**FIX:** force `precision="highest"` on every LBM moment matmul —
`compute_macroscopic`, `equilibrium`, `guo_forcing` (d3q19 and d2q9), and
the momentum-exchange / stress-torque sums in `bounce_back.py`.

**RESULT** — velocity-profile-fit Couette torque error vs analytical:

| n³ | before fix | after fix |
|----|-----------|-----------|
| 64 | 0.9 % | 1.8 % |
| 96 | 4.0 % | 1.1 % |
| 128 | 7.0 % | 1.3 % |

The resolution-growth is eliminated; the residual ~1–2 % is the genuine
wall-position / compressibility accuracy (it no longer grows). Mass
conservation is preserved (drift ~1e-11/step). MIME-VER-008,
MIME-VER-009, `TestBouzidiCI::test_torque_accuracy_under_5_percent` and
`test_bouzidi_convergence_order` are **un-xfailed and pass** (MEM torque
1.6 % / 0.1 % / 0.4 %). The forced Poiseuille tests (`test_d3q19`,
`test_d2q9`) — previously frozen — now develop correctly. The fix also
shifts every LBM flow slightly (it is more accurate): the
`TestBouzidiRegression` IBLBM baseline was re-validated 17.4442 → 17.1138.

This closes the investigation. The two float32 defects — mass
non-conservation (UPDATE a) and moment-matmul precision (here) — are both
fixed. An intermediate "momentum non-conservation in the collision"
framing during the investigation was a symptom of this same TF32
corruption, not a separate bug.

---

> **The 2026-05-20 investigation below is retained as the historical record.**
> Its evidence that the mass leak is real, monotonic, ∝Ω², resolution-
> independent and not a benchmark-design artefact (sections E1, E2, E4, E5,
> E6) stands. Its **root-cause attribution — the bounce-back, the "Root cause"
> section and E3 — is superseded** by the UPDATE above: the leak is in the BGK
> collision, and it is now fixed. The "Verdict" and "Recommendation" sections
> below are likewise superseded.

---

## Verdict

The three rotating-cylinder Couette torque benchmarks **cannot be made into
genuine, converged accuracy benchmarks** at any resolution or step budget,
because the D3Q19 **moving-wall bounce-back does not conserve mass**. With a
rotating wall the total lattice mass leaks monotonically (≈ 4–5 × 10⁻⁷ of the
domain mass per step), so the flow never reaches a steady state: the velocity
field, and every torque derived from it, drift without bound.

This was investigated as a benchmark-redesign task (narrow well-resolved
annulus, realistic step counts, validated tolerances). The redesign spec
**cannot be satisfied** — there is no steady state to converge to. The correct
deliverable is therefore a written explanation (handoff deliverable **(b)**),
not a re-toleranced test.

This supersedes the open questions in `bouzidi_ibb_diagnostics.md` (2026-03-22):
that report chased a "30–80 % torque overshoot" through q-values, the Ladd
correction, and feedback amplification, but never checked mass conservation.
Mass non-conservation is the missing root cause.

---

## The benchmarks

| Test | ID | Scheme | Original config | Original bar |
|------|----|--------|-----------------|--------------|
| `test_couette_benchmark` | MIME-VER-008 | simple halfway BB | 128³, R₁=16, R₂=55, 10000 steps | error < 5 % |
| `test_bouzidi_benchmark` | MIME-VER-009 | Bouzidi IBB | 128³, R₁=16.3, R₂=55.3, 10000 steps | error < 1 % |
| `TestBouzidiCI::test_torque_accuracy_under_5_percent` | — | Bouzidi IBB | 64³, R₁=8.3, R₂=27.3, 5000 steps | error < 5 % |

Analytical reference (concentric-cylinder Couette, inner cylinder rotating at
Ω inside a static outer wall), verified correct:

```
u_θ(r) = A·r + B/r,   A = -Ω·R₁²/(R₂²-R₁²),   B = Ω·R₁²·R₂²/(R₂²-R₁²)
T      = -4π·ν·B·n_z   (per the n_z=3 periodic slices; ρ=1 so μ=ν)
```

---

## Evidence

All runs: D3Q19, `tau=0.8` → `nu=0.1`, float32 (LBM is float32 by design),
RTX A2000 GPU. Convergence/diagnostic runs used a `jax.lax.scan` harness that
reproduces the test functions' arithmetic exactly; the mass-conservation
results were re-confirmed against the **un-jitted** test code path.

### E1 — Error grows with resolution; nothing converges

Resolution-convergence study, **same** annulus geometry at every resolution
(R₁ = 0.30·n, R₂ = 0.46·n) at **constant Mach number** (u_wall = 0.04). A
genuine method must show error *decreasing* with resolution.

| n_grid | gap (cells) | simple BB error | Bouzidi error |
|--------|-------------|-----------------|---------------|
| 64  | 10.2 | ~10–12 % | ~0.0–0.7 % |
| 96  | 15.4 | ~19–22 % | ~16–18 % |
| 128 | 20.5 | ~32–40 % | ~15–19 % |

Error *increases* with resolution, and the torque **oscillates** between
successive 5000-step samples (0.5–3 % swings) instead of settling — there is
no converged value to read.

### E2 — Mass conservation: exact for a static wall, leaks for a moving wall

Total lattice mass Σρ (fluid + ghost-solid nodes), 128³, both cylinders:

| Configuration | mass @ 0 | mass @ 100 000 steps | drift |
|---------------|----------|----------------------|-------|
| Ω = 0   (static walls) | 49152.004 | 49152.012 | **+1.6 × 10⁻⁴ %** (float32 round-off) |
| Ω = 0.001 (rotating inner wall) | 49152.004 | 47140.344 | **−4.09 %** |

With a static wall, mass is conserved to float32 round-off. The instant the
inner wall rotates, total mass leaks monotonically. Confirmed identically for
inner-cylinder-only and outer-wall-only geometries, and for both the simple
and Bouzidi schemes. Confirmed in the **un-jitted** test code path (rules out
the scan/JIT harness).

### E3 — The leak is the bounce-back, not the Ladd velocity correction

Single instrumented step at a developed flow state (64³, R₁=16, R₂=28,
Ω=0.002):

```
leak with STATIC bounce-back (no wall correction)  :  -0.005859  / step
leak with MOVING bounce-back (with wall correction):  -0.005859  / step
Σ over the domain of the Ladd correction term      :  +0.000000   (exact)
```

The Ladd moving-wall correction `2·w·(e·u_wall)/cs²` sums to **exactly zero**
— it is mass-neutral, as it should be for a tangentially-moving wall. The
entire leak comes from the **core bounce-back replacement** itself.

### E4 — The leak scales as Ω²

Un-jitted test path, 64³, R₁=16, R₂=28, 2000 steps:

| Ω | Δ mass (2000 steps) | per-step rel. rate |
|------|---------------------|--------------------|
| 0.000 | +0.003 (round-off) | +1.2 × 10⁻¹⁰ |
| 0.001 | −1.588 | −6.5 × 10⁻⁸ |
| 0.002 | −7.703 | −3.1 × 10⁻⁷ |
| 0.004 | −31.99 | −1.3 × 10⁻⁶ |

Each doubling of Ω quadruples the leak: the imbalance is **second-order in the
wall speed** (the linear-in-velocity part cancels by wall impermeability; the
residual is the O(u²) compressible part of the bounce-back imbalance).

### E5 — The leak does not vanish under grid refinement

Same annulus fraction, same Mach number, developed flow:

| n_grid | rel. leak / step (steps 5k–15k) | (steps 15k–30k) |
|--------|---------------------------------|-----------------|
| 64  | 5.21 × 10⁻⁷ | 5.31 × 10⁻⁷ |
| 96  | 4.69 × 10⁻⁷ | 4.76 × 10⁻⁷ |
| 128 | 4.35 × 10⁻⁷ | 4.42 × 10⁻⁷ |

The relative leak rate is essentially **resolution-independent**. There is no
grid at which it refines away — so there is no grid at which the benchmark
becomes genuine.

### E6 — Consequence: no steady state, so no measurable torque

Long run, 128³ Bouzidi, R₁=38.4, R₂=58.88. The velocity profile is fit to
`u_θ/r = A + B/r²`; `B_fit/B = 1` would mean the field equals the analytical
Couette solution.

| step | mass | B_fit / B | T_profile err | T_mem (test) err |
|------|------|-----------|---------------|------------------|
| 10 000 | 48924 | 0.956 | 4.4 % | 19.1 % |
| 50 000 | 48015 | 0.979 | 2.1 % | 16.5 % |
| 90 000 | 47080 | 1.002 | 0.2 % | 13.4 % |
| 150 000 | 45637 | 1.034 | 3.4 % | 13.3 % |

`B_fit/B` sweeps *through* 1.0 and keeps climbing — the flow accelerates
without bound as mass bleeds away (less mass carrying the same wall-injected
momentum). There is **no step count at which the flow sits at the analytical
solution**. The "convergence" seen in earlier 10k–80k sweeps is this slow
mass-leak drift, not viscous spin-up (viscous spin-up across a 20-cell gap is
~10³ steps; the observed drift runs for >10⁵ steps).

The crossing `B_fit/B = 1` is reached at a wildly config-dependent step count
(~6–10k at 64³, ~88k at 128³) — tuning `n_steps` to land on it would be
exactly the "reverse-engineered to pass" anti-pattern the redesign spec
forbids, and what lands on it is a *single drifting instant*, not a steady
state. At 64³ the crossing happens early, after only ~0.5 % mass loss, so the
momentum-exchange torque the tests assert on (`T_mem`) does momentarily read
the analytical value — this is the fleeting transient that the original 64³
test was implicitly (and unknowingly) catching. At 128³ the crossing is not
reached until ~88k steps, by which point the field has already lost ~6 % of
its mass, and `T_mem` reads 13 % high while the profile says 0.2 %. So the
momentum-exchange torque and the profile-implied torque agree only at a
fleeting, low-mass-loss transient and disagree by 13–33 % otherwise — the
physics-bug guard in the redesign spec ("at steady state they must match"),
triggered: there is no steady state at which to check.

---

## Root cause

`apply_bounce_back` / `apply_bouzidi_bounce_back` deliberately keep solid
("ghost") nodes populated and let them collide and stream as a ghost fluid
(see the comment at `bounce_back.py:139–141`, "Zeroing them causes mass
leakage"). The bounce-back then, at each boundary link, replaces the
post-stream population at the fluid node with a copy of *that fluid node's
own pre-stream population in the opposite direction*:

```
f_bb[x, q] = f_pre[x, opp_q]      (at links where x − e_q is solid)
```

But that same population `f_pre[x, opp_q]` **also streamed normally into the
adjacent ghost node** `x − e_q`, which the bounce-back does not touch. So the
population is **duplicated** (it now exists at both the fluid node and the
ghost node), while the population that streamed *out of* the ghost node into
the fluid node — `f_post[x, q]` — is **discarded**.

Per boundary link the mass change is `f_pre[x, opp_q] − f_pre[x−e_q, q]`:

- For a **uniform** field (`f ≡ W`, the rest state) this is `W[opp_q] − W[q] =
  0` — mass is conserved exactly. This is why a static wall, which never
  develops a flow, conserves mass perfectly (E2).
- For a **developed flow** the two populations differ. With a static wall any
  flow decays, so the imbalance is a bounded, self-correcting transient
  (verified: an initialised swirl with static walls perturbs mass by −1.3 then
  recovers to −0.03 within 2000 steps). With a **continuously driven**
  rotating wall the flow never decays, so the per-link imbalance is sustained
  and **accumulates without bound** — the observed monotonic leak.

The Ladd moving-wall velocity correction is **not** at fault (E3): it sums to
zero. The defect is structural in the ghost-node bounce-back formulation.

---

## Why this is a physics bug, not a benchmark-design problem

The redesign spec was applied in full: a narrow, well-resolved annulus
(R_inner ≥ 16 cells, gap 12–22 cells), constant Mach number, low Mach number,
multi-resolution convergence study, step budgets up to 150 000. Every one of
those redesigned configurations **still fails**, and fails *worse* at higher
resolution (E1). The blocker — mass non-conservation — is independent of
geometry, resolution (E5), and step count, and is present in the LBM core
(`bounce_back.py`), unchanged since the v0.1.0 release tag. No choice of
benchmark parameters can produce a converged steady state when the governing
discretisation does not conserve mass.

---

## Recommendation (LBM source — out of scope for this task)

A correct bounce-back for this domain must not let ghost nodes accumulate
duplicated populations. Standard fixes:

1. Treat solid nodes as genuine no-slip nodes (full bounce-back on the solid
   side as well), so streaming into the solid is reflected rather than stored;
   or
2. Use a wet-node / link-wise bounce-back that conserves mass per link by
   construction (the population reflected back is the *same* population that
   streamed toward the wall, not a copy).

Until then the rotating-cylinder Couette torque benchmarks are not realisable.
The static-wall paths (Poiseuille, lid-less channels) are unaffected — mass is
conserved exactly there (E2).

---

## Reproduction

```bash
source /home/nick/MSF/msf/.venv/bin/activate
cd /home/nick/MSF/msf/MIME
python -m pytest -q tests/verification/test_ladd_cylinder.py \
    -m 'slow or not slow'
```

The three benchmarks above report as `xfail`. The minimal mass-conservation
check (E2) is a six-line script: run `lbm_step_split` + `apply_bounce_back`
with a rotating `wall_velocity` and watch `jnp.sum(f)` decrease; set Ω = 0 and
it is constant to float32 round-off.
