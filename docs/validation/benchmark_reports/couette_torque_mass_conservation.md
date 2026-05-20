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
