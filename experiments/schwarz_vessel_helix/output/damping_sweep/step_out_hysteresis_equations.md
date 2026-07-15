# Step-out equations for the hysteretic (underdamped) regime — with validity checks

*Companion to the damping-ratio sweep (`plans/MIME_EP1_damping_sweep.md`). Equations are
validated against `mod1_scaling_sweep.jsonl` (see the numbers inline) and reduce to the
standard overdamped Adler/Man–Lauga step-out in the appropriate limit.*

## 1. Reduced model

A hard-magnetic helical swimmer driven by a field rotating at Ω_d, with magnetic moment m,
field magnitude B (write k ≡ mB), rotational drag C about the spin axis, and axial moment of
inertia I. In the frame co-rotating with the field, the phase lag φ = ψ − Ω_d t obeys the
**driven damped pendulum** (identical to the RSJ / Stewart–McCumber Josephson equation):

    I φ̈ + C φ̇ + mB sin φ = −C Ω_d.                                              (1)

Nondimensionalise with the natural frequency ω_n = √(mB/I) and τ = ω_n t (′ ≡ d/dτ). Using
the damping ratio ζ ≡ C/(2√(I·mB)) = C/(2 I ω_n) and normalized drive ν ≡ Ω_d/ω_n:

    φ″ + 2ζ φ′ + sin φ = −2ζ ν.                                                  (2)

Everything below depends on only **two numbers: ζ (damping ratio) and ν (drive/ω_n).**

## 2. The two step-out thresholds

**(a) Pull-OUT — loss of lock (ramp-up step-out): saddle-node of the locked state.**
A phase-locked solution (φ = const) requires sin φ = −2ζν, which has a real root iff |2ζν| ≤ 1:

    ┌─────────────────────────────────────────────────────────────────┐
    │  ν_so = 1/(2ζ)      ⇔      f_so = mB/C = ω_n/(2ζ).                 │   (3)
    └─────────────────────────────────────────────────────────────────┘

This is **exact for all ζ**, inertia-independent, and is *identical* to the standard
overdamped Adler / Man–Lauga step-out. Validated: `f_so_sim/ω_n` tracks 1/(2ζ) to within
−4…+4 % over the entire underdamped range (`mod1_scaling_sweep.jsonl`).

**(b) Pull-IN — recovery of lock (ramp-down re-lock): homoclinic bifurcation of the running
state.** The whirling (stepped-out) solution persists below f_so, down to the drive at which
the running limit cycle collides with the saddle. A Melnikov energy balance on the pendulum
separatrix (φ′ = 2 cos(φ/2), so ∮φ′² dτ = 8 and the constant torque does 2π·(2ζν) per turn)
gives 2π(2ζν) = 2ζ·8, i.e.

    ┌─────────────────────────────────────────────────────────────────┐
    │  ν_si = 4/π         ⇔      f_si = (4/π) ω_n = (4/π)√(mB/I).        │   (4)
    └─────────────────────────────────────────────────────────────────┘

Leading order in ζ. Validated: `f_si_sim/ω_n` sits at 1.27–1.33 vs 4/π = 1.273 (±8 %) across
ζ = 0.008 … 0.40 — i.e. essentially constant through the whole hysteretic band.

> **Attribution.** Eq. (4) is *exactly* the Stewart–McCumber **retrapping current** of the RCSJ
> Josephson junction, written in mechanical variables: I_r/I_c = 4/(πQ) with Q = √(mB·I)/C, so
> C·Ω_si = (4/π)mB/Q ⟹ Ω_si = (4/π)√(mB/I) — **the drag C cancels identically**. See Stewart
> (1968), McCumber (1968), Strogatz *Nonlinear Dynamics & Chaos* §8.5 (homoclinic bifurcation of
> the driven damped pendulum). We do **not** claim this formula as new physics; the contribution is
> identifying it as the *lower* step-out threshold for a magnetically actuated helical robot. The
> pull-in < pull-out asymmetry itself is the "hunting" / pull-in-torque phenomenology of
> synchronous-machine theory (which, however, gives no clean (4/π) closed form).

## 3. Unified master relation (collapses to the standard result)

The synchronization boundary is a *hysteresis band* [f_si, f_so]. Both branches can be written
as single expressions that automatically collapse to the classical single-valued step-out:

    ┌───────────────────────────────────────────────────────────────────────────┐
    │  f_pull-out = ω_n / (2ζ)              = mB/C            (loss of lock, up)   │
    │  f_pull-in  = ω_n · min( 1/(2ζ),  4/π )                (re-lock, down)       │   (5)
    │  Δf_hyst    = f_pull-out − f_pull-in  = ω_n · max( 0,  1/(2ζ) − 4/π )        │
    └───────────────────────────────────────────────────────────────────────────┘

- The `min`/`max` are the collapse switches. For **ζ ≥ π/8** they give f_pull-in = f_pull-out =
  mB/C and Δf_hyst = 0 → the **standard overdamped Adler step-out, no hysteresis**.
- For **ζ < π/8** they give the underdamped band [ (4/π)ω_n , mB/C ] with positive width.
- The two branches meet exactly at the critical damping ratio

    ┌─────────────────────────────────────────────────────────────────┐
    │  ζ_c = π/8 ≈ 0.3927     (hysteresis for  ζ ≲ ζ_c, leading order). │   (6)
    └─────────────────────────────────────────────────────────────────┘

  because 1/(2ζ_c) = 4/π. (Verified: the sim window closes at ζ = 0.393.) **Soften this in the
  paper:** ζ_c = π/8 comes from equating the *weak-damping asymptotic* f_si to f_so, so it is a
  leading-order estimate, not the exact bifurcation boundary — which is an O(1) constant near the
  conventional Josephson value β_c ~ 1 (π/8 corresponds to β_c ≈ 1.6 vs the textbook β_c = 1;
  ~60 % apart because the (4/π) form is being extrapolated to where it is no longer valid). The
  exact onset needs numerical continuation. This is irrelevant to real mm devices: FL-9 has
  β_c = Q² = 1/(4ζ²) ≈ 1000–1300, three orders of magnitude inside the underdamped regime.

**Observed (protocol-dependent) step-out.** Within the band the measured value depends on how
the device is driven: a clean ramp-up from rest follows the high branch → f_so; a perturbed or
ramped-down device drops to the floor → f_si. de Jongh's measured 27 Hz for the 9-mm UMR sits
just above f_si = (4/π)ω_n ≈ 20–25 Hz (a finite-perturbation device sitting near the homoclinic
floor), which is *impossible* in an overdamped model (ceiling ω_n/2 ≈ 8.7 Hz) — direct evidence
for this underdamped regime.

## 4. When do these equations apply? (three validity checks)

**Check 1 — regime (ζ vs ζ_c).** Compute ζ = C/(2√(I·mB)).
- ζ ≥ π/8 → **overdamped**: single step-out f = mB/C, Eq. (3); Eqs. (4)–(6) give no hysteresis.
- ζ < π/8 → **underdamped hysteretic**: use the full band, Eq. (5).
All 17 de Jongh designs have ζ = 0.013–0.017 ≪ π/8 (deeply hysteretic); true µm swimmers have
ζ ≳ 1 (overdamped — why the classical literature used Adler). The crossover (ζ = π/8) is at body
length ≈ 0.30 mm (isometric scaling of FL-9).

**Check 2 — Melnikov accuracy for the floor.** Eq. (4)’s leading-order 4/π is accurate to a few
percent for ζ ≲ 0.25 (verified). Near the merge the exact homoclinic curve bends up to meet f_so,
but because the merge is reached mainly by f_so = 1/(2ζ) *descending* to 4/π, the constant-4/π
floor stays a good approximation right up to ζ_c. Outside ζ < π/8 the floor is not defined
(no running state to lose).

**Check 3 (the important one) — 1-DOF axial pendulum vs 3-D wobble instability.** Eqs. (1)–(5)
assume the body spins about a *fixed* axis (pure axial phase-slip). The coupled two-scale runs
show this can be *preempted* by a **transverse wobble instability**: for elongated screws in the
RPM’s field gradient the spinning body tumbles (β → 90°) at a drive **well below** the axial
saddle-node. Coupled evidence (`mod1_figC_coupled.png`):
- FL-9 (L/R_cyl ≈ 4.8) and FW-1 (≈ 6.2): wobble step-out at f ≈ 215–230 Hz — far below the 1-DOF
  f_so = 621 Hz. The pendulum f_so is then only an *upper bound*.
- FW-6 (stubby, L/R_cyl ≈ 2.6): wobble-stable, tracks past 260 Hz (no step-out in range).

So the operative step-out is **f_step = min( f_so^(1-DOF), f_wobble )**, where f_wobble is set by
the transverse alignment mode ω_⊥ = √(mB/I_⊥) (I_⊥ = I_transverse ≈ m(3a²+L²)/12) and the field
gradient, and has no simple closed form — it must come from the coupled solver. Practical
criterion: the 1-DOF equations hold when the body is rotationally stiff/near-isometric
(I_⊥/I_∥ ≈ 1, small gradient); they *over-predict* the step-out for slender bodies (I_⊥ ≫ I_∥)
in strong gradients, where f_wobble governs. **The homoclinic floor f_si, by contrast, remains the
relevant *low* threshold in all cases** (it matches the coupled sticky-tumble hysteresis and the
de Jongh 27 Hz).

**Secondary caveats.** (i) C is taken quasi-steady; if the unsteady/Basset number β_B = √(ωa²/ν)
is not ≪ 1 at the operating frequency, replace C → C(ω). (ii) Hard-magnetic lock assumed (moment
fixed in the body); (iii) field uniform over the body for the pendulum reduction (gradient enters
only the wobble term of Check 3).

## 5. Wall / confinement effects — where the wall enters

The equations' **form is wall-agnostic** (it is just the driven pendulum). The wall enters through
exactly **one parameter: the rotational drag C** (and, negligibly for a dense body, added-inertia
in I). mB is magnetic (wall-independent). So the *structure* — the two thresholds, ζ_c = π/8, the
collapse — is universal; only the *value* of C is confinement-specific. Concretely, holding the
FL-9 body fixed and varying the cylindrical confinement ratio R_ves/R_cyl (centred BEM, all other
inputs fixed; data `mod1_wall_ratio_sweep.jsonl`, figure **`mod1_figE_wall_ratio.png`**):

| ratio R_ves/R_cyl | C = R[5,5] (N·m·s) | ζ | ω_n (Hz) | **f_so = mB/C** | **f_si = (4/π)ω_n** |
|---|---|---|---|---|---|
| 1.526 (tight)   | 8.07e-10 | 0.0249 | 19.8 | 398 Hz | 25.2 Hz |
| **2.035 (ours, ¼″ tube)** | **5.16e-10** | **0.0159** | 19.8 | **621 Hz** | **25.2 Hz** |
| 2.500           | 4.55e-10 | 0.0140 | 19.8 | 706 Hz | 25.2 Hz |
| 3.054           | 4.24e-10 | 0.0131 | 19.8 | 756 Hz | 25.2 Hz |
| 4.071           | 4.03e-10 | 0.0124 | 19.8 | 796 Hz | 25.2 Hz |
| 8.333 (loose)   | 3.89e-10 | 0.0120 | 19.8 | 826 Hz | 25.2 Hz |

Reading this off:
- **f_so = mB/C IS wall-sensitive** (∝ 1/C): tighter confinement raises the drag (×1.33 at our 2.035
  vs near-unbounded 8.33; ×2 at the tightest 1.526) and *lowers* the pull-out step-out (826 → 398 Hz).
- **f_si = (4/π)ω_n is wall-drag-INDEPENDENT (to leading order)** — it contains no drag term
  (ω_n = √(mB/I) depends only on the
  magnetic torque and the inertia). The homoclinic floor is **25.2 Hz at every ratio**. So the
  physically-relevant *low* threshold — the one that matches de Jongh's anomalous 27 Hz — is wall-
  *drag*-invariant. Likewise ζ_c ≈ π/8 (regime boundary) is universal; only ζ itself moves.
  **Residual qualifier:** this is "independent of the wall's *drag* contribution to leading order."
  If tight confinement modifies the *effective inertia* I (hydrodynamic added-mass coupling), f_si
  = (4/π)√(mB/I) picks up a weak residual through I — accurate to state as "wall-drag-invariant,"
  not "wall-invariant." The distinction (the retrapping *current* I_r = CΩ_si IS drag-sensitive; the
  retrapping *frequency* Ω_si = I_r/C is not, because the two C's cancel) is exactly what makes the
  observation non-obvious.

**So the equations are not tied to our particular ratio.** They apply to any confinement (or free
space) once C is evaluated for that geometry: our numbers (ζ=0.016, f_so=621 Hz) are the ratio-2.035
*centred* values; plug in the C for a different wall and the same equations give the new f_so and ζ,
while f_si and ω_n stay put.

**Three wall subtleties beyond "just rescale C":**
1. **Off-centre / near-wall position.** The table above is centred; a dense screw actually rides the
   tube floor, where the confined drag is higher still (the off-centre R(d) grid) — so our *coupled*
   f_so is pushed below the centred 621. C is a function of radial position, not just the ratio.
2. **Off-diagonal wall coupling.** Near a wall the 6×6 resistance develops rotation↔translation and
   transverse couplings that the scalar-C (R[5,5]) reduction ignores. Secondary for the axial
   phase-slip, but they are the *seed* of point 3.
3. **The dominant wall effect is the wobble instability, not C.** In our confined runs the wall +
   field gradient drive the transverse tumble (Regime C, Check 3) that preempts the axial pull-out —
   the coupled step-out (215–230 Hz) is set by that wall/gradient/geometry physics, **not** by the
   closed-form f_so at all. That part is genuinely case-specific and needs the coupled solver; the
   pendulum equations bound it from above (f_so) and below (the wall-independent f_si).

## 6. Practical recipe

1. Compute ω_n = √(mB/I), ζ = C/(2√(I mB)) (C = |R[5,5]| from the confined BEM; I = ½ m a²).
2. If ζ ≥ π/8: report the single overdamped step-out f = mB/C (Eq. 3). Done.
3. If ζ < π/8: report the hysteresis band [f_si, f_so] = [(4/π)ω_n, mB/C] (Eqs. 3–5); state the
   protocol (ramp-up → f_so; ramp-down/perturbed → f_si).
4. Check elongation/gradient (Check 3): if slender + gradient, the *upper* branch is replaced by
   the coupled wobble step-out f_wobble ≤ f_so; the floor f_si still holds.
