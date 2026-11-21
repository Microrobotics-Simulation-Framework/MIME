# w7fib predictions (committed before any arm; 2026-11-21 17:43 CET)

A needle into a fibrin-like gel on a bond carrier: bonds (horizon 3 spacings) with a shared volumetric pressure, the surface
correction, a blend with a continuum, a softening bond law, alignment at a crack tip. Preflight checks only (1-D blend sums, the
stencil's tensor, per-node surface energy) were run before this file. P = my probability.

## Arm A: sphere (R 5 mm, d/R 0.01-0.05) and flat punch (a 1 mm) on a bond zone blended into a graded continuum block
(quarter, 20 mm, bottom fixed), E 50 kPa, nu 0.25 / 0.4 / 0.45, h 0.2 / 0.141 / 0.1 mm (a_c/h about 2.5 to 11)
- A0 harness (continuum only) within 2 % of Hertz x Sneddon x block factor at h 0.1, nu 0.4: P 0.8.
- No correction: F/F_ref - 1 between -3 % and -12 % (a soft surface layer), shrinking with h roughly as h / a_c: P 0.6.
  Error under 15 % at every h with a_c/h >= 5 (i.e. NOT the 15-40 % band): P 0.75.
- Volume correction (2 V0 / (V_i + V_j)): error roughly halved, still over 2 % at h 0.2: P 0.5.
- Ghost bonds through the local gradient (exact under uniform strain): within 2 % of the continuum on the same grid at every h:
  P 0.6; within 3 % of Hertz x Sneddon x block at h 0.1: P 0.6.
- The surface error larger at nu 0.45 than at nu 0.25 (the pressure carries volume without a surface effect, shear is all bonds):
  P 0.5.
- Flat punch: the bond / continuum gap larger than the sphere's at equal h: P 0.6.
- Kill (spike 3's line): the contact force not within 10 % of Hertz at any correction tried. P(kill met) 0.15.

## Arm B: the blend, 3-D residual forces
- Affine field, isotropic-calibrated stencil, continuum partner with the same tensor: residual follows the 1-D scaling (linear ramp
  ~1/w at its kinks, C1 joins ~1/w^2, C2 ~1/w^3) within a factor 3 of the 1-D numbers: P 0.5.
- Constant-micromodulus stencil (13 % off isotropy) against an isotropic partner: first-order residual (~ mismatch x alpha'), over
  10 x the calibrated one at w = 12: P 0.7. A Cauchy-Born partner built from the same stencil removes it: P 0.9.
- Quadratic field: even a C1/C2 weight leaves a residual ~ m^2 alpha'' kappa; the linear ramp's does not fall with w: P 0.7.
- Kill (mine): C2 weight at w = 4 horizons leaves an affine residual over 1 % of the bulk nodal force scale. P(met) 0.2.

## Arm C: softening bond law, toughness and strength against the horizon (2-D plane-strain strip, 30 mm wide, 20 mm grips,
notch 10.5 mm, E 5.7 kPa, Gc 7.6 J/m^2, horizon 0.9 and 0.45 mm (3 spacings at h 0.3 and 0.15 mm))
- Toughness (dissipation per crack advance, and the implied G at the peak) within 10 % between the two horizons with the bilinear
  law: P 0.45. Also within 10 % with the critical-stretch law in this notched strip: P 0.45. So the notched strip does not tell the
  two laws apart: P 0.5.
- Strength without a notch: the critical-stretch law's peak changes by about sqrt(2) between the horizons; the bilinear law's within
  10 %: P 0.7.
- An undamaged shared pressure (nu 0.45 by K_p): toughness at least 1.5 x the K_p = 0 value: P 0.6. With the pressure's tension
  part degraded by the cell's bond damage: within 15 % of K_p = 0: P 0.5.
- Kill (mine): the bilinear law's toughness moves over 20 % between the two horizons. P(met) 0.3.

## Arm D: alignment and densification
- Symmetric linear bonds: J > 1 in tension, no densification; order parameter under 0.3 at strain 2.5: P 0.8.
- Tension-only bonds with a compressive fraction 0.01: effective Poisson ratio over 0.5 from strain 0.5, densification >= 2 at
  strain 2.5: P 0.5. Brown's order parameter 0.7 at strain 2.5 needs fibre stiffening as well: P 0.4.
- An incompressible shared pressure gives J = 1 within 1e-3 near the tip, densification 1.0: P 0.95 (the fourfold is unreachable).

## Arm E: the indenter as MIME's RigidInertialNode, the zone's response as resistance (stiffness x dt) and load
- Equilibrium depth under a push within 0.5 % of the direct quasi-static solve: P 0.7.

## Controls (each with its detection rate)
- The pair law alone reads Poisson 1/4 (calibrated stencil); the constant-micromodulus stencil must read 0.278 on [100].
- Broken-bond plant (2 % of bonds in the top layers under the punch, 10 seeds): every seed moves the force beyond 10 x the solve's
  resolution, by the amount an effective-medium estimate predicts within a factor 2.
- A linear blend weight leaves the residual the exact discrete 1-D sum gives (-0.0556 x slope at the kinks for m = 3), not the
  Taylor formula's zero.
- Gc x 1.5 must move the implied toughness x 1.5 within 7 %.
