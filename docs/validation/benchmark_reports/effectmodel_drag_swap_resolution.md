# EffectModel drag-swap — method differences & resolution behaviour

Companion to the EffectModel concept-proof
(`tests/verification/test_effectmodel_stokes_drag_swap.py`). It records how far
apart the `HydrodynamicModel` backends are on the *same* prescribed-motion
sphere, why, and whether the gap closes with resolution.

Scope: the concept-proof runs **Stokeslet** and **FVM** (a free-space
translating sphere). **IBLBM is compile-only here** — it is hardcoded to the
rotating UMR helix and cannot resolve a free-space translating sphere without
new code; its production use (de Boer) runs at 192³.

## Measured drag / analytical Stokes (`6πμaV`)

**Stokeslet (free-space sphere)** — converges monotonically with surface
refinement (continuum-exact BEM):

| surface points | drag / 6πμaV |
|---|---|
| 80   | 0.928 |
| 320  | 0.972 |
| 1280 | 0.988 |

**FVM (sphere in a pipe, confinement `a/R ≈ 0.30`)** — erratic at the coarse
grids runnable in a test, because the sphere spans only 1–2.5 cells:

| grid | sphere radius | \|drag\| / 6πμaV |
|---|---|---|
| 8³   | ~1.0 cell  | 8.83 |
| 12³  | ~1.5 cells | 1.25 |
| 16³  | ~2.0 cells | 1.14 |
| 20³  | ~2.5 cells | 0.71 |

So in the concept-proof: **Stokeslet ≈ 1.00× analytical (exact); FVM ranges
~0.7–9× depending on resolution.** This is why the concept-proof asserts
Stokeslet to ~5% but FVM only to within an order of magnitude — at 8³ the FVM
drag is not a converged quantity.

## Why they differ

- **Stokeslet (BEM)** solves the *unbounded* Stokes equations with the exact
  Green's function on the body surface — continuum-exact; the only error is
  surface-mesh density + the regularization `ε`, both → 0 with refinement.
- **FVM (Navier–Stokes + IBM, Cartesian grid)** has several errors, dominant
  at coarse resolution:
  1. **Under-resolved sphere** — at 1–2.5 cells/radius the immersed boundary
     is barely represented (the 8.83 at ~1 cell is essentially noise).
  2. **IBM diffuse-interface smearing** (`ibm_eps ~ dx`) mislocates the no-slip
     surface by ~1 cell.
  3. **Finite domain / confinement** — `a/R ≈ 0.30` is a *real physical* drag
     enhancement (Haberman wall correction ≈ 2–2.5×), **not** a discretization
     error; it does not vanish with grid refinement.
  4. **Transient** (finite step count) and **finite Re** (full NS, not pure
     Stokes).
- **IBLBM / LBM** (same family) add: lattice resolution, an O(dx) staircased /
  bounce-back wall position, and weak compressibility (Mach).

The deep point: **the methods solve different boundary-value problems** —
unbounded Stokes (Stokeslet) vs. a walled, finite-Re box (FVM/LBM) — so they
converge to *different* limits in general.

## Does the gap close at higher resolution?

- **Stokeslet: yes, cleanly** — a convergent BEM (1% at 1280 points).
- **FVM / LBM: the coarse-grid scatter disappears, but not the whole gap.**
  With ~8–16+ cells/radius and enough steps the under-resolution + IBM-smearing
  errors collapse and the drag *stabilises* to a well-defined value — but that
  value is the **confined** drag (≈2–2.5× free-space here), not Stokeslet's
  free-space `1.0`. Matching Stokeslet additionally requires a **large domain**
  (`a/R → 0`) and **low Re**. Agreement only holds in the shared regime — which
  is exactly what the de Boer / de Jongh cross-validation establishes at
  production resolution, not what the (coarse, swap-surface) concept-proof
  attempts.

**Implication:** quantitative FVM drag at microrobot scale needs a far
better-resolved / higher-fidelity FVM path than the coarse pilot grid (a
slender helix in a vessel spans an even wider scale range than this sphere) —
the convergence behaviour here is why the swap-out roadmap (E6) flags FVM
fidelity as a dedicated work item.
