# Defect Correction Method for Confined Microrobot Drag

A domain-decomposition method that couples a regularised Stokeslet BEM (near-field body drag) with an LBM (far-field vessel wall interaction) via immersed boundary force spreading and analytical free-space subtraction.

## 1. Algorithm

### Pipeline

```
For each unit motion (3 translations + 3 rotations → 6×6 resistance matrix):

1. BEM body-only solve:
   A·f₀ = u_body                  → free-space traction f₀
   F₀, T₀ = ∫ f₀ dA              → free-space drag (column of R_free)

2. Iterative defect correction (until convergence):
   a. IB force spreading:
      F_k = f_n · w_k             → point forces at BEM surface nodes
      g(x) = Σ_k F_k · δ_h(x-X_k)  → Eulerian force field (Peskin 4-pt delta)

   b. Walled LBM:
      Run LBM with Guo forcing g(x), pipe bounce-back, open axial BCs
      Warm-start from previous state (200 steps sufficient after initial 500)
      → velocity field u_walled(x)

   c. Multi-radius wall correction:
      For R_eval in [1.25a, 1.3a, 1.5a, 1.7a, 2.0a, 2.5a, 3.0a]:
        Sample u_walled at eval sphere via IB interpolation
        Compute u_freespace = BEM Stokeslet integral (analytical, exact)
        Δu(R_eval) = mean_over_sphere(u_walled - u_freespace)

   d. Polynomial extrapolation:
      Fit Δu(R) = c₀ + c₁(a/R) + c₂(a/R)² + c₃(a/R)³
      Extrapolate: Δu_body = Δu(R=a) = c₀ + c₁ + c₂ + c₃

   e. BEM re-solve with wall correction:
      A·f_{n+1,raw} = u_body - Δu_body
      f_{n+1} = (1-α)·f_n + α·f_{n+1,raw}    (under-relaxation)

   f. Convergence check:
      If |drag_{n+1} - drag_n| / |drag_n| < tol, stop.

3. Final drag = ∫ f_converged dA   (column of R_confined)
```

### Unit Conversions

```
BEM traction f:     Pa (N/m²) — physical
BEM weights w:      m² — physical
Point force F=f·w:  N — physical
LBM force:          F_lu = F_phys × dt²/(ρ_phys × dx⁴)
LBM velocity:       u_phys = u_lu × dx/dt
Eval sphere vel:    u_walled sampled in lattice units, converted to physical for Δu
```

## 2. Why Each Component Is Needed

### Failure modes of simpler approaches

| Approach | Failure mode | Root cause |
|---|---|---|
| Direct Schwarz (velocity at interface sphere) | Double-counts body flow | LBM velocity at interface includes body-induced component that BEM can't separate from wall correction |
| Interface-as-BEM-surface (Dirichlet) | Velocity tautology | SLP velocity at boundary = prescribed BC; no new information transferred |
| Body-only BEM + velocity eval at interface | Wall signal buried | 0.2% wall signal in 99.8% body-dominated velocity field |
| IB force spreading (single pass, compare at body surface) | IB smoothing mismatch | Peskin delta smooths body force over 2h; LBM velocity at body surface ≠ BEM Stokeslet velocity |
| IB force spreading (iterative, compare at body surface) | Converges to wrong value | IB-smoothed LBM velocity at surface systematically overestimates → BEM subtracts too much → drag ≈ 56% of correct |
| Twin-LBM subtraction | Domain-size issue for translation | Unwalled LBM periodic images contribute O(a/L) to Stokeslet (1/r decay); requires impractically large unwalled domain |
| Single-LBM + BEM eval (non-iterative, single radius) | Captures ~67% of wall effect | Linear extrapolation from R_eval=2a to body surface misses image stresslet/source-dipole terms |
| Single-LBM + BEM eval (non-iterative, multi-radius polynomial) | Captures ~67% of wall effect | Free-space traction f₀ underestimates confined traction; wall response is proportionally too weak |

### How defect correction sidesteps each

1. **No velocity transfer at body surface** — the IB smoothing is irrelevant because we compare LBM and BEM at eval spheres (R > 1.25a) where the smoothing has decayed to zero.

2. **BEM free-space reference is exact** — `evaluate_velocity_field()` computes the Stokeslet integral analytically. No domain truncation, no periodic images, no second LBM run.

3. **Polynomial in a/R captures image singularity structure** — the wall correction involves image Stokeslet (~a/R), stresslet (~(a/R)²), source dipole (~(a/R)³). The cubic polynomial fit matches these terms.

4. **Iteration resolves traction↔wall coupling** — the free-space traction underestimates the confined forcing. Each iteration uses the updated traction, producing stronger LBM forcing and a proportionally larger wall correction. Convergence is geometric in Stokes flow (linear operator).

5. **Under-relaxation controls stability** — at high confinement (κ > 0.2), the wall effect > 100% and the unrelaxed iteration overshoots. α = 0.3 gives stable monotonic convergence; α = 0.5-0.7 are faster but oscillate past the fixed point.

## 3. Validation Results

### VER-030: Sphere in Cylinder, λ = 0.3

**Setup:** a=1, R_cyl=3.33, 48³ LBM, body n_refine=2 (320 BEM pts), ε=0.1

**NN-BEM reference:** T_zz=25.17 (1.002× free), F_zz=44.39 (2.355× free)

#### Rotation (1 pass, no iteration needed)

| Method | T_z | Multiplier | Error vs NN-BEM |
|---|---|---|---|
| Free-space BEM | 25.48 | 1.014× | 1.2% |
| Defect correction (1 pass) | 25.48 | 1.000× | 1.2% |
| NN-BEM direct | 25.17 | 1.002× | — |

Wall effect is 0.2% — one pass captures it fully. Polynomial extrapolation adds negligible correction.

#### Translation (iterative, α=0.3)

| Iteration | F_z | Multiplier | Error vs NN-BEM |
|---|---|---|---|
| 0 (free) | 18.91 | 1.003× | — |
| 1 | 22.76 | 1.207× | 48.7% |
| 2 | 26.24 | 1.392× | 40.9% |
| 5 | 34.71 | 1.841× | 21.8% |
| 8 | 40.76 | 2.162× | 8.2% |
| 9 | 42.35 | 2.247× | 4.6% |
| 10 | 43.76 | 2.321× | **1.4%** |

Monotonic convergence to 2.32× (target 2.36×). Geometric convergence rate ≈ 0.7 per iteration.

#### Relaxation parameter comparison (translation, κ=0.3)

| α | Best error | At iteration | Stable? | Estimated iters to 1% |
|---|---|---|---|---|
| 0.3 | 1.4% | 10 | Yes (monotonic) | 11 |
| 0.5 | 0.7% | 6 | No (overshoots at iter 7) | 6 (if stopped) |
| 0.7 | 0.4% | 4 | No (overshoots at iter 5) | 4 (if stopped) |
| 1.0 | 5.3% | 3 | No (diverges at iter 4) | N/A |

## 4. Performance Analysis

### Cost per column of R matrix

| Component | 48³ (RTX 2060) | 64³ (H100) | 128³ (H100) |
|---|---|---|---|
| BEM body-only solve (LU backsubst) | 0.001s | 0.001s | 0.001s |
| IB spreading (320 pts) | 0.001s | 0.001s | 0.001s |
| LBM initial spin-up (500 steps) | 3s | 0.3s | 2s |
| LBM warm-start per iter (200 steps) | 1.2s | 0.1s | 0.8s |
| BEM eval at 7 spheres (320×7 pts) | 0.1s | 0.1s | 0.1s |
| Polynomial fit | 0.001s | 0.001s | 0.001s |
| **Total per R column (rotation, 1 iter)** | **~4s** | **~0.5s** | **~3s** |
| **Total per R column (translation, 10 iter)** | **~15s** | **~1.5s** | **~11s** |
| **Full 6×6 R matrix** | **~60s** | **~6s** | **~40s** |

### Comparison with NN-BEM direct

| Method | Wall mesh required | Cost (6×6 R) | Geometry limitation |
|---|---|---|---|
| NN-BEM direct | Yes (N_wall mesh) | 0.05s (cylinder) | Smooth walls only (BEM mesh quality) |
| Defect correction | No | 6s (H100, 64³) | Any geometry (LBM bounce-back) |

NN-BEM is 100× faster for smooth cylinders. Defect correction's value is geometry-agnostic wall handling — anatomical vessels, irregular geometries, or time-varying wall shapes where BEM meshing is impractical.

**Crossover point:** At N_wall > 50,000 (anatomical geometry), the NN-BEM system matrix (N_body + N_wall)² exceeds 10GB and the LU factorization dominates. The defect correction's cost is independent of wall complexity — the LBM handles arbitrary walls via bounce-back with no increase in matrix size.

## 5. Key Properties

1. **BEM system is always body-only** — O(N_body³) factorization, independent of wall complexity. For UMR (N_body ≈ 2600), the BEM matrix is 7800×7800 — trivial.

2. **LBM handles arbitrary wall geometry** — no wall meshing. Anatomical vessel walls from CT/MRI → voxelized → bounce-back mask. No BEM surface mesh quality concerns.

3. **IB smoothing cancels** — the LBM-minus-BEM subtraction at eval spheres (R > 1.25a) eliminates IB artifacts because both the Peskin-smoothed Stokeslet (LBM) and the analytical Stokeslet (BEM) match at distances >> smoothing width.

4. **Polynomial extrapolation recovers image singularity structure** — cubic in a/R captures Stokeslet, stresslet, and source dipole image terms.

5. **Iteration resolves traction↔wall coupling** — the free-space traction underestimates the confined forcing; iteration amplifies the wall response to self-consistency.

6. **Warm-starting makes dynamic updates cheap** — when the body moves between timesteps, the LBM state from the previous step provides a good initial condition. Only 100-200 steps needed per defect correction iteration.

7. **3D flow field is a byproduct** — the walled LBM velocity field IS the visualization flow field. No separate one-way coupling needed.

## 6. Analysis

### 6.1 Convergence Rate vs Confinement

The wall effect magnitude (drag multiplier - 1) determines the spectral radius of the unrelaxed iteration. Defining W = (R_confined - R_free) / R_free:

| κ | W (axial translation) | Unrelaxed stable? | Safe α (est.) |
|---|---|---|---|
| 0.15 | ~0.5 | Yes | 0.5 |
| 0.22 | ~1.0 | Marginal | 0.4 |
| 0.30 | ~1.35 | No | 0.3 |
| 0.35 | ~2.0 | No | 0.2 |
| 0.40 | ~3.0 | No | 0.2 |

Heuristic: α ≈ 0.8/(1 + W) gives a safe default. For κ=0.3: α ≈ 0.8/2.35 = 0.34, consistent with α=0.3 being robustly stable and α=0.5 overshooting at iteration 7 in the proof of concept. The naive 1/(1+W) ≈ 0.43 is slightly optimistic — validated data shows the overshoot boundary is near α=0.45 at this κ.

For rotation, W < 0.05 at all κ — one pass with α=1 suffices.

### 6.2 Resolution Dependence

The innermost clean eval sphere radius scales as R_min = a + 2h, where h is the lattice spacing and 2h is the Peskin delta support. In units of body radius:

| Resolution | Body radius (lu) | R_min/a | a/R_min | Extrapolation distance |
|---|---|---|---|---|
| 48³ | ~6 | 1.33 | 0.75 | 25% |
| 64³ | ~8 | 1.25 | 0.80 | 20% |
| 128³ | ~16 | 1.125 | 0.89 | 11% |

At 128³, the cubic polynomial extrapolation from a/R=0.89 to a/R=1.0 is only 11% — very stable. The translation result should improve to < 1% error.

### 6.3 Rotation vs Translation Cost

The 6 columns of the resistance matrix are independent in Stokes flow — each unit motion can be computed separately. The 3 rotational columns need 1 iteration each (wall effect < 5%). The 3 translational columns need ~10 iterations each at κ=0.3 with α=0.3.

The LBM force field changes with each unit motion, so the LBM cannot be shared across columns. However, the pipe bounce-back mask and open BCs are identical — only the IB force field changes. The LBM JIT compilation is shared across all 6 columns.

**Cost breakdown for 6×6 R matrix (H100, 64³):**
- 3 rotational columns: 3 × 0.5s = 1.5s
- 3 translational columns: 3 × 1.5s = 4.5s
- Total: ~6s

### 6.4 IQN-ILS Acceleration

The unrelaxed iteration data (α=1.0) shows:
- Iter 1: 31.74, Iter 2: 40.73, Iter 3: 46.74 → brackets the answer (44.39) between iters 2-3.

IQN-ILS with 2 Anderson vectors would detect the overshoot after iter 3 and compute a secant-based correction that lands near 44.4 on iter 4. Estimated convergence: **3-4 iterations** vs 10 iterations with fixed α=0.3.

MADDENING's CouplingGroup already implements IQN-ILS. The defect correction iteration maps directly: the BEM is the "solver" and the wall correction Δu is the "residual." The CouplingGroup handles relaxation, convergence checking, and acceleration automatically.

### 6.5 Where NN-BEM Direct Is Better

NN-BEM direct is the right choice when:
- Wall geometry is smooth and easily meshed (cylinders, tubes)
- Wall geometry is static (mesh once, reuse)
- Speed matters (0.05s vs 6s for the full R matrix)
- The body is small relative to the vessel (κ < 0.4)

Defect correction is the right choice when:
- Wall geometry is complex (anatomical vessels from imaging)
- Wall meshing is impractical (branches, stenoses, patient-specific geometry)
- The wall shape changes over time (pulsatile vessels, peristalsis)
- You also need the 3D flow field for visualization (free byproduct)

For VER-029 (cylinder sweep), both methods work. For the outreach demo (clinical scenario with anatomy), defect correction is required.

## 7. Comparison Table

| Method | VER-025 (BEM only) | VER-030 (Schwarz) | Wall mesh | Anatomy | Cost (6×6 R) | Iteration |
|---|---|---|---|---|---|---|
| NN-BEM direct | <2% | N/A | Yes | Local patch only | 0.05s | No |
| Defect correction | N/A | Rot: 1.2%, Trans: 1.4% | No | Yes | ~6s (H100) | Translation only |
| One-way BEM→LBM | N/A (viz only) | N/A | No | Yes | ~3s (H100) | No |

## 8. References

- Peskin (2002), Acta Numerica 11:479-517 — IB method, Peskin 4-point delta
- Guo, Zheng & Shi (2002), Phys. Rev. E 65:046308 — Guo forcing term for LBM
- Tian et al. (2011), J. Comput. Phys. 230:7266-7283 — IB-LBM coupling
- Elleithy et al. (2001), Eng. Anal. Bound. Elem. 25:685-695 — D-N iteration convergence
- Haberman & Sayre (1958) — sphere-in-cylinder drag correlations
