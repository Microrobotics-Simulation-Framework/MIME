# TF32 matmul-precision audit — float32 physics paths

**Date:** 2026-05-21
**Branch:** `audit/tf32-matmul-precision`
**Trigger:** a GPU matmul-precision bug (TF32) was found silently corrupting
the D3Q19 LBM by up to 100%+ (see
`couette_torque_mass_conservation.md`). TF32 is a process-wide GPU default,
so this audit checks every other float32 matmul-based physics path in MIME.

## The TF32 trap

On Ampere+ NVIDIA GPUs, JAX's default float32 matmul precision is **TF32** —
a ~10-bit mantissa (~3 decimal digits, ~1e-3 relative error). Any float32
matmul dispatched to GPU tensor cores silently runs at TF32 unless
`precision="highest"` is passed.

It is **catastrophic** only when the matmul result is a *near-cancellation* —
a residual much smaller in magnitude than the input terms (relative
condition number ≫ 1). TF32's ~1e-3 relative error on the inputs then
swamps the result. It is **harmless** (~0.1% noise, tolerable) when the
output is the same order of magnitude as the inputs.

Matmuls with all dimensions ≲ 16–32 (3×3 rotations, 4×4 transforms, 6×6
resistance matrices, 3-vector dots) are **never dispatched to tensor cores**
— XLA does them inline in full float32 — so TF32 cannot reach them.

The audit was run in genuine float32 (`jax_enable_x64=False`); each suspect
path was run at default precision vs `precision="highest"` and compared.

## Verdicts

| Path | Verdict | Evidence |
|---|---|---|
| **FVM dense-DCT pressure / Helmholtz** (`fvm/pressure.py`) | **AFFECTED — FIXED** | 0.45% error vs `highest` |
| **Twin-LBM step** (`lbm/pallas_lbm.py`) | **AFFECTED — FIXED** | identical `f@e` moment bug as the D3Q19 core |
| Stokeslet free-space BEM resistance | not sensitive | bit-identical (no tensor-core matmul) |
| Stokeslet nearest-neighbour resistance | not sensitive | ≤1e-4 rel. (well-conditioned `K@P`) |
| Stokeslet cylinder-Green's-function resistance | not sensitive | bit-identical |
| Kinematics / robot / actuation / FVM operators | not sensitive | all small or contract over dim ≤3 |

## Affected paths (fixed)

### 1. FVM dense-DCT pressure / Helmholtz solver — `fvm/pressure.py`

`make_pressure_solver` / `make_helmholtz_solver` apply the DCT/DST/real-DFT
as **dense matmuls** (`jnp.tensordot`, in `_apply_dct_along_axis`) — the
`transform_backend="auto"` default for meshes ≲256³, i.e. essentially all
production runs (`FVMFluidNode` builds a float32 mesh).

The transform matrices are orthonormal (condition number 1), so this is not
the catastrophic class — but the solver divides by the Laplacian
eigenvalues `1/λ` (dynamic range ~N²), which mildly amplifies TF32 noise
across the 6 chained forward+inverse transforms.

**Test** (`make_pressure_solver`, smooth zero-mean rhs, genuine float32):

| grid | `‖p_default − p_highest‖ / ‖p‖` |
|---|---|
| 32³ neumann | 4.6e-3 |
| 64³ neumann | 4.5e-3 |
| 128³ neumann | 4.2e-3 |
| 64³ periodic | 3.8e-3 |

A direct spectral linear solver returning a **~0.45%** error is a genuine
degradation. The dense transform is not the PISO bottleneck (the report's
own docstring notes this), so `precision="highest"` costs essentially
nothing.

**Fix:** `precision="highest"` on the `jnp.tensordot` in
`_apply_dct_along_axis` — the single chokepoint feeding both solvers.

### 2. Twin-LBM step — `lbm/pallas_lbm.py`

Despite the filename this is **plain JAX**, not a Pallas/Triton kernel. It
is used by `DefectCorrectionFluidNode`'s twin-LBM path and was explicitly
left out of the D3Q19 LBM fix.

`momentum = f @ e` (line 125) is the **identical operation** to the D3Q19
moment bug — an LBM momentum, a ~1e-5 result summed from ~0.05-magnitude
populations. Verified directly on this path: for a flow with `u_x ≈ 3e-5`,
`f @ e` returns **`0.0` at default precision** and the correct
**`2.9996e-5` at `precision="highest"`** — TF32 destroys the moment
entirely. `u @ e.T` (equilibrium) and `force @ e.T` (Guo forcing) are the
same moment matmuls fixed in the D3Q19 core.

**Fix:** `precision="highest"` on all three moment matmuls — the same fix
the D3Q19 core received.

## Checked, not sensitive

### Stokeslet BEM / resistance (`stokeslet/`)

Resistance matrices computed at default vs `highest` precision (genuine
float32, separate processes so jit-caching cannot mask the difference):

| method | `‖R_default − R_highest‖ / ‖R‖` |
|---|---|
| free-space BEM (`compute_resistance_matrix`) | 0 (bit-identical) |
| free-space NN (`compute_nn_resistance_matrix`) | 3e-6 |
| confined NN (`compute_nn_confined_resistance_matrix`) | 1e-4 |
| cylinder-Green's-fn (`compute_gcyl_confined_resistance_matrix`) | 0 (bit-identical) |

Why the Stokeslet is safe:
- `assemble_system_matrix` builds the regularised-Stokeslet matrix with a
  double `vmap` over `stokeslet_tensor` — **element-wise, not a matmul**.
- `solve_bem` / `solve_bem_multi_rhs` use `jax.scipy.linalg`
  `lu_factor`/`lu_solve` — **cuSOLVER**, not tensor-core GEMM.
- The NN method's `A = K @ P` *is* a large matmul, but `P` is a
  nearest-neighbour **selection** matrix, so each `A[i,j]` is a sum of
  similar-sign Stokeslet-kernel values weighted by positive quadrature
  weights — **no near-cancellation**, well-conditioned.
- The cylinder Green's-function correction is assembled in **NumPy on the
  CPU** (no TF32); its only JAX-side matmuls are batched **3×3**
  Green's-tensor × rotation products (too small for tensor cores).
- `bem.py` `K @ u_n` and `flow_field.py` `S @ f` are **3×3** matvecs.

### Everything else

A full triage of every `jnp.matmul` / `jnp.dot` / `jnp.einsum` /
`jnp.tensordot` / `@` site in `control/kinematics/`, `nodes/robot/`,
`nodes/actuation/`, `core/`, and the rest of `nodes/environment/fvm/`
(PISO, operators, IBM, SDF, lifting, simple) found **no dangerous sites**:

- Kinematics / robot dynamics / quaternion / transforms — all **3×3, 4×4,
  6×6, or N×N with N = joint count (≲30)** — never tensor-core dispatched.
- FVM `einsum`s (`"fd,fd->f"`, `"...i,i->..."`, etc.) all contract over the
  **spatial dimension (2 or 3)** — far below the tensor-core threshold; XLA
  lowers them as batched multiply-reduce in full float32, not a GEMM. (The
  Rhie–Chow `grad_p_along` *result* is residual-like, but the einsum that
  produces it contracts over dim ≤3 and the cancellation itself is plain
  float32 subtraction — not a matmul.)

## Out of scope (correctly skipped)

- D3Q19 / D2Q9 LBM core (`d3q19.py`, `d2q9.py`, `bounce_back.py`) — already
  fixed in prior work.
- NN surrogates (`fvm/gnn.py`, `surrogates/cholesky_mlp.py`, the MLP in
  `mlp_resistance_node.py`) — neural-net inference is TF32-tolerant by
  design; forcing full precision would only slow them.

## Fix pattern

Per-call `precision="highest"` on the affected matmul — never a global
`jax_default_matmul_precision` flag (that would needlessly slow the
TF32-tolerant surrogates). Each fixed site carries a comment explaining why
TF32 corrupts that specific matmul.

## Test status

Existing tests stay green. The FVM verification tests run in float64 (where
`precision="highest"` is a no-op — TF32 is float32-only), so they are
unchanged. `pallas_lbm.py` has no numerical baseline test. No re-baselining
was required.
