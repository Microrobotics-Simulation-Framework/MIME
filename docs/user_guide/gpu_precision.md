# GPU precision and TF32

This is an awareness note. If you only run MIME's built-in experiments,
**no action is needed** — the relevant internal paths already force full
precision. Read on if you write your own GPU code.

## What TF32 is

On Ampere and newer NVIDIA GPUs, JAX's default float32 matmul precision is
**TF32** — a reduced format with a ~10-bit mantissa (~3 decimal digits,
~1e-3 relative error). Any float32 matmul dispatched to GPU tensor cores
silently runs at TF32 unless you ask for full precision explicitly.

## Why it can corrupt physics

TF32 is harmless when a matmul's output is the same order of magnitude as
its inputs — it just adds ~0.1% noise. It is **catastrophic** when the
result is a near-cancellation: a residual far smaller than the input terms
(an LBM momentum moment summed from much larger populations, a spectral
pressure residual). There, TF32's ~1e-3 input error swamps the answer
entirely — a moment that should be `3e-5` can come back as exactly `0.0`.

## What MIME already does

MIME's v0.2 fit-up forced full precision on the affected internal paths —
the LBM moment transforms (D3Q19 / D2Q9 moment matrices) and the FVM
pressure solver. These fixes are per-call `precision="highest"` on the
specific matmuls; TF32-tolerant paths (neural-net surrogates) are left
fast. Nothing is required of you when using these solvers.

## If you write your own GPU code

For precision-sensitive operations — anything where the result is much
smaller than its inputs — request full precision explicitly:

```python
import jax.numpy as jnp

y = jnp.matmul(a, b, precision="highest")   # also jnp.dot / jnp.einsum / jnp.tensordot
```

Prefer per-call `precision=` over the global
`jax_default_matmul_precision` flag, so TF32-tolerant code stays fast.
Small matmuls (all dimensions ≲ 16–32 — 3×3 rotations, 4×4 transforms)
are never dispatched to tensor cores, so TF32 cannot reach them.

## Further reading

The full investigation — every audited float32 matmul path in MIME, with
verdicts and measured errors — is in
[`tf32_matmul_precision_audit.md`](../validation/benchmark_reports/tf32_matmul_precision_audit.md).
