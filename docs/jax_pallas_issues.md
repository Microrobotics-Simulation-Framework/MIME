# JAX Pallas GPU Issues (for filing)

Discovered while implementing a D3Q19 Lattice Boltzmann kernel.
All issues tested on RunPod A40 (Ampere, CC 8.6) with CUDA 12.x.

---

## Issue 1: `reduce_sum` with `axis` fails in multi-tile kernel

**Environment:** JAX 0.9.2 (Mosaic GPU backend), A40, CUDA 12.8
**Also reproduced on:** JAX 0.5.3 (Triton backend) — different error, same outcome

**Minimal reproducer:**

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

N = 16; BX = BY = BZ = 8; Q = 32
x = jnp.ones((N, N, N, Q))

def sum_kernel(x_ref, o_ref):
    o_ref[...] = jnp.sum(x_ref[...], axis=-1)

o = pl.pallas_call(
    sum_kernel,
    out_shape=jax.ShapeDtypeStruct((N, N, N), jnp.float32),
    grid=(N // BX, N // BY, N // BZ),
    in_specs=[pl.BlockSpec((BX, BY, BZ, Q), lambda i, j, k: (i*BX, j*BY, k*BZ, 0))],
    out_specs=pl.BlockSpec((BX, BY, BZ), lambda i, j, k: (i*BX, j*BY, k*BZ)),
)(x)
# Expected: all elements = 32.0
# Actual (0.9.2): "No support for axes yet"
# Actual (0.5.3): First tile correct (32.0), subsequent tiles write zeros
```

**Expected:** `o[i,j,k] = 32.0` for all `i,j,k`
**Actual (0.9.2):** `NotImplementedError: No support for axes yet`
**Actual (0.5.3):** `o[0,0,0] = 32.0` but `o[8,0,0] = 0.0` (second tile not written)

**Use case:** D3Q19 Lattice Boltzmann — needs `sum` over Q=19 (padded to 32) distributions per lattice node. This is the density computation `ρ = Σ_q f_q`, one of the most fundamental operations in computational fluid dynamics.

**Workaround:** Manual accumulation loop unrolled at trace time:
```python
rho = f[..., 0]
for q in range(1, 32):
    rho = rho + f[..., q]
```

---

## Issue 2: `slice` primitive not implemented in GPU lowering

**Environment:** JAX 0.5.3 (Triton backend), A40, CUDA 12.8
**Also fails on:** JAX 0.9.2 (Mosaic backend)

**Minimal reproducer:**

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

def slice_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = x[..., 0]  # extract first component

o = pl.pallas_call(
    slice_kernel,
    out_shape=jax.ShapeDtypeStruct((8, 8, 8), jnp.float32),
    grid=(1,),
    in_specs=[pl.BlockSpec((8, 8, 8, 4), lambda i: (0, 0, 0, 0))],
    out_specs=pl.BlockSpec((8, 8, 8), lambda i: (0, 0, 0)),
)(jnp.ones((8, 8, 8, 4)))
```

**Expected:** `o[i,j,k] = 1.0`
**Actual:** `NotImplementedError: Unimplemented primitive in Pallas GPU lowering: slice`

**Use case:** Extracting velocity components (x, y, z) from a force vector `force[..., 0:3]`. Essential for any multi-component physics kernel.

**Workaround:** Dot-product with one-hot mask vector passed as explicit input:
```python
mask_x = jnp.array([1.0, 0.0, 0.0, 0.0])  # passed as kernel input
fx = jnp.sum(force * mask_x, axis=-1)
```

---

## Issue 3: `dot_general` requires all dimensions ≥ 16

**Environment:** JAX 0.5.3 (Triton backend), A40, CUDA 12.8
**Also fails on:** JAX 0.9.2 (Mosaic backend) — `ValueError`

**Minimal reproducer:**

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

def matmul_kernel(f_ref, e_ref, o_ref):
    o_ref[...] = f_ref[...] @ e_ref[...]

o = pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((8, 8, 8, 3), jnp.float32),
    grid=(1,),
    in_specs=[
        pl.BlockSpec((8, 8, 8, 19), lambda i: (0, 0, 0, 0)),
        pl.BlockSpec((19, 3), lambda i: (0, 0)),
    ],
    out_specs=pl.BlockSpec((8, 8, 8, 3), lambda i: (0, 0, 0, 0)),
)(jnp.ones((8, 8, 8, 19)), jnp.ones((19, 3)))
```

**Expected:** `o[i,j,k] = [19, 19, 19]`
**Actual (0.5.3):** `ValueError: all dimensions of b must be >= 16`
**Actual (0.9.2):** `ValueError` (similar constraint)

**Use case:** Computing momentum `p = f @ E` where `f` is (nx,ny,nz,19) distributions and `E` is (19,3) velocity vectors. Standard in lattice Boltzmann, finite element, and particle methods.

**Workaround:** Element-wise multiply + manual accumulation:
```python
px = f[...,0]*E[0,0]; py = f[...,0]*E[0,1]; pz = f[...,0]*E[0,2]
for q in range(1, 19):
    px = px + f[...,q]*E[q,0]
    py = py + f[...,q]*E[q,1]
    pz = pz + f[...,q]*E[q,2]
```

---

## Issue 4: `concatenate` limited to 2-argument, `[..., 1]` shapes only

**Environment:** JAX 0.5.3 (Triton backend), A40, CUDA 12.8
**Also fails on:** JAX 0.9.2 (Mosaic backend) — different error

**Minimal reproducer:**

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

a = jnp.ones((8, 8, 8, 1))

def concat_kernel(a_ref, b_ref, c_ref, o_ref):
    o_ref[...] = jnp.concatenate([a_ref[...], b_ref[...], c_ref[...]], axis=-1)

o = pl.pallas_call(
    concat_kernel,
    out_shape=jax.ShapeDtypeStruct((8, 8, 8, 3), jnp.float32),
    grid=(1,),
    in_specs=[pl.BlockSpec((8,8,8,1), lambda i: (0,0,0,0))] * 3,
    out_specs=pl.BlockSpec((8, 8, 8, 3), lambda i: (0, 0, 0, 0)),
)(a, a, a)
```

**Expected:** `o.shape = (8, 8, 8, 3)`, all ones
**Actual (0.5.3):** `NotImplementedError: Only 2-argument concatenate is supported`
**Actual (0.9.2):** `GMEM strides` alignment error

**Use case:** Assembling velocity output `u = stack([ux, uy, uz])` from per-component scalars. Common in any physics kernel that outputs vector fields.

**Workaround:** Use separate output refs for each component.

---

## Issue 5: Non-power-of-2 array dimensions

**Environment:** JAX 0.5.3 (Triton backend), A40, CUDA 12.8

**Minimal reproducer:**

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

def copy_kernel(f_ref, o_ref):
    o_ref[...] = f_ref[...]

o = pl.pallas_call(
    copy_kernel,
    out_shape=jax.ShapeDtypeStruct((16,16,16,19), jnp.float32),
    grid=(1,),
    in_specs=[pl.BlockSpec((16,16,16,19), lambda i: (0,0,0,0))],
    out_specs=pl.BlockSpec((16,16,16,19), lambda i: (0,0,0,0)),
)(jnp.ones((16,16,16,19)))
```

**Expected:** Identity copy
**Actual:** `ValueError: ...size is a power of 2. Encountered an array of shape (16, 16, 16, 19)`

**Note:** JAX 0.9.2 (Mosaic backend) has a different constraint — shared memory size limit instead of power-of-2.

**Use case:** D3Q19 lattice Boltzmann has Q=19 velocity directions — a prime number. D3Q27 (Q=27) and D2Q9 (Q=9) are also non-power-of-2.

**Workaround:** Pad to next power of 2 (19→32). Wastes 40% of compute/memory.

---

## Summary

These five issues collectively prevent implementing a standard D3Q19 LBM kernel in Pallas GPU. The workarounds (manual loops, padding, separate outputs) add complexity but are functional. The manual loop approach (Issue 1 workaround) needs validation — see separate test.

**Impact:** Lattice Boltzmann is one of the most widely used computational fluid dynamics methods. Pallas GPU support for the operations above would enable high-performance LBM kernels that bypass XLA's autotuning overhead (which causes 60+ min compilation on H100 for standard JAX LBM code).
