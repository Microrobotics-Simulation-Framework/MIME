"""Test-suite conftest.

Sets JAX env vars *before* any test module imports JAX. Three knobs
matter on this rig:

1. ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` and
   ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`` — without these, cuSolver's
   handle-creation can fail with ``INTERNAL: gpusolverDnCreate(&handle)
   failed`` because JAX's default preallocation grabs the whole device
   and leaves nothing for cuSolver to claim. ``RobotArmNode.update``
   calls ``jnp.linalg.solve(M, rhs)`` (cuSolver-backed on GPU), so this
   matters.

2. ``XLA_FLAGS=--xla_gpu_autotune_level=0`` — XLA picks the "best"
   gemm/solver kernel per shape on first call. For our tiny matrices
   (≤ 6×6 mass matrix) the autotune cost dominates the runtime cost
   and the chosen kernel is in the noise. Skipping autotune saves
   tens of seconds per first-call compile on a 2060.

3. ``jax_compilation_cache_dir`` (a JAX config setting, not an env
   var) — persistent on-disk cache for compiled XLA executables.
   First run of a test still pays the full compile, subsequent runs
   load the cached artefact in milliseconds. Cache lives at
   ``~/.cache/jax_compilation_cache`` by default; override with the
   ``JAX_COMPILATION_CACHE_DIR`` env var.

All knobs are no-ops on CPU. Override locally with
``JAX_PLATFORMS=cpu`` if a particular rig has unrelated GPU issues —
the production code is backend-agnostic.
"""

import os
from pathlib import Path

# Env vars: must be set BEFORE any JAX import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

# Persistent compile cache.
_cache = Path(
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path.home() / ".cache" / "jax_compilation_cache"),
    )
)
_cache.mkdir(parents=True, exist_ok=True)

# Importing jax here is cheap and idempotent; doing so lets us set the
# config in one place rather than relying on every test module to
# remember.
import jax

jax.config.update("jax_compilation_cache_dir", str(_cache))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
