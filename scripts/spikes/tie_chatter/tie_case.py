"""The memory estimate's raw program: the allocator scanned over the longest path (OctoMag, 90000 ticks)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

jax: Any = h.jax
jnp: Any = jax.numpy


def factory() -> tuple[Any, tuple[Any, ...]]:
    spec = h.specs()[0]
    dtype = jnp.zeros(()).dtype
    commands = spec.matrix.shape[1]
    signs = jnp.asarray(h.law._sets(commands, spec.matrix.shape[0]).signs)
    matrix, scales = jnp.asarray(spec.matrix, dtype=dtype), jnp.asarray(spec.scales, dtype=dtype)

    def tick(previous: Any, target: Any) -> Any:
        found = h.law.exact_allocation(matrix, target, lower=-h.LIMIT, upper=h.LIMIT, scales=scales, owner="spike")
        face = jnp.argmax(jnp.all(signs == found.active[None, :], axis=1)).astype(jnp.int32)
        return face, (found.commands, face)

    def program(targets: Any) -> Any:
        return h.lax.scan(tick, jnp.int32(-1), targets)[1]

    compiled = jax.jit(program)
    compiled(jnp.asarray(spec.w0[None, :] * jnp.ones((90000, 1)), dtype=dtype))  # run once at full size: the run's resident set is the factory's
    return compiled, (jax.ShapeDtypeStruct((90000, 8), dtype),)
