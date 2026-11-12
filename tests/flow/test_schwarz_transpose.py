"""The Schwarz columns' operator transposes as reverse mode needs it (F5; the solve-mask spike's finding on nested
solves): ``<c, J b> = <J^T c, b>`` through the whole matvec, the walled CG inside each product and, in an open vessel,
the openings eliminated in it, with ``J^T`` by ``jax.vjp`` (each inner solve's own transpose, the open grid's weighted
one, #269). The inner solves' stops are formed inside their own solve from their own right-hand side, never closed over
as a constant of the outer matvec, so the transpose is defined at all (a constant that depends on the transposed input
is ``NotImplementedError`` in JAX's transpose of a linear solve).

- **The case**: a drawn ``b`` and ``c`` (tractions at physical scale) on the open vessel's and the periodic box's
  operator at the run's start, both dtypes, the identity's miss on the products' scale under the inner solves'
  tolerance.
- **Control** (``MIME_PLANT_SCHWARZ=plain``): the open grid's solve transposed plainly (declared symmetric, its weighted
  transpose dropped), which misses the identity on every draw."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import draw, plant
from schwarz_experiments import schwarz_sphere as experiment

from mime.core import run_dtype
from mime.nodes.flow.fvm._schwarz import product_of

#: The identity's line on the products' scale: the inner solves' tolerance (1e-2 of the columns'), with a margin of a
#: hundred in float64 (1e-10) and none in float32 (1e-6). Float32's was predicted at 1e-3 and set here after the first
#: run, which put the plain transpose's miss at 1.5e-5 inside that and the clean miss at 1.6e-8.
DOT_LINE = {"float64": 1e-8, "float32": 1e-6}


def _operator(opened: bool) -> tuple[Any, int]:
    """The columns' operator of the sphere's flow at the run's start, on one column, and its size."""
    built = experiment(opened=opened).build()
    node: Any = built.graph.get_node("water.flow")
    p = {**node.constants(), **node.params_pytree()}
    state = node.initial_state()
    dtype = run_dtype()
    velocity = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
    force = node.schwarz_forcing(node.fields(p, node.grid(p)), {}, jnp.asarray(p["schwarz_drive"], dtype=dtype), dtype)
    at = node.schwarz_system(p, velocity, jnp.asarray(0, dtype=jnp.int32), node.delta_t, force,
                             jnp.asarray(state["schwarz_0_at"], dtype=dtype), state)
    product = product_of(at.grid, at.box, at.held, at.body, at.walled, at.shift, at.nu, at.rho, at.now)
    return (lambda t: product(t[:, None])[0][:, 0]), int(at.body.placed.count) * 3


class _Plain:
    """The plant's ``lax`` for the shared CG: every linear solve declared symmetric, its own transpose dropped (the open
    grid's weighted one among them), as the plain transpose would take it."""

    def __init__(self, real: Any) -> None:
        self.real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)

    def custom_linear_solve(self, matvec: Any, b: Any, solve: Any, transpose_solve: Any = None,
                            symmetric: bool = False, **settings: Any) -> Any:
        del transpose_solve, symmetric
        return self.real.custom_linear_solve(matvec, b, solve, symmetric=True, **settings)


def _plain_transpose() -> Any:
    """The plant: the shared CG's solves transposed plainly; returns what it replaced."""
    import mime.nodes.flow.fvm._solve as shared

    held = shared.lax
    shared.lax = _Plain(held)
    return held


@pytest.mark.parametrize("opened", [True, False], ids=["open", "periodic"])
def test_the_columns_operator_transposes_through_its_walled_solves(dtype, opened):
    import mime.nodes.flow.fvm._solve as shared

    held = _plain_transpose() if plant() == "plain" and opened else None
    try:
        apply, size = _operator(opened)
        rng = draw()
        scale = 1e-3  # Pa: a tenth of a micronewton over a hundred micrometres' cell
        kind = run_dtype()
        b = jnp.asarray(scale * rng.standard_normal(size), dtype=kind)
        c = jnp.asarray(rng.standard_normal(size), dtype=kind)
        jb, pullback = jax.vjp(apply, b)
        (jtc,) = pullback(c)
    finally:
        if held is not None:
            shared.lax = held
    left, right = float(jnp.vdot(c, jb)), float(jnp.vdot(jtc, b))
    size_of = float(jnp.linalg.norm(c) * jnp.linalg.norm(jb) + jnp.linalg.norm(jtc) * jnp.linalg.norm(b))
    miss = abs(left - right) / size_of
    print(f"F5 transpose [{dtype}, {'open' if opened else 'periodic'}]: <c, J b> {left:.12g}, <J^T c, b> {right:.12g}, "
          f"miss {miss:.3g} of the products' scale")
    assert np.isfinite(miss) and miss <= DOT_LINE[dtype], (left, right, miss)
