"""The Schwarz columns' operator transposes as reverse mode needs it (F5; the solve-mask spike's finding on nested
solves): ``<c, J b> = <J^T c, b>`` through the whole matvec, the walled CG inside each product and, in an open vessel,
the openings eliminated in it, with ``J^T`` by ``jax.vjp`` (each inner solve's own transpose, the open grid's weighted
one, #269). The inner solves' stops are formed inside their own solve from their own right-hand side, never closed over
as a constant of the outer matvec, so the transpose is defined at all (a constant that depends on the transposed input
is ``NotImplementedError`` in JAX's transpose of a linear solve).

- **The case**: a drawn ``b`` and ``c`` (tractions at physical scale) on the open vessel's (1 mm long) and the
  periodic box's operator at the run's start, both dtypes: the whole product's identity and the walled read's (its
  walled solve's own, where the CG's stop sits), each on its own inputs' scale, under lines derived from the CG's stop
  and the dtype's rounding (``walled_line``, ``whole_line``).
- **Control** (``MIME_PLANT_SCHWARZ=plain``): the open grid's solve transposed plainly (declared symmetric, its weighted
  transpose dropped). Float64 resolves it by both identities, by six decades on the walled read's. **Float32 does not
  discriminate it**: the walled read's derived line (the CG stop's guaranteed bound, 5.66e-4) sits within a factor of
  the plant's least miss (6.1e-4 seen, 1.08 times the line; up to 2.9e-2 over eight draws), and the whole product's
  (the near field's rounding at the products' length) above the plant's share of that scale on most draws. The float32
  case checks the transpose holds to float32's line; the control is float64's"""

from __future__ import annotations

from typing import Any

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import draw, plant
from schwarz_experiments import schwarz_sphere as experiment

from mime.core import run_dtype
from mime.nodes.flow.fvm._schwarz import INNER_SHARE, SCHWARZ_TOLERANCE, _forces, _gathered, _scattered, product_of
from mime.nodes.flow.fvm._solve import AT_FLOOR_BY_DEFAULT, FLOOR_STOP_TOLERANCES

#: The identities' lines, derived before the run that judges them (the coordinator's point on builder 4's float32 line,
#: which had been set after a run). In the product, the near field's dense product and the free box's spectral solve
#: are exact transposes up to their rounding; the walled CG is not: it stops at a true relative residual ``tau`` of its
#: right-hand side, forward and in its transpose solve. With ``x = A^-1 r`` and ``y = A^-T c'`` (``r`` the scattered
#: forces, ``c'`` the gathered cotangent), the walled read's identity then misses by
#: ``<y, rho_1> - <rho_2, x>``, at most ``tau (|y| |r| + |c'| |x|)``: on that solve's own inputs' scale, ``tau`` each
#: way, times ``sqrt(2)`` where the CG measures its residual in the open box's half-weighted norm. An unflagged CG's
#: ``tau`` is its tolerance (:data:`INNER_SHARE` of the columns'), raised in a dtype that stops at its rounding floor
#: (float32) to at most :data:`FLOOR_STOP_TOLERANCES` tolerances, plus that floor again for the true residual's own
#: rounding: 2e-4 in float32, 1e-10 in float64. The identities are evaluated in float64 (the dot products of the run's
#: vectors), so their evaluation adds nothing at these lines.
def walled_line(dtype: str) -> float:
    """The walled read's identity's line on its own inputs' scale (above)."""
    tau = INNER_SHARE * SCHWARZ_TOLERANCE[dtype]
    if dtype in AT_FLOOR_BY_DEFAULT:
        tau = 2.0 * FLOOR_STOP_TOLERANCES * tau
    return math.sqrt(2.0) * 2.0 * tau


#: The whole product's identity, on its own inputs' scale ``|c||J b| + |J^T c||b|``: the walled read's miss carried at
#: its share of that scale (measured each draw: the ratio of the two scales), and the near field's and the box's
#: rounding in the run's dtype, each at most ``n eps`` of its own scale (``n`` the force points' 3N, the products'
#: length; the box's transforms add ``log2`` of its cells, under ``n`` here).
def whole_line(dtype: str, walled_share: float, size: int) -> float:
    return walled_line(dtype) * walled_share + 2.0 * size * float(np.finfo(dtype).eps)


def _operator(opened: bool) -> tuple[Any, int, Any]:
    """The columns' operator of the sphere's flow at the run's start, on one column, its size and its system."""
    built = experiment(opened=opened, length=LENGTH).build()
    node: Any = built.graph.get_node("water.flow")
    p = {**node.constants(), **node.params_pytree()}
    state = node.initial_state()
    dtype = run_dtype()
    velocity = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
    force = node.schwarz_forcing(node.fields(p, node.grid(p)), {}, jnp.asarray(p["schwarz_drive"], dtype=dtype), dtype)
    at = node.schwarz_system(p, velocity, jnp.asarray(0, dtype=jnp.int32), node.delta_t, force,
                             jnp.asarray(state["schwarz_0_at"], dtype=dtype), state)
    product = product_of(at.grid, at.box, at.held, at.body, at.walled, at.shift, at.nu, at.rho, at.now)
    return (lambda t: product(t[:, None])[0][:, 0]), int(at.body.placed.count) * 3, at


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


def _in64(*vectors: Any) -> list[np.ndarray]:
    return [np.concatenate([np.ravel(np.asarray(c, dtype=np.float64)) for c in v]) if isinstance(v, tuple)
            else np.ravel(np.asarray(v, dtype=np.float64)) for v in vectors]


def _identity(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """``<a, b> - <c, d>`` for ``left = (a, b)``, ``right = (c, d)``, on the inputs' scale ``|a||b| + |c||d|``: the
    miss and the scale."""
    (a, b), (c, d) = left, right
    scale = float(np.linalg.norm(a) * np.linalg.norm(b) + np.linalg.norm(c) * np.linalg.norm(d))
    return abs(float(a @ b) - float(c @ d)) / scale, scale


def _walled(at: Any, b: Any, c: Any) -> tuple[float, float]:
    """The walled read's identity: ``<c', x>`` against ``<y, r>`` (``r`` the forces of ``b`` scattered, ``x`` its
    walled solve, ``c'`` the gathered cotangent ``c``, ``y`` the walled solve's transpose at it, by ``jax.vjp``)."""
    forces = _forces(at.body.placed, b[:, None])[..., 0]
    r = tuple(d / at.rho for d in _scattered(at.grid, at.held.grid, forces))
    x, pullback = jax.vjp(lambda rhs: at.walled(rhs)[0], r)
    _, gathered = jax.vjp(lambda field: _gathered(at.grid, at.held.grid, field), x)
    (c_grid,) = gathered(jnp.reshape(c, (-1, 3)))
    (y,) = pullback(c_grid)
    cg, xx, yy, rr = _in64(c_grid, x, y, r)
    return _identity((cg, xx), (yy, rr))


#: The open vessel's length (mm): short, so that the opening planes, where the plain transpose drops the half weights,
#: hold a share of the field the identity resolves.
LENGTH = 1.0


@pytest.mark.parametrize("opened", [True, False], ids=["open", "periodic"])
def test_the_columns_operator_transposes_through_its_walled_solves(dtype, opened):
    import mime.nodes.flow.fvm._solve as shared

    held = _plain_transpose() if plant() == "plain" and opened else None
    try:
        apply, size, at = _operator(opened)
        rng = draw()
        scale = 1e-3  # Pa: a tenth of a micronewton over a hundred micrometres' cell
        kind = run_dtype()
        b = jnp.asarray(scale * rng.standard_normal(size), dtype=kind)
        c = jnp.asarray(rng.standard_normal(size), dtype=kind)
        jb, pullback = jax.vjp(apply, b)
        (jtc,) = pullback(c)
        walled, walled_scale = _walled(at, b, c)
    finally:
        if held is not None:
            shared.lax = held
    cc, jjb, jjtc, bb = _in64(c, jb, jtc, b)
    whole, whole_scale = _identity((cc, jjb), (jjtc, bb))
    share = walled_scale / whole_scale
    lines = walled_line(dtype), whole_line(dtype, share, size)
    print(f"F5 transpose [{dtype}, {'open' if opened else 'periodic'}]: the walled read's miss {walled:.3g} (line "
          f"{lines[0]:.3g}), the whole product's {whole:.3g} (line {lines[1]:.3g}, the walled read's share of its scale "
          f"{share:.3g})")
    assert np.isfinite(walled) and walled <= lines[0], (walled, lines)
    assert np.isfinite(whole) and whole <= lines[1], (whole, lines)
