"""Shared set-ups for the FVM's tests: trigonometric fields on a (2 pi)^3 box, straight periodic tubes and their
nodes, and the analytic oracles."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.flow.fvm import ImplicitFlowNode, MACGrid, periodic_tube_layout


def eps(dtype: str) -> float:
    return float(np.finfo(dtype).eps)


def trig_grid(n: int) -> MACGrid:
    return MACGrid((n, n, n), 2.0 * np.pi / n)


def sampled(grid: MACGrid, functions: Any, dtype: Any) -> tuple[Any, Any, Any]:
    """Each component's function sampled at its own faces."""
    return tuple(functions[c](grid.face_points(c, dtype)) for c in range(3))  # type: ignore[return-value]


def divergence_free(grid: MACGrid, dtype: Any, a: float = 1.0, b: float = 0.5) -> tuple[Any, Any, Any]:
    """S1's two divergence-free trigonometric fields, ``a U1 + b U2`` at t = 0."""

    def u(p: Any) -> Any:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return a * jnp.sin(x) * jnp.cos(y) * jnp.cos(z) + b * jnp.sin(y + z)

    def v(p: Any) -> Any:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return a * jnp.cos(x) * jnp.sin(y) * jnp.cos(z) + b * jnp.sin(z + x)

    def w(p: Any) -> Any:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return -2.0 * a * jnp.cos(x) * jnp.cos(y) * jnp.sin(z) + b * jnp.sin(x + y)

    return sampled(grid, (u, v, w), dtype)


def tube_node(dh: int = 16, *, nu: float = 1.0, rho: float = 1.0, g: float = 8.0, dt: float = 0.02,
              offset: tuple[float, float] = (0.0, 0.0), length_cells: int = 4, **settings: Any) -> ImplicitFlowNode:
    """The empty periodic tube of D = 1 along z (S7's tube box: D/h + 3 + 8 cells across), driven by a body force g
    on the fluid, in units where D, nu and rho are one unless given."""
    layout = periodic_tube_layout(0.5, length_cells / dh, cells_across=dh, offset_cells=offset)
    return ImplicitFlowNode("flow", dt, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                            spacing=layout.spacing, viscosity=nu, density=rho, tube_centre=layout.centre,
                            tube_axis=layout.axis, tube_radius=layout.radius, drive_mean=(0.0, 0.0, g), **settings)


def run(node: ImplicitFlowNode, steps: int, state: Any = None, record: Any = None) -> Any:
    """``steps`` of the node's step under one compiled scan, from ``state`` or the node's initial state; with
    ``record`` (a function of the state), also each step's record, stacked."""
    start = node.initial_state() if state is None else state
    dt = node.delta_t
    p = node._step_constants(None)  # noqa: SLF001 - the step's own constants

    def body(s: Any, _: Any) -> tuple[Any, Any]:
        ended = node.advance(p, s, {}, dt)
        return ended, (record(ended) if record is not None else None)

    ended, records = jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps))(dict(start))
    return dict(ended) if record is None else (dict(ended), records)


@contextlib.contextmanager
def in_dtype(dtype: str) -> Iterator[None]:
    """The run's dtype set for a block and restored after it (Hypothesis tests cannot take a function fixture)."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)
