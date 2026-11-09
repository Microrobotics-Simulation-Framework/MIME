"""Shared set-ups for the FVM's tests: trigonometric fields on a (2 pi)^3 box, straight periodic tubes and their
nodes, and the analytic oracles."""

from __future__ import annotations

import contextlib
import dataclasses
import weakref
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.flow._waveform import waveform
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


@dataclasses.dataclass(frozen=True)
class Waveform:
    """The drive a test plays the law's part with: ``along (steady + amplitude cos(omega t + phase))``, delivered to
    the node's body-force input at each step's end, as ``BodyForce``'s node does."""

    steady: Any = 0.0
    amplitude: Any = 0.0
    omega: Any = 0.0
    phase: Any = 0.0
    along: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def at(self, time: Any) -> Any:
        value = self.steady + self.amplitude * jnp.cos(self.omega * time + self.phase)
        return jnp.asarray(self.along) * value

    def coefficients(self, dtype: Any) -> Any:
        """The drive as the law delivers it to a scheme whose stages read it at their own instants (RK3)."""
        return waveform(self.along, jnp.asarray(self.steady, dtype=dtype), self.amplitude, self.omega / (2 * np.pi),
                        self.phase)


#: Each node's drive, by the node (a test constructs a node and runs it with its own drive).
DRIVES: "weakref.WeakKeyDictionary[ImplicitFlowNode, Waveform]" = weakref.WeakKeyDictionary()


def driven(node: ImplicitFlowNode, waveform: Waveform) -> ImplicitFlowNode:
    DRIVES[node] = waveform
    return node


def tube_node(dh: int = 16, *, nu: float = 1.0, rho: float = 1.0, g: float = 8.0, dt: float = 0.02,
              offset: tuple[float, float] = (0.0, 0.0), length_cells: int = 4, amplitude: float = 0.0,
              omega: float = 0.0, **settings: Any) -> ImplicitFlowNode:
    """The empty periodic tube of D = 1 along z (S7's tube box: D/h + 3 + 8 cells across), driven by a body force g
    on the fluid (with a harmonic part, ``amplitude cos(omega t)``), in units where D, nu and rho are one unless
    given."""
    layout = periodic_tube_layout(0.5, length_cells / dh, cells_across=dh, offset_cells=offset)
    node = ImplicitFlowNode("flow", dt, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                            spacing=layout.spacing, viscosity=nu, density=rho, tube_centre=layout.centre,
                            tube_axis=layout.axis, tube_radius=layout.radius, **settings)
    return driven(node, Waveform(steady=g, amplitude=amplitude, omega=omega))


def step_of(node: ImplicitFlowNode, p: Any, waveform: Waveform | None = None) -> Any:
    """The node's step with its drive delivered as the law delivers it, ``(state) -> state``: the value at the
    instant a scheme reads it once a step at, and the waveform, which a scheme whose stages read their own instants
    evaluates at each."""
    drive = waveform if waveform is not None else DRIVES.get(node, Waveform())
    dt = node.delta_t

    instant = node.scheme.drive_instants[0]

    def step(s: Any) -> Any:
        dtype = s["velocity_z"].dtype
        time = (jnp.asarray(s["steps"]).astype(dtype) + instant) * dt
        return node.advance(p, s, {"body_force": drive.at(time).astype(dtype),
                                   "waveform": drive.coefficients(dtype)}, dt)

    return step


def run(node: ImplicitFlowNode, steps: int, state: Any = None, record: Any = None) -> Any:
    """``steps`` of the node's step under one compiled scan, from ``state`` or the node's initial state, driven by
    its waveform; with ``record`` (a function of the state), also each step's record, stacked."""
    start = node.initial_state() if state is None else state
    step = step_of(node, node._step_constants(None))  # noqa: SLF001 - the step's own constants

    def body(s: Any, _: Any) -> tuple[Any, Any]:
        ended = step(s)
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
