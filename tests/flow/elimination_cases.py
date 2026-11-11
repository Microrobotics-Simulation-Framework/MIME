"""Shared set-ups for the one elimination over a flow's exchange partners (F3's second PR): a flow node holding point
bodies (and, open, its openings), its step fed as a body and its other laws would feed it, and the monolithic
reference the return-path gate judges a step against: the whole step, the field's every face, the openings' pressures,
each body's force, couple and twist, as one dense linear system solved by one LU, independent of the unit fields, their
warm starts, the elimination's small system and the conjugate gradient. Each plant is switched by
``MIME_PLANT_<NAME>`` (``point_cases.plant``), each draw seeded by ``DETECTION_SEED``."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.core import run_dtype
from mime.nodes.flow.fvm import ImplicitFlowNode, OpenFlowNode
from mime.nodes.flow.fvm._partners import DELIVERIES, delivered, read, stencils
from mime.nodes.flow.fvm._ports import RowKind

WATER_MU, WATER_RHO = 1e-3, 1e3
#: The monolithic step's matrix is formed this many columns at a time (the step's linearisation applied to unit
#: vectors), so that its memory is the matrix's own and not the batched step's over every column at once.
COLUMNS = 256


@dataclass(frozen=True)
class Body:
    """A point body as the gate feeds its flow: its radius and point (m, in the world), its twist (m/s; rad/s), its
    mass and moment of inertia, its kernel's width (cells) and the loads of its other laws (N; N m)."""

    radius: float
    point: tuple[float, float, float]
    twist: tuple[float, ...]
    mass: float
    moment: float
    width: float = 1.0
    loads: tuple[float, ...] = (0.0,) * 6


def body_inputs(body: Body, b: int, dt: float, dtype: Any) -> dict[str, Any]:
    """What body ``b``'s dynamics and other laws feed the flow: its pose, its own step's row (its stepping mass ``diag(m
    I, J I)``, an isotropic body's, no gyroscopic term, beside its free load ``M V / dt``) and its other loads."""
    mass = np.diag([body.mass] * 3 + [body.moment] * 3)
    twist = np.asarray(body.twist)
    return {f"partner_{b}_pose": jnp.asarray([*body.point, 1.0, 0.0, 0.0, 0.0], dtype=dtype),
            f"partner_{b}_step_row": jnp.asarray(np.concatenate([mass, (mass @ twist / dt)[:, None]], axis=1),
                                                 dtype=dtype),
            f"partner_{b}_loads": jnp.asarray(body.loads, dtype=dtype),
            f"partner_{b}_resistances": jnp.zeros((6, 6), dtype=dtype)}


def _flat(velocity: Sequence[Any]) -> Any:
    return jnp.concatenate([jnp.reshape(c, (-1,)) for c in velocity])


def _unflat(z: Any, shapes: Sequence[tuple[int, ...]]) -> tuple[Any, Any, Any]:
    out = []
    start = 0
    for shape in shapes:
        size = int(np.prod(shape))
        out.append(jnp.reshape(z[start:start + size], shape))
        start += size
    return (out[0], out[1], out[2])


@dataclass(frozen=True)
class Monolithic:
    """The monolithic step: the field at the solution, the openings' pressures (absolute) and fluxes, and each body's
    force and couple, twist and reading, a row a body."""

    velocity: tuple[np.ndarray, np.ndarray, np.ndarray]
    pressures: np.ndarray
    fluxes: np.ndarray
    forces: np.ndarray
    twists: np.ndarray
    readings: np.ndarray


def monolithic_step(node: ImplicitFlowNode, p: Mapping[str, Any], state: Mapping[str, Any],
                    inputs: Mapping[str, Any], dt: float) -> Monolithic:
    """One step of ``node`` (IMEX Euler or its steady limit) from ``state`` as one dense system in float64, the
    unknowns the field's every face, each opening's pressure relative to the reference with the level, and each body's
    force, couple and twist: the field's projected momentum ``P A u + (I - P) u = P (rhs + openings' forces + bodies'
    forces)``, each opening's row, each body's law ``(M0 - M_inf) f + G_F u' = V'`` and own row ``S V' + f = r``.
    Its matrix by the step's own operators applied to unit vectors (its linearisation, :data:`COLUMNS` columns at a
    time), solved by one LU."""
    with jax.enable_x64(True):
        p = {k: (jnp.asarray(v, dtype=jnp.float64) if isinstance(v, (np.ndarray, jax.Array, float)) else v)
             for k, v in p.items()}
        state = {k: jnp.asarray(v, dtype=jnp.float64) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v
                 for k, v in state.items()}
        inputs = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in inputs.items()}
        grid = node.grid(p)
        fields = node.fields(p, grid)
        sigma = node.resistivity(p, fields)
        scheme = node.scheme
        shift = 1.0 / dt if scheme.stores_inertia else 0.0
        rho = jnp.asarray(p["density"])
        before = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
        drive = node.forcing(fields, inputs, scheme, jnp.float64)(0, 0.0)
        convection = grid.convection(before) if node.constants()["convection"] else grid.zeros(jnp.float64)
        rhs = tuple(shift * before[k] + drive[k] + convection[k] for k in range(3))
        operator = node.operator_of(p, grid, sigma)(shift)
        bodies = node.body_partners(p, inputs, dt, jnp.float64) if node.body_count else []
        held = [stencils(grid, body.point, body.width) for body in bodies]
        shapes = [grid.shape(k) for k in range(3)]
        faces = sum(int(np.prod(s)) for s in shapes)
        opened = isinstance(node, OpenFlowNode)
        count = node.opening_count() if opened else 0
        if opened:
            assert isinstance(node, OpenFlowNode)
            rows = node.rows()
            reference = node.reference()
            others = node.others()
            a, blocks = node.law_rows(jnp.asarray(p["opening_parameters"]), jnp.asarray(state["opening_state"]),
                                      (jnp.asarray(state["steps"]) + 1) * dt, dt)
            coupling = node.coupling()
        n = faces + count + 2 * DELIVERIES * len(bodies)

        def residual(z: Any) -> Any:
            u = _unflat(z[:faces], shapes)
            start = faces
            forcing = rhs
            out = []
            if opened:
                relative = jnp.zeros(count, dtype=jnp.float64)
                for k, j in enumerate(others):
                    relative = relative.at[j].set(z[start + k])
                level = z[start + count - 1]
                start += count
                port = grid.port_force(relative, rho)
                forcing = tuple(forcing[k] + port[k] for k in range(3))
            twists, forces = [], []
            for b, body in enumerate(bodies):
                force = z[start + 12 * b:start + 12 * b + 6]
                twist = z[start + 12 * b + 6:start + 12 * b + 12]
                forces.append(force)
                twists.append(twist)
                density = delivered(grid, held[b], force, body.delivery)
                forcing = tuple(forcing[k] + density[k] / rho for k in range(3))
            applied = operator(u)
            projected = grid.project(tuple(applied[k] - forcing[k] for k in range(3)))
            complement = tuple(u[k] - grid.project(u)[k] for k in range(3))
            out.append(_flat(tuple(projected[k] + complement[k] for k in range(3))))
            if opened:
                q = grid.opening_fluxes(u)
                for i in range(count):
                    if rows[i] is RowKind.FLUX:
                        out.append(jnp.reshape(q[i] - a[i], (1,)))
                    else:
                        pushed = sum((blocks[i, j] * q[j] for j in range(count) if coupling[i][j]),
                                     start=jnp.zeros((), dtype=jnp.float64))
                        out.append(jnp.reshape(level + relative[i] - pushed - a[i], (1,)))
            for b, body in enumerate(bodies):
                out.append(body.law * forces[b] + read(grid, held[b], u, body.reading) - twists[b])
                out.append(body.system @ twists[b] + forces[b] - body.right)
            return jnp.concatenate(out)

        zero = jnp.zeros(n, dtype=jnp.float64)
        offset, linear = jax.linearize(residual, zero)
        columns = jax.jit(jax.vmap(linear))
        chunk = min(n, COLUMNS)
        transposed = []
        for start in range(0, n, chunk):
            basis = jnp.zeros((chunk, n), dtype=jnp.float64).at[jnp.arange(chunk), jnp.minimum(
                start + jnp.arange(chunk), n - 1)].set(1.0)
            transposed.append(np.asarray(columns(basis))[:min(chunk, n - start)])
        matrix = np.concatenate(transposed, axis=0).T
        solution = np.linalg.solve(matrix, -np.asarray(offset))
        u = tuple(np.asarray(c) for c in _unflat(jnp.asarray(solution[:faces]), shapes))
        start = faces
        pressures = fluxes = np.zeros(0)
        if opened:
            relative = np.zeros(count)
            for k, j in enumerate(others):
                relative[j] = solution[start + k]
            pressures = relative + solution[start + count - 1]
            fluxes = np.asarray(grid.opening_fluxes(tuple(jnp.asarray(c) for c in u)))
            start += count
        forces = np.stack([solution[start + 12 * b:start + 12 * b + 6] for b in range(len(bodies))]) if bodies \
            else np.zeros((0, 6))
        twists = np.stack([solution[start + 12 * b + 6:start + 12 * b + 12] for b in range(len(bodies))]) if bodies \
            else np.zeros((0, 6))
        readings = np.stack([np.asarray(read(grid, held[b], tuple(jnp.asarray(c) for c in u), body.reading))
                             for b, body in enumerate(bodies)]) if bodies else np.zeros((0, 6))
        return Monolithic((u[0], u[1], u[2]), pressures, fluxes, forces, twists, readings)


def sphere_mass(radius: float, ratio: float) -> tuple[float, float]:
    """A uniform sphere's mass and moment of inertia at ``ratio`` times water's density."""
    mass = ratio * WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    return mass, 0.4 * mass * radius ** 2


__all__ = ["WATER_MU", "WATER_RHO", "Body", "Monolithic", "body_inputs", "monolithic_step", "sphere_mass"]
