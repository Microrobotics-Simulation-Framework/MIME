"""Shared set-ups for F5's Schwarz coupling (an MRS body's near field in free space, the walls by the flow's grid): an MRS
sphere's per-body node as the mrs rung builds it in free space, a small walled flow node, the Schwarz step's pieces
(``mime.nodes.flow.fvm._schwarz``) on them, and the monolithic reference gate 3 judges a step against: the whole step,
the field's every face, the body's tractions and its twist, as one dense linear system solved by one LU, independent of
the seven columns, GMRES, its preconditioner, the walled solves inside its products and the composition."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.nodes.flow.fvm import ImplicitFlowNode
from mime.nodes.flow.fvm._grid import Velocity
from mime.nodes.flow.fvm._schwarz import SchwarzBody, Stencils, free_box, image_terms, own_decayed, own_response, \
    stencils
from mime.nodes.flow.fvm._solve import REPORT_SIZE, Stopping as CGStopping, preconditioner_named, projected_solve, \
    stagnation_window
from mime.nodes.flow.fvm._grid import TRUNCATION
from mime.nodes.hydro.mrs._quadrature import turned
from mime.nodes.hydro.near_field import MRSOperator
from mime.nodes.hydro.near_field._contract import Placed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "near_field"))

WATER_MU, WATER_RHO = 1e-3, 1e3
#: The monolithic step's matrix is formed this many columns at a time (each column applies the free box's transforms).
COLUMNS = 24
#: The walled solves' tolerance in the function-level cases: tight, so that the comparison is the scheme's.
TIGHT = 1e-12


def plant() -> str:
    """The Schwarz gates' plant switched on for this run (``MIME_PLANT_SCHWARZ``'s value), or an empty string."""
    return os.environ.get("MIME_PLANT_SCHWARZ", "")


def draw() -> np.random.Generator:
    """The run's draws (``DETECTION_SEED``, zero by default): poses within a cell, twists and histories at physical
    scales."""
    return np.random.default_rng(int(os.environ.get("DETECTION_SEED", "0")))


def drawn_sphere() -> dict[str, float]:
    """``tests/schwarz_experiments.py``'s sphere drawn for the books' gates (the run's draws): its distance from the
    axis, 0 to 0.15 mm (its gap to the wall 1.6 to 2.8 of the grid's cells), and the water's viscosity, log-uniform
    from 0.5 to 4 mPa s."""
    rng = draw()
    return dict(offset=float(rng.uniform(0.0, 0.15)), viscosity=float(np.exp(rng.uniform(np.log(5e-4), np.log(4e-3)))))


def sphere_member(radius: float, spacing: float) -> Any:
    """An MRS sphere's per-body node (``MRSDragNode``) as the mrs rung builds it in free space at ``spacing``."""
    from near_field_cases import bodies  # noqa: PLC0415

    from mime.interactions import Drag
    from mime.scene import Sphere
    from mime.units import m

    experiment = bodies("schwarz_sphere", {"s0": (Sphere(radius=m(radius)), (0.0, 0.0, 0.0))},
                        Drag.MRS(wall_kind="free_space", spacing=m(spacing)))
    return experiment.build().graph.get_node("drag:s0")


def schwarz_body(member: Any, pose: Any, mu: float) -> SchwarzBody:
    """The body as the step holds it: its placement at ``pose``, its free-space near field at ``mu`` and that matrix's
    inverse turned to the pose (the preconditioner)."""
    operator = MRSOperator(member)
    theta = {key: jnp.asarray(value) for key, value in member.built_theta().items()}
    pose = jnp.asarray(pose)
    placed = operator.placed(theta, pose)
    held = member.self_interaction(theta)
    from mime.core import quaternion  # noqa: PLC0415

    matrix = turned(held, quaternion.rotation_matrix(pose[3:])) / mu
    inverse = jnp.linalg.inv(held)
    return SchwarzBody(placed, matrix, operator.preconditioner(inverse, pose, mu),
                       operator.preconditioner(inverse, pose, mu, transpose=True))


def tube_flow(cells: int, length_cells: int, spacing: float, radius: float, dt: float, scheme: str = "imex_euler",
              mu: float = WATER_MU, rho: float = WATER_RHO) -> ImplicitFlowNode:
    """A straight periodic tube along z on a small box, Stokes."""
    centre = (0.5 * cells * spacing, 0.5 * cells * spacing, 0.0)
    return ImplicitFlowNode("flow", dt, cells_x=cells, cells_y=cells, cells_z=length_cells, spacing=spacing,
                            viscosity=mu, density=rho, tube_centre=centre, tube_axis=(0.0, 0.0, 1.0),
                            tube_radius=radius, scheme=scheme, convection=False, tolerance=TIGHT)


@dataclass(frozen=True)
class Pieces:
    """The flow's step operator and its solves at one step, as the Schwarz functions take them."""

    grid: Any
    operator: Any
    project: Any
    walled: Any
    shift: float
    nu: float
    rho: float
    mu: float


def pieces(node: ImplicitFlowNode, dt: float, tolerance: float = TIGHT) -> Pieces:
    p = node.params_pytree()
    grid = node.grid(p)
    fields = node.fields(p, grid)
    sigma = node.resistivity(p, fields)
    shift = 1.0 / dt if node.scheme.stores_inertia else 0.0
    operator = node.operator_of(p, grid, sigma)(shift)
    diagonal = node.diagonal_of(p, grid, sigma, shift)
    magnitude = node.magnitude_of(p, grid, sigma)(shift)
    stopping = CGStopping(tolerance=tolerance, cap=20000, check_every=20, window=stagnation_window(grid.cells, 20))
    tap = jnp.zeros(REPORT_SIZE)

    def walled(rhs: Velocity, guess: Velocity | None = None) -> tuple[Velocity, Any]:
        return projected_solve(grid, operator, diagonal, rhs, guess, stopping, tap, preconditioner_named("jacobi"),
                               magnitude)

    mu, rho = float(p["viscosity"]), float(p["density"])
    return Pieces(grid, operator, grid.project, walled, shift, mu / rho, rho, mu)


def reach_cells(width: float) -> int:
    return math.ceil(TRUNCATION * width)


def box_for(radius: float, spacing: float, width: float = 1.0, side: int | None = None) -> Any:
    """The free box for a sphere of ``radius`` on a grid of ``spacing`` (``side`` cells where given: a small case's
    box, its body and kernel inside it)."""
    from mime.nodes.flow.fvm._exchange import own_box  # noqa: PLC0415

    if side is not None:
        return own_box(side, spacing, jnp.zeros(()).dtype)
    return free_box(2.0 * radius, spacing, reach_cells(width), jnp.zeros(()).dtype)


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
    """The monolithic step: the field at the solution, the body's tractions and its twist."""

    velocity: tuple[np.ndarray, np.ndarray, np.ndarray]
    tractions: np.ndarray
    twist: np.ndarray


def monolithic_step(at: Pieces, box: Any, held: Stencils, body: SchwarzBody, rhs: Velocity, carried: Velocity,
                    image: Any, mass: np.ndarray, right: np.ndarray, origin: Any) -> Monolithic:
    """One step of the flow with the body as one dense system in float64: the field's every face, the body's tractions
    and its twist; the field's projected momentum ``P A u + (I - P) u = P (rhs + S F / rho)``, the near field's
    equations ``K t + G u - G (u_o + O(S F)) - (c_carried + c F) = M V`` and the body's own row ``(M / dt) V + W^T t =
    r``. Its matrix by the step's own operators applied to unit vectors, solved by one LU."""
    del origin
    grid = at.grid
    shapes = [grid.shape(k) for k in range(3)]
    faces = sum(int(np.prod(s)) for s in shapes)
    placed: Placed = body.placed
    count = placed.count
    now, _ = image_terms(box.side, box.spacing, at.shift, at.nu, at.mu)
    from mime.nodes.flow.fvm._exchange import gather, scatter  # noqa: PLC0415
    from mime.nodes.flow.fvm._grid import MACGrid  # noqa: PLC0415

    boxed = MACGrid((box.side,) * 3, box.spacing)
    motions = placed.motions()
    wrench = placed.wrench()
    n = faces + 3 * count + 6

    def residual(z: Any) -> Any:
        u = _unflat(z[:faces], shapes)
        t = z[faces:faces + 3 * count]
        v = z[faces + 3 * count:]
        forces = placed.area[:, None] * t.reshape(count, 3)
        density = tuple(scatter(grid, k, forces[:, k], held.grid[k]) for k in range(3))
        forcing = tuple(rhs[k] + density[k] / at.rho for k in range(3))
        applied = at.operator(u)
        projected = at.project(tuple(applied[k] - forcing[k] for k in range(3)))
        complement = tuple(u[k] - at.project(u)[k] for k in range(3))
        own_force = own_response(box, tuple(scatter(boxed, k, forces[:, k], held.box[k]) for k in range(3)),
                                 at.shift, at.nu, at.rho)
        own = tuple(carried[k] + own_force[k] for k in range(3))
        read = jnp.stack([gather(grid, k, u[k], held.grid[k]) for k in range(3)], axis=1)
        read_own = jnp.stack([gather(boxed, k, own[k], held.box[k]) for k in range(3)], axis=1)
        images = image + now * jnp.sum(forces, axis=0)
        near = body.matrix @ t + (read - read_own - images[None, :]).reshape(-1) - motions @ v
        row = jnp.asarray(mass) @ v + wrench.T @ t - jnp.asarray(right)
        return jnp.concatenate([_flat(tuple(projected[k] + complement[k] for k in range(3))), near, row])

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
    return Monolithic((u[0], u[1], u[2]), solution[faces:faces + 3 * count], solution[faces + 3 * count:])


def open_monolithic_step(node: Any, p: Mapping[str, Any], state: Mapping[str, Any], body: SchwarzBody,
                         mass: np.ndarray, right: np.ndarray, dt: float) -> tuple[Monolithic, np.ndarray]:
    """One step of an open vessel's flow with its Schwarz body as one dense system in float64, from ``state`` (the
    body's carried box field and image history zero, its first step): the field's every face, each opening's pressure
    relative to the reference with the level (F3's ``elimination_cases.monolithic_step``'s rows), the body's tractions
    and its twist (:func:`monolithic_step`'s rows), solved by one LU; with the openings' pressures."""
    from mime.nodes.flow.fvm._exchange import gather, scatter  # noqa: PLC0415
    from mime.nodes.flow.fvm._grid import MACGrid  # noqa: PLC0415
    from mime.nodes.flow.fvm._ports import RowKind  # noqa: PLC0415
    from mime.nodes.flow.fvm._schwarz_host import WIDTH  # noqa: PLC0415

    grid = node.grid(p)
    fields = node.fields(p, grid)
    sigma = node.resistivity(p, fields)
    shift = 1.0 / dt if node.scheme.stores_inertia else 0.0
    rho, mu = float(p["density"]), float(p["viscosity"])
    operator = node.operator_of(p, grid, sigma)(shift)
    before = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
    drive = node.forcing(fields, {}, node.scheme, jnp.float64)(0, 0.0)
    rhs = tuple(shift * before[k] + drive[k] for k in range(3))
    box = node.schwarz_box()
    boxed = MACGrid((box.side,) * 3, box.spacing)
    held = stencils(grid, box, body.placed.force - jnp.asarray(p["schwarz_origin"]), WIDTH)
    now, _ = image_terms(box.side, box.spacing, shift, mu / rho, mu)
    placed: Placed = body.placed
    count = placed.count
    rows, others, coupling = node.rows(), node.others(), node.coupling()
    openings = node.opening_count()
    a, blocks = node.law_rows(jnp.asarray(p["opening_parameters"]), jnp.asarray(state["opening_state"]),
                              (jnp.asarray(state["steps"]) + 1) * dt, dt)
    shapes = [grid.shape(k) for k in range(3)]
    faces = sum(int(np.prod(s)) for s in shapes)
    motions, wrench = placed.motions(), placed.wrench()
    n = faces + openings + 3 * count + 6

    def residual(z: Any) -> Any:
        u = _unflat(z[:faces], shapes)
        relative = jnp.zeros(openings, dtype=jnp.float64)
        for k, j in enumerate(others):
            relative = relative.at[j].set(z[faces + k])
        level = z[faces + openings - 1]
        t = z[faces + openings:faces + openings + 3 * count]
        v = z[faces + openings + 3 * count:]
        port = grid.port_force(relative, rho)
        forces = placed.area[:, None] * t.reshape(count, 3)
        density = tuple(scatter(grid, k, forces[:, k], held.grid[k]) for k in range(3))
        forcing = tuple(rhs[k] + port[k] + density[k] / rho for k in range(3))
        applied = operator(u)
        projected = grid.project(tuple(applied[k] - forcing[k] for k in range(3)))
        complement = tuple(u[k] - grid.project(u)[k] for k in range(3))
        out = [_flat(tuple(projected[k] + complement[k] for k in range(3)))]
        q = grid.opening_fluxes(u)
        for i in range(openings):
            if rows[i] is RowKind.FLUX:
                out.append(jnp.reshape(q[i] - a[i], (1,)))
            else:
                pushed = sum((blocks[i, j] * q[j] for j in range(openings) if coupling[i][j]),
                             start=jnp.zeros((), dtype=jnp.float64))
                out.append(jnp.reshape(level + relative[i] - pushed - a[i], (1,)))
        own = own_response(box, tuple(scatter(boxed, k, forces[:, k], held.box[k]) for k in range(3)), shift,
                           mu / rho, rho)
        read_grid = jnp.stack([gather(grid, k, u[k], held.grid[k]) for k in range(3)], axis=1)
        read_own = jnp.stack([gather(boxed, k, own[k], held.box[k]) for k in range(3)], axis=1)
        images = now * jnp.sum(forces, axis=0)
        out.append(body.matrix @ t + (read_grid - read_own - images[None, :]).reshape(-1) - motions @ v)
        out.append(jnp.asarray(mass) @ v + wrench.T @ t - jnp.asarray(right))
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
    relative = np.zeros(openings)
    for k, j in enumerate(others):
        relative[j] = solution[faces + k]
    pressures = relative + solution[faces + openings - 1]
    tail = faces + openings
    return Monolithic((u[0], u[1], u[2]), solution[tail:tail + 3 * count], solution[tail + 3 * count:]), pressures


def carried_field(box: Any, seed: int, scale: float) -> Velocity:
    """A divergence-free field on the box (a history the box carries), drawn and projected by the box's own response."""
    rng = np.random.default_rng(seed)
    n = box.side
    density = tuple(jnp.asarray(rng.standard_normal((n, n, n)) * scale) for _ in range(3))
    return own_response(box, density, 0.0, 1.0, 1.0)


def decayed(box: Any, field: Velocity, at: Pieces) -> Velocity:
    return own_decayed(box, field, at.shift, at.nu)


__all__ = ["COLUMNS", "TIGHT", "draw", "plant", "WATER_MU", "WATER_RHO", "Monolithic", "Pieces", "box_for",
           "carried_field", "decayed", "drawn_sphere", "monolithic_step", "open_monolithic_step", "pieces",
           "reach_cells", "schwarz_body", "sphere_member", "stencils", "tube_flow"]


def schwarz_flow(member: Any, pose: Sequence[float], cells: int, length_cells: int, spacing: float, tube: float,
                 dt: float, extent: float, scheme: str = "imex_euler", drive: Sequence[float] = (0.0, 0.0, 0.0),
                 mu: float = WATER_MU, rho: float = WATER_RHO, **settings: Any) -> ImplicitFlowNode:
    """A small walled tube's flow holding ``member`` (an MRS per-body node) as its Schwarz partner, the body starting at
    ``pose`` (the grid's frame is the world's), as the FVM's build gives it."""
    from mime.nodes.hydro.mrs._walls import wall_correction_named  # noqa: PLC0415

    constants = dict(member.constants())
    common = {key: constants.pop(key) for key in ("viscosity", "wall_kind", "wall_kind_version")}
    walls = {f"wall_{p}" for p in wall_correction_named(common["wall_kind"], common["wall_kind_version"]).parameters}
    hosted = {f"schwarz_m0_{key}": value for key, value in constants.items() if key not in walls}
    hosted.update({f"schwarz_{key}": constants[key] for key in walls})
    centre = (0.5 * cells * spacing, 0.5 * cells * spacing, 0.0)
    return ImplicitFlowNode("flow", dt, cells_x=cells, cells_y=cells, cells_z=length_cells, spacing=spacing,
                            viscosity=mu, density=rho, tube_centre=centre, tube_axis=(0.0, 0.0, 1.0),
                            tube_radius=tube, scheme=scheme, convection=False, tolerance=TIGHT,
                            schwarz_members=("s0",), schwarz_ids=("drag:s0",), schwarz_origin=(0.0, 0.0, 0.0),
                            schwarz_starts=tuple(float(c) for c in pose), schwarz_drive=tuple(drive),
                            schwarz_extent=float(extent), schwarz_wall_kind=common["wall_kind"],
                            schwarz_wall_kind_version=common["wall_kind_version"], **settings, **hosted)
