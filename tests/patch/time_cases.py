"""The patch in time, in F5's structure (item 3 and gate 10; the signed F5-patch interface's point 5): a sphere held
above a plane wall and given an impulsive velocity, its force stepped by backward Euler under the step operator
``rho/dt - mu L + mu/K``; each step a GMRES over the near field's forces (MIME's MRS free part, right-preconditioned by
its LU), each product a column (the forces' reflection, no history), the base column carrying the history.

**The forms** (each a reflection the near field reads at its force points):

- ``reference``: the whole box at the fine cell, the walled step less the cluster's own box, both by the exchange law's
  Gaussian (the fine monolithic BE reference);
- ``coarse``: the same at the coarse cell;
- ``be_patch``: the coarse walled step and own box handing the patch its given values (the walled less the own),
  the patch under the step operator with its state (the last step's field and the own box's history at its shell);
- ``quasi_steady``: the same given values, the patch's shift zero and no history.

The box is periodic with a solid slab (the plane below the sphere, a far wall above it). The own box is a periodic box
of the same cells, solved by its transform: the step operator is diagonal there. Plants (``PATCH_PLANT``), for a sphere
moved one fine cell a step along the wall: ``slab_unfilled`` leaves the slab a snap uncovers zero; ``history_unmoved``
keeps the last step's field where it was, one cell off the moved patch."""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from lubrication_cases import RADIUS
from patch_cases import mrs_sphere
from mime.nodes.flow.fvm._band import DEFAULT_BAND, DEFAULT_INTERIOR, band_named, interior_depth, wall_fields
from mime.nodes.flow.fvm._exchange import gather, scatter
from mime.nodes.flow.fvm._grid import MACGrid
from mime.nodes.flow.fvm._patch import PatchHistory, RefinementPatch, moved, patch_reflection, patch_stopping, placed, \
    snapped_origin
from mime.nodes.flow.fvm._patch_transfers import patch_boundary, patch_faces, shell_sample
from mime.nodes.flow.fvm._solve import REPORT_SIZE, Jacobi, Stopping, projected_solve, stagnation_window

#: The fluid's density (glycerol-like, kg/m^3) and the stage-1 corners: viscosity (Pa s) and step (s).
DENSITY = 1260.0
CORNERS = {"m50-dt0.5": (0.05, 5e-4), "m50-dt5": (0.05, 5e-3), "m200-dt0.5": (0.2, 5e-4), "m200-dt5": (0.2, 5e-3)}
FORMS = ("reference", "coarse", "be_patch", "quasi_steady")
#: The walled step's CG tolerance and the near field's GMRES tolerance: the outer above the inner solves' (the patch's
#: CG at 1e-8, the walled one here): a GMRES below them cannot converge and runs hundreds of products (the read probe).
WALLED_TOLERANCE = 1e-8
GMRES_TOLERANCE = 1e-7


@dataclass(frozen=True)
class TimeCase:
    """A sphere of ``radius`` (m) with its surface ``gap`` radii above the slab's top; ``cells`` fine cells across its
    radius, the coarse cell ``ratio`` fine ones, the coarse box ``box`` coarse cells a side with a slab ``slab`` thick;
    the patch the sphere plus ``margin`` fine cells a side; the impulse along ``direction`` (0: along the wall, 2:
    its normal); ``steps`` steps; ``moving``: the sphere moved one fine cell a step along x (every step snaps)."""

    viscosity: float
    dt: float
    gap: float
    direction: int = 2
    radius: float = 0.7e-3
    cells: float = 10.0
    ratio: int = 2
    box: int = 24
    slab: int = 4
    margin: int = 8
    steps: int = 40
    moving: bool = False
    warm: bool = True

    @property
    def fine(self) -> float:
        return self.radius / self.cells

    @property
    def coarse(self) -> float:
        return self.fine * self.ratio

    @property
    def lam(self) -> float:
        return self.radius / math.sqrt(self.viscosity / DENSITY * self.dt)


class SlabSolid:
    """The solid slab ``0 <= z <= top`` in a periodic box of height ``length``, its signed distance positive in it."""

    region = "wall"

    def __init__(self, top: float, length: float) -> None:
        self.top, self.length = top, length

    def signed_distance(self, points: Any) -> Any:
        z = jnp.mod(points[..., 2], self.length)
        return jnp.where(z <= self.top, jnp.minimum(z, self.top - z), -jnp.minimum(z - self.top, self.length - z))


class PlaneSolid:
    region = "wall"

    def __init__(self, at: float) -> None:
        self.at = at

    def signed_distance(self, points: Any) -> Any:
        return self.at - points[..., 2]


@functools.partial(jax.jit, static_argnums=(0, 1))
def _walled(grid: MACGrid, stopping: Stopping, mu: Any, shift: Any, sigma: Any, rhs: Any, guess: Any) -> Any:
    """The walled step: ``(shift - mu L + sigma) u + G p = rhs``, divergence-free."""

    def apply(u: Any) -> Any:
        lap = grid.laplacian(u)
        return tuple(shift * u[c] - mu * lap[c] + sigma[c] * u[c] for c in range(3))

    def magnitude(u: Any) -> Any:
        lap = grid.laplacian_magnitude(u)
        return tuple(shift * u[c] + mu * lap[c] + sigma[c] * u[c] for c in range(3))

    base = shift - mu * grid.laplacian_diagonal()
    diagonal = tuple(base + sigma[c] for c in range(3))
    solved, _ = projected_solve(grid, apply, diagonal, rhs, guess, stopping, jnp.zeros(REPORT_SIZE), Jacobi(),
                                magnitude)
    return solved


@functools.partial(jax.jit, static_argnums=(0,))
def _own(grid: MACGrid, symbol: Any, rhs: Any) -> Any:
    """The own box's step by its transform, its mean mode zero: a free fluid carries no mean drift, and the periodic
    box's mean (a net force accelerating the whole box, unresisted) is the box's artefact, which under the step
    operator is Hasimoto's term screened to zero (the F5-patch interface)."""
    held = []
    for r in rhs:
        modes = jnp.fft.fftn(r) / symbol
        held.append(jnp.fft.ifftn(modes.at[0, 0, 0].set(0.0)).real)
    return grid.project((held[0], held[1], held[2]))


@functools.partial(jax.jit, static_argnums=(0,))
def _scatter(grid: MACGrid, stencils: Any, forces: Any) -> Any:
    return tuple(scatter(grid, c, forces[:, c], stencils[c]) for c in range(3))


@functools.partial(jax.jit, static_argnums=(0,))
def _gather(grid: MACGrid, stencils: Any, field: Any) -> Any:
    return jnp.stack([gather(grid, c, field[c], stencils[c]) for c in range(3)], axis=1)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _reflect(patch: RefinementPatch, stopping: Stopping, placement: Any, boundary: Any, forces: Any, start: Any,
             history: Any) -> Any:
    return patch_reflection(patch, placement, boundary, forces, start, stopping, jnp.zeros(REPORT_SIZE), history)


@functools.partial(jax.jit, static_argnums=(0, 2))
def _given(grid: Any, origin: Any, coarse: MACGrid, walled: Any, own: Any) -> Any:
    w, _ = patch_boundary(grid, origin, coarse, walled, jnp.zeros(3))
    o, _ = patch_boundary(grid, origin, coarse, own, jnp.zeros(3))
    return (w[0] - o[0], w[1] - o[1], w[2] - o[2])


class Box:
    """A periodic box of ``cells`` cubic cells of ``spacing`` holding the slab: the walled step's operator and solve, the
    own box's transform solve, the exchange law's stencils at the sphere's points, width one cell."""

    def __init__(self, cells: int, spacing: float, slab: int, viscosity: float, shift: float, points: np.ndarray):
        self.grid = MACGrid((cells, cells, cells), spacing)
        self.mu, self.shift = viscosity, shift
        solid = SlabSolid(slab * spacing, cells * spacing)
        fields = wall_fields(self.grid, [solid], band_named(DEFAULT_BAND), interior_depth(DEFAULT_INTERIOR),
                             jnp.float64)
        self.sigma = tuple(viscosity * k for k in fields.inverse_permeability)
        self.points = jnp.asarray(points)
        self.stencils = [self.grid.exchange_stencil(c, self.points, 1.0) for c in range(3)]
        n = cells
        s = [jnp.sin(jnp.pi * jnp.arange(n) / n) ** 2 for _ in range(3)]
        self.symbol = shift + viscosity * (4.0 / spacing ** 2) * (s[0][:, None, None] + s[1][None, :, None]
                                                                 + s[2][None, None, :])
        self.stopping = Stopping(tolerance=WALLED_TOLERANCE, cap=20000, window=stagnation_window(self.grid.cells, 20))

    def scatter(self, forces: Any) -> Any:
        return _scatter(self.grid, tuple(self.stencils), forces)

    def gather(self, field: Any) -> Any:
        return _gather(self.grid, tuple(self.stencils), field)

    def walled(self, rhs: Any, guess: Any = None) -> Any:
        return _walled(self.grid, self.stopping, self.mu, self.shift, self.sigma, rhs,
                       guess if guess is not None else self.grid.zeros(jnp.float64))

    def own(self, rhs: Any) -> Any:
        """The own box's step, no walls: the transform solve, projected."""
        return _own(self.grid, self.symbol, rhs)


def sphere_points(case: TimeCase, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """MIME's 338-point sphere scaled to ``case.radius`` at ``centre``: its force points, cell areas, ``eps^2`` and the
    free part's LU at the case's viscosity."""
    held = mrs_sphere(0.25)
    scale = case.radius / RADIUS
    return (held.force * scale + centre, held.area * scale ** 2, held.eps_squared * scale ** 2,
            sla.lu_factor(held.matrix * scale / case.viscosity))


def plant() -> str | None:
    return os.environ.get("PATCH_PLANT") or None


def run(case: TimeCase, form: str) -> dict[str, Any]:
    """One form's force history (N, along the impulse) and its GMRES products a step."""
    shift = DENSITY / case.dt
    length = case.box * case.coarse
    top = case.slab * case.coarse
    centre = np.array([0.5 * length + 0.37 * case.fine, 0.5 * length + 0.21 * case.fine,
                       top + case.radius * (1.0 + case.gap)])
    spacing = case.fine if form == "reference" else case.coarse
    cells = case.box * (case.ratio if form == "reference" else 1)
    velocity = np.zeros(3)
    velocity[case.direction] = 1e-3
    points, area, eps2, factor = sphere_points(case, centre)
    box = Box(cells, spacing, case.slab * (case.ratio if form == "reference" else 1), case.viscosity, shift, points)
    walled_old = box.grid.zeros(jnp.float64)
    own_old = box.grid.zeros(jnp.float64)
    patch: RefinementPatch | None = None
    state: dict[str, Any] = {}
    n = points.shape[0]
    forces_out: list[float] = []
    products_out: list[int] = []
    residuals: list[float] = []
    iterations_out: list[float] = []
    base_out: list[float] = []
    if form in ("be_patch", "quasi_steady"):
        size = int(math.ceil(2.0 * (case.cells + case.margin)))
        patch = RefinementPatch((size, size, size), case.fine, read="point")
        state = patch.initial_state(snapped_origin(patch, jnp.asarray(centre)), jnp.float64)
    for step in range(case.steps):
        if case.moving and step > 0:
            centre = centre + np.array([case.fine, 0.0, 0.0])
            points, area, eps2, factor = sphere_points(case, centre)
            box = Box(cells, spacing, case.slab * (case.ratio if form == "reference" else 1), case.viscosity, shift,
                      points)
        history_walled = box.walled(tuple(shift * w for w in walled_old), walled_old)
        history_own = box.own(tuple(shift * w for w in own_old))
        step_iterations = [0.0]
        base_iterations = [0.0]
        if patch is None:
            def column(forces: np.ndarray) -> np.ndarray:
                density = box.scatter(jnp.asarray(forces))
                return np.asarray(box.gather(box.walled(density)) - box.gather(box.own(density)))

            base = np.asarray(box.gather(history_walled) - box.gather(history_own))
            compose = None
        else:
            stepped = form == "be_patch"
            origin = snapped_origin(patch, jnp.asarray(centre))
            placement = placed(patch, [PlaneSolid(top)], origin, 0, jnp.asarray(points), jnp.asarray(eps2),
                               case.viscosity, shift=shift if stepped else 0.0)
            stopping = patch_stopping(patch, jnp.float64)
            grid = patch.grid()
            previous = (box.grid, walled_old, own_old)
            fill = patch_faces(grid, origin, previous[0], tuple(a - b for a, b in zip(walled_old, own_old)),
                               jnp.zeros(3))
            if plant() == "slab_unfilled":
                fill = grid.zeros(jnp.float64)
            if stepped and plant() == "history_unmoved":
                held = (state["patch_field_x"], state["patch_field_y"], state["patch_field_z"])
            else:
                held = patch.held_field(state, origin, fill) if stepped else grid.zeros(jnp.float64)
            start = held if case.warm and stepped else None
            history = PatchHistory(held, shell_sample(placement.shell_points, placement.shell_components, box.grid,
                                                      history_own, jnp.zeros(3))) if stepped else None

            def given(walled: Any, own: Any) -> Any:
                return _given(grid, origin, box.grid, walled, own)

            def column(forces: np.ndarray) -> np.ndarray:
                density = box.scatter(jnp.asarray(forces))
                solved = _reflect(patch, stopping, placement, given(box.walled(density), box.own(density)),
                                  jnp.asarray(forces), start, None)
                step_iterations[0] += float(solved.report[0])
                return np.asarray(solved.values)

            base_solve = _reflect(patch, stopping, placement, given(history_walled, history_own),
                                  jnp.zeros((n, 3)), start, history)
            step_iterations[0] += float(base_solve.report[0])
            base_iterations[0] = float(base_solve.report[0])
            base = np.asarray(base_solve.values)

            def compose(forces: np.ndarray, walled: Any, own: Any) -> Any:
                return _reflect(patch, stopping, placement, given(walled, own), jnp.asarray(forces), start, history)

            if not stepped:
                compose = None

        target = np.tile(velocity, (n, 1)) - base
        count = [0]

        def operator(y: np.ndarray) -> np.ndarray:
            count[0] += 1
            traction = sla.lu_solve(factor, y).reshape(n, 3)
            return y + column(traction * area[:, None]).reshape(-1)

        y, info = spla.gmres(spla.LinearOperator((3 * n, 3 * n), matvec=operator, dtype=np.float64),
                             target.reshape(-1), rtol=GMRES_TOLERANCE, restart=60, maxiter=6)
        residuals.append(float(np.linalg.norm(operator(y) - target.reshape(-1)) / np.linalg.norm(target)))
        count[0] -= 1
        forces = sla.lu_solve(factor, y).reshape(n, 3) * area[:, None]
        density = box.scatter(jnp.asarray(forces))
        walled_new = box.walled(tuple(shift * w + d for w, d in zip(walled_old, density)), walled_old)
        own_new = box.own(tuple(shift * w + d for w, d in zip(own_old, density)))
        if patch is not None and compose is not None:
            solved = compose(forces, walled_new, own_new)
            state = patch.advanced_state(origin, solved.field, solved.report)
        walled_old, own_old = walled_new, own_new
        forces_out.append(float(forces.sum(0)[case.direction]))
        products_out.append(count[0])
        iterations_out.append(step_iterations[0])
        base_out.append(base_iterations[0])
    return {"form": form, "force": forces_out, "products": products_out, "lambda": case.lam,
            "gmres_true_residual": residuals, "patch_iterations": iterations_out,
            "base_iterations": base_out}
