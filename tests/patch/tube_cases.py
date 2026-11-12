"""Gate 2's harness: the patch spike's question 2 rebuilt on MIME's kind (the steady Schwarz loop, test code). A flux-free
periodic straight tube at D/h 16 (the coarse grid, MIME's MAC grid and band, its steady walled solve), the cluster's own
field in a free periodic box of 96 coarse cells about the sphere with Hasimoto's term (the exchange law's Gaussian at
one coarse cell), MIME's 338-point MRS sphere, and the patch (F2, the trilinear read) handing back its reflection; the
coupled resistance by GMRES; against dev1's tube table through MIME (``Drag.MRS(wall_kind="tube")``) at the same,
true position. ``PATCH_PLANT=r2_offset`` moves the sphere 0.37 coarse cells toward the wall (and 0.21 aside) from its
reference's position, R2's error."""

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

from lubrication_cases import RADIUS, drag_node, in_tube, stepper, tube_pose
from patch_cases import mrs_sphere
from time_cases import _gather, _reflect, _scatter, _walled
from mime.nodes.flow.fvm._band import DEFAULT_BAND, DEFAULT_INTERIOR, TubeSolid, band_named, interior_depth, \
    wall_fields
from mime.nodes.flow.fvm._exchange import HASIMOTO
from mime.nodes.flow.fvm._grid import MACGrid
from mime.nodes.flow.fvm._patch import RefinementPatch, patch_stopping, placed, snapped_origin
from mime.nodes.flow.fvm._patch_transfers import patch_boundary
from mime.nodes.flow.fvm._solve import Stopping, stagnation_window

VISCOSITY = 1e-3
#: The free box's side in coarse cells (R2's).
FREE_BOX = 96
#: The coarse box's cells across beyond the tube's diameter (R2's 27 at D/h 16): the solid between the tube and its
#: periodic images thick enough for the stiff interior (3 cells across left the images' bands nearly touching).
MARGIN_CELLS = 11


@dataclass(frozen=True)
class TubeCase:
    """A sphere of ``a_over_r`` of the tube's radius, its centre ``b_over_r`` of it off the axis along -x; the coarse
    grid ``d_over_h`` cells across the tube; the patch ``ratio`` fine cells a coarse one, ``npc`` coarse cells a side."""

    a_over_r: float
    b_over_r: float
    ratio: int = 4
    npc: int = 12
    d_over_h: int = 16
    dtype: str = "float64"

    @property
    def tube(self) -> float:
        return RADIUS / self.a_over_r

    @property
    def coarse(self) -> float:
        return 2.0 * self.tube / self.d_over_h


def offset() -> np.ndarray:
    return np.array([0.37, 0.21, 0.0]) if os.environ.get("PATCH_PLANT") == "r2_offset" else np.zeros(3)


@functools.partial(jax.jit, static_argnums=(0,))
def _free(grid: MACGrid, density: Any) -> Any:
    """The own field in the free box, steady: each component by the Laplacian's transform, its mean mode zero (the box's
    net force balanced by a uniform pressure gradient, Hasimoto's term added apart), then the grid's exact projector,
    which couples the components (``steady_free`` takes one face set's response to its own forcing alone: F3's
    self-mobility, not a field)."""
    h = grid.spacing
    s = [jnp.sin(jnp.pi * jnp.arange(n) / n) ** 2 for n in grid.cells]
    symbol = VISCOSITY * (4.0 / h ** 2) * (s[0][:, None, None] + s[1][None, :, None] + s[2][None, None, :])
    safe = symbol.at[0, 0, 0].set(1.0)
    held = []
    for r in density:
        modes = jnp.fft.fftn(r) / safe
        held.append(jnp.fft.ifftn(modes.at[0, 0, 0].set(0.0)).real.astype(r.dtype))
    return grid.project((held[0], held[1], held[2]))


def table_reference(case: TubeCase) -> dict[str, float]:
    """The tube table's radial (x) and axial (z) resistance at the case's true position, over the free MRS sphere's."""
    built = in_tube(0.25, lubrication=False, tube=case.tube).build()
    node, state, leaves, step = stepper(built)
    gap = case.tube * (1.0 - case.b_over_r) - RADIUS
    found = np.asarray(step(tube_pose(gap, tube=case.tube), leaves)["resistance"])
    free = np.asarray(drag_node(built).free_resistance(drag_node(built).built_theta())) * VISCOSITY
    return {"radial": float(found[0, 0] / free[0, 0]), "axial": float(found[2, 2] / free[2, 2])}


def coupled(case: TubeCase, form: str) -> dict[str, Any]:
    """The coupled radial and axial resistance over the free sphere's: ``form`` "coarse" (R2's law at D/h 16) or
    "patch"."""
    dtype = jnp.dtype(case.dtype)
    h = case.coarse
    cells = (int(math.ceil(2 * case.tube / h)) + MARGIN_CELLS, int(math.ceil(2 * case.tube / h)) + MARGIN_CELLS,
             2 * case.d_over_h)
    grid = MACGrid(cells, h)
    centre_box = np.array([0.5 * cells[0] * h, 0.5 * cells[1] * h, 0.0])
    solid = TubeSolid(tuple(centre_box), (0.0, 0.0, 1.0), case.tube,
                      tuple((0.0, 0.0, s * cells[2] * h) for s in (-1, 0, 1)))
    fields = wall_fields(grid, [solid], band_named(DEFAULT_BAND), interior_depth(DEFAULT_INTERIOR), dtype)
    sigma = tuple(VISCOSITY * k for k in fields.inverse_permeability)
    sphere = mrs_sphere(0.25)
    a = RADIUS
    centre = centre_box + np.array([-case.b_over_r * case.tube, 0.0, 0.5 * cells[2] * h + 0.11 * h]) + offset() * h
    points = sphere.force + centre
    area = sphere.area
    factor = sla.lu_factor(sphere.matrix / VISCOSITY)
    n = points.shape[0]
    stop = Stopping(tolerance=1e-10 if dtype == jnp.float64 else 1e-5, at_floor=dtype == jnp.float32, cap=20000,
                    window=stagnation_window(cells, 20))
    stencils = tuple(grid.exchange_stencil(c, jnp.asarray(points, dtype=dtype), 1.0) for c in range(3))
    free_grid = MACGrid((FREE_BOX,) * 3, h)
    shift = np.round((centre - 0.5 * FREE_BOX * h) / h) * h
    free_stencils = tuple(free_grid.exchange_stencil(c, jnp.asarray(points - shift, dtype=dtype), 1.0)
                          for c in range(3))
    image = HASIMOTO / (6.0 * math.pi * VISCOSITY * FREE_BOX * h)
    zero = grid.zeros(dtype)
    axial_unit = (zero[0], zero[1], jnp.ones_like(zero[2]))
    unit = _walled(grid, stop, VISCOSITY, 0.0, sigma, axial_unit, zero)
    unit_flux = float(jnp.mean(unit[2]))

    def walled(density: Any) -> Any:
        u = _walled(grid, stop, VISCOSITY, 0.0, sigma, density, zero)
        scale = jnp.mean(u[2]) / unit_flux
        return tuple(u[c] - scale * unit[c] for c in range(3))

    if form == "patch":
        hf = h / case.ratio
        side = case.npc * case.ratio
        patch = RefinementPatch((side, side, side), hf, read="point")
        origin = snapped_origin(patch, jnp.asarray(centre))
        placement = placed(patch, [solid], origin, 0, jnp.asarray(points, dtype=dtype),
                           jnp.asarray(sphere.eps_squared, dtype=dtype), VISCOSITY)
        stopping = patch_stopping(patch, dtype)
        pgrid = patch.grid()
    record: dict[str, Any] = {"iterations": []}

    def reflection(forces: np.ndarray) -> np.ndarray:
        held = jnp.asarray(forces, dtype=dtype)
        u = walled(_scatter(grid, stencils, held))
        uf = _free(free_grid, _scatter(free_grid, free_stencils, held))
        total = held.sum(0) * image
        if form == "coarse":
            return np.asarray(_gather(grid, stencils, u) - _gather(free_grid, free_stencils, uf) - total[None, :],
                              dtype=np.float64)
        walled_given, _ = patch_boundary(pgrid, origin, grid, u, jnp.zeros(3, dtype=dtype))
        free_given, _ = patch_boundary(pgrid, origin, free_grid, uf, jnp.asarray(shift, dtype=dtype))
        given = tuple(walled_given[c] - free_given[c] - total[c] for c in range(3))
        solved = _reflect(patch, stopping, placement, given, held, None, None)
        record["iterations"].append(float(solved.report[0]))
        return np.asarray(solved.values, dtype=np.float64)

    free_force = {}
    out: dict[str, Any] = {"form": form}
    for name, axis in (("radial", 0), ("axial", 2)):
        target = np.tile(np.eye(3)[axis] * 1e-3, (n, 1)).reshape(-1)
        free_force[name] = float((sla.lu_solve(factor, target).reshape(n, 3) * area[:, None]).sum(0)[axis])
        products = [0]

        def operator(y: np.ndarray) -> np.ndarray:
            products[0] += 1
            traction = sla.lu_solve(factor, y).reshape(n, 3)
            return y + reflection(traction * area[:, None]).reshape(-1)

        y, info = spla.gmres(spla.LinearOperator((3 * n, 3 * n), matvec=operator, dtype=np.float64), target,
                             rtol=1e-7 if dtype == jnp.float64 else 1e-5, restart=60, maxiter=6)
        force = float((sla.lu_solve(factor, y).reshape(n, 3) * area[:, None]).sum(0)[axis])
        out[name] = force / free_force[name]
        out[f"{name}_products"] = products[0]
        out[f"{name}_gmres_info"] = int(info)
    out["patch_iterations"] = record["iterations"]
    return out
