"""The patch's oracle cases (the patch phase's gates 1 and 8; the window spike's q3 rebuilt on MIME's kind): MIME's MRS
sphere above the plane z = 0, the patch with the plane's exact boundary values, the coupled resistance by GMRES; the
same sphere with the plane's image read at its force points, or over each force point's cell, in place of the patch;
Brenner's exact series. The plants a gate's control switches on are read from ``PATCH_PLANT`` when a case is built, so
that ``.github/probe/detection_rate.py`` can fan a control out over fresh processes."""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from lubrication_cases import RADIUS, VISCOSITY, above_plane, drag_node
from mime.nodes.flow.fvm import _patch
from mime.nodes.flow.fvm._patch import RefinementPatch, patch_reflection, patch_stopping, placed, snapped_origin
from mime.nodes.flow.fvm._patch_transfers import Quadrature
from mime.nodes.flow.fvm._solve import REPORT_SIZE
from mime.nodes.hydro.mrs._plane import plane_image

#: The plane's parameters as MIME's image reads them: the medium above z = 0.
PLANE = {"normal": np.array([0.0, 0.0, 1.0]), "offset": 0.0}
#: The columns a case solves: translation along the wall's normal and along the wall, rotation about the wall's y.
COLUMNS = ("Uz", "Ux", "Oy")
#: The plants: the forces' own Stokeslets left out of the patch (its near-wall reflection then only what its faces
#: carry), the boundary values halved, and the patch's cell read and its transpose scaled by 5 % (consistently, so
#: the transpose identity holds: the review's mutant).
PLANTS = ("no_source", "half_boundary", "cell_scale")
CELL_SCALE = 1.05


def plant() -> str | None:
    held = os.environ.get("PATCH_PLANT") or None
    if held is not None and held not in PLANTS:
        raise ValueError(f"PATCH_PLANT is one of {PLANTS}, not {held!r}")
    return held


class PlaneSolid:
    """The solid below the plane ``z = at`` (m), its signed distance positive in it."""

    region = "wall"

    def __init__(self, at: float) -> None:
        self.at = at

    def signed_distance(self, points: Any) -> Any:
        return self.at - points[..., 2]


@dataclass(frozen=True)
class MRSSphere:
    """MIME's MRS sphere of radius ``RADIUS`` as its drag node builds it, in its own frame: force points, the separated
    quadrature (points, weights, owners), each cell's area, ``eps^2`` per force point, and the free part's matrix at
    unit viscosity."""

    force: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    owners: np.ndarray
    area: np.ndarray
    eps_squared: np.ndarray
    matrix: np.ndarray

    @property
    def count(self) -> int:
        return self.force.shape[0]


@functools.cache
def mrs_sphere(spacing: float) -> MRSSphere:
    """The sphere at ``spacing`` radii, from the plane case's built node (its free part's matrix as held)."""
    node = drag_node(above_plane(spacing, lubrication=False).build())
    theta = node.built_theta()
    held = node.body(theta)
    owners = node.owners
    assert owners is not None
    return MRSSphere(np.asarray(held.force), np.asarray(held.points), np.asarray(held.weights), np.asarray(owners),
                     np.asarray(held.area), np.broadcast_to(np.asarray(held.eps_squared), (held.force.shape[0],)).copy(),
                     np.asarray(node.held_in_body_frame))


def regularised_free(targets: np.ndarray, sources: np.ndarray, eps_squared: np.ndarray, forces: np.ndarray,
                     viscosity: float) -> np.ndarray:
    """The regularised Stokeslets of ``forces`` at ``sources`` (one ``eps^2`` each) at ``targets``, ``(t, 3)``."""
    out = np.zeros((targets.shape[0], 3))
    for start in range(0, targets.shape[0], 2048):
        r = targets[start:start + 2048, None, :] - sources[None, :, :]
        r2 = np.sum(r * r, axis=-1)
        scale = 1.0 / (8.0 * math.pi * viscosity * (r2 + eps_squared[None, :]) ** 1.5)
        along = np.einsum("tsk,sk->ts", r, forces)
        out[start:start + 2048] = (np.einsum("ts,sk->tk", scale * (r2 + 2.0 * eps_squared[None, :]), forces)
                                   + np.einsum("ts,tsk->tk", scale * along, r))
    return out


#: The pairs (targets against sources) one chunk of the image holds: its four terms' blocks fit in about 200 MB.
IMAGE_PAIRS = 1 << 19


@functools.cache
def _image_chunk() -> Callable[..., Any]:
    def chunk(targets: Any, sources: Any, eps: Any, forces: Any, offset: Any, viscosity: Any) -> Any:
        parameters = {"normal": jnp.asarray(PLANE["normal"]), "offset": offset}
        blocks = plane_image(targets, sources, parameters, viscosity, eps).matrix
        return jnp.einsum("tsij,sj->ti", blocks, forces)
    return jax.jit(chunk)


def image_at(targets: np.ndarray, sources: np.ndarray, eps_squared: np.ndarray, forces: np.ndarray,
             wall: float, viscosity: float) -> np.ndarray:
    """MIME's regularised plane image of ``forces`` at ``targets`` (the plane at ``z = wall``), ``(t, 3)``, in chunks of
    targets and sources of at most :data:`IMAGE_PAIRS` pairs (each chunk padded to its full size, so one program)."""
    out = np.zeros((targets.shape[0], 3))
    per_source = min(sources.shape[0], 4096)
    per_target = max(1, IMAGE_PAIRS // per_source)
    chunk = _image_chunk()
    for s0 in range(0, sources.shape[0], per_source):
        held_s = np.zeros((per_source, 3)) + np.array([0.0, 0.0, 1.0])
        held_f = np.zeros((per_source, 3))
        held_e = np.ones(per_source)
        count = min(per_source, sources.shape[0] - s0)
        held_s[:count] = sources[s0:s0 + count]
        held_f[:count] = forces[s0:s0 + count]
        held_e[:count] = eps_squared[s0:s0 + count]
        for t0 in range(0, targets.shape[0], per_target):
            n = min(per_target, targets.shape[0] - t0)
            held_t = np.zeros((per_target, 3)) + np.array([0.0, 0.0, 2.0])
            held_t[:n] = targets[t0:t0 + n]
            out[t0:t0 + n] += np.asarray(chunk(jnp.asarray(held_t), jnp.asarray(held_s), jnp.asarray(held_e),
                                               jnp.asarray(held_f), jnp.asarray(wall), viscosity))[:n]
    return out


@dataclass(frozen=True)
class Case:
    """One pose: the sphere's surface ``gap`` radii above the plane, the fine cell ``radius / cells``, the patch the
    sphere plus ``margin`` fine cells a side, its read, and the sphere's spacing."""

    spacing: float
    cells: float
    gap: float
    margin: int = 8
    read: str = "point"
    tolerance: float = 1e-8
    lateral: tuple[float, float] = (0.37, 0.21)
    dilation: int = 0


def built_patch(case: Case) -> tuple[RefinementPatch, Any, np.ndarray, MRSSphere]:
    """The patch and its placement for ``case``, the sphere's centre (m) and the sphere."""
    sphere = mrs_sphere(case.spacing)
    h = RADIUS / case.cells
    centre = np.array([case.lateral[0] * h, case.lateral[1] * h, RADIUS * (1.0 + case.gap)])
    n = int(math.ceil(2.0 * (case.cells + case.margin)))
    patch = RefinementPatch((n, n, n), h, read=case.read)
    origin = snapped_origin(patch, jnp.asarray(centre))
    quadrature = Quadrature(jnp.asarray(sphere.points + centre), jnp.asarray(sphere.weights),
                            jnp.asarray(sphere.owners, dtype=jnp.int32), jnp.asarray(sphere.area))
    placement = placed(patch, [PlaneSolid(0.0)], origin, case.dilation, jnp.asarray(sphere.force + centre),
                       jnp.asarray(sphere.eps_squared), VISCOSITY, quadrature=quadrature)
    return patch, placement, centre, sphere


def patch_reader(case: Case) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """The patch's reflection at the force points as a function of the cell forces ``(n, 3)``, with the plane's exact
    boundary values (MIME's regularised image above the wall, less the forces' own Stokeslets below), and its record
    (CG iterations a product)."""
    patch, placement, centre, sphere = built_patch(case)
    grid = patch.grid()
    held = plant()
    if held == "cell_scale":
        _scale_cell_read()
    origin = np.asarray(placement.origin) * patch.spacing
    sources = sphere.force + centre
    given_points = [np.asarray(grid.given_points(c, jnp.float64)) + origin for c in range(3)]
    stopping = patch_stopping(patch, jnp.float64, tolerance=case.tolerance)

    def solve(boundary: Any, forces: Any) -> Any:
        if held == "no_source":
            forces = jnp.zeros_like(forces)
        return patch_reflection(patch, placement, boundary, forces, None, stopping, jnp.zeros(REPORT_SIZE))

    solver = jax.jit(solve)
    record: dict[str, Any] = {"iterations": [], "cells": patch.cells, "shell_needed": int(placement.shell_needed)}

    def reflection(forces: np.ndarray) -> np.ndarray:
        given = []
        for c in range(3):
            at = given_points[c]
            above = at[:, 2] > 0.0
            values = np.zeros(at.shape[0])
            values[above] = image_at(at[above], sources, sphere.eps_squared, forces, 0.0, VISCOSITY)[:, c]
            values[~above] = -regularised_free(at[~above], sources, sphere.eps_squared, forces, VISCOSITY)[:, c]
            if held == "half_boundary":
                values = 0.5 * values
            given.append(jnp.asarray(values))
        out = solver((given[0], given[1], given[2]), jnp.asarray(forces))
        record["iterations"].append(float(out.report[0]))
        return np.asarray(out.values)

    return reflection, record


def _scale_cell_read() -> None:
    """The ``cell_scale`` plant, for this process: the patch's cell read and its transpose by :data:`CELL_SCALE`."""
    read, back = _patch.cell_read, _patch.cell_read_transpose
    if getattr(read, "planted", False):
        return

    def scaled_read(*args: Any) -> Any:
        values, outside = read(*args)
        return CELL_SCALE * values, outside

    def scaled_back(*args: Any) -> Any:
        return tuple(CELL_SCALE * part for part in back(*args))

    scaled_read.planted = True  # type: ignore[attr-defined]
    _patch.cell_read, _patch.cell_read_transpose = scaled_read, scaled_back


def image_reader(case: Case, read: str) -> Callable[[np.ndarray], np.ndarray]:
    """The plane's exact regularised image of the cell forces read at the force points (``"point"``), each force
    point's cell average over the quadrature (``"cell"``: the target's), or each source spread over its cell's
    quadrature (``"source"``: MIME's integrated image, the target the force point)."""
    sphere = mrs_sphere(case.spacing)
    h = RADIUS / case.cells
    centre = np.array([case.lateral[0] * h, case.lateral[1] * h, RADIUS * (1.0 + case.gap)])
    points = sphere.force + centre
    quadrature = sphere.points + centre
    eps = sphere.eps_squared

    def reflection(forces: np.ndarray) -> np.ndarray:
        if read == "point":
            return image_at(points, points, eps, forces, 0.0, VISCOSITY)
        if read == "cell":
            at = image_at(quadrature, points, eps, forces, 0.0, VISCOSITY) * sphere.weights[:, None]
            sums = np.zeros((sphere.count, 3))
            np.add.at(sums, sphere.owners, at)
            return sums / sphere.area[:, None]
        spread = (forces / sphere.area[:, None])[sphere.owners] * sphere.weights[:, None]
        return image_at(points, quadrature, eps[sphere.owners], spread, 0.0, VISCOSITY)

    return reflection


def column(name: str, points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    if name == "Uz":
        return np.tile([0.0, 0.0, 1.0], (points.shape[0], 1))
    if name == "Ux":
        return np.tile([1.0, 0.0, 0.0], (points.shape[0], 1))
    if name == "Oy":
        return np.cross(np.array([0.0, 1.0, 0.0])[None, :], points - centre)
    raise ValueError(name)


def resistance(case: Case, reflection: Callable[[np.ndarray], np.ndarray] | None, columns: tuple[str, ...] = COLUMNS,
               rtol: float = 1e-9) -> dict[str, dict[str, Any]]:
    """The coupled resistance's columns: GMRES on ``(A / mu) t + W(area t) = V`` right-preconditioned by ``A``'s LU, the
    force and torque about the centre, each column's GMRES products."""
    sphere = mrs_sphere(case.spacing)
    h = RADIUS / case.cells
    centre = np.array([case.lateral[0] * h, case.lateral[1] * h, RADIUS * (1.0 + case.gap)])
    points = sphere.force + centre
    factor = sla.lu_factor(sphere.matrix / VISCOSITY)
    n = sphere.count
    out: dict[str, dict[str, Any]] = {}
    for name in columns:
        target = column(name, points, centre).reshape(-1)
        products = [0]

        def operator(y: np.ndarray) -> np.ndarray:
            traction = sla.lu_solve(factor, y).reshape(n, 3)
            if reflection is None:
                return y
            products[0] += 1
            return y + reflection(traction * sphere.area[:, None]).reshape(-1)

        if reflection is None:
            y = target
        else:
            y, info = spla.gmres(spla.LinearOperator((3 * n, 3 * n), matvec=operator, dtype=np.float64), target,
                                 rtol=rtol, restart=60, maxiter=8)
            residual = float(np.linalg.norm(operator(y) - target) / np.linalg.norm(target))
        forces = sla.lu_solve(factor, y).reshape(n, 3) * sphere.area[:, None]
        out[name] = {"force": forces.sum(0).tolist(), "torque": np.cross(points - centre, forces).sum(0).tolist(),
                     "products": products[0]}
        if reflection is not None:
            # GMRES's own stop is recorded with the true residual, never asserted: a near-wall system the point read
            # makes ill-conditioned stops short of 1e-9 and is read with its residual beside it.
            out[name].update(gmres_info=int(info), true_residual=residual)
    return out


def normal_k(columns: dict[str, dict[str, Any]]) -> float:
    """The normal translation's force over Stokes's ``6 pi mu a``."""
    return columns["Uz"]["force"][2] / (6.0 * math.pi * VISCOSITY * RADIUS)
