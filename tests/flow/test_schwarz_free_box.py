"""Gate 1: one MRS body coupled through a grid with no wall is the body uncoupled (the free box removes exactly its own
field from what the near field reads).

- **The inertial step** (IMEX Euler, through the flow's own CG on a periodic box whose tube is wider than the box: no
  wall, no solid): the handed 6x6 is the free-space MRS's own resistance, to the images of a periodic box under the
  step operator (screened, their irrotational part about ``6 pi nu dt a / L^3`` of the self term, ``2e-4`` here).
- **The steady limit** (a free periodic box of another side as the "grid", by its transform, the mean removed): the
  handed 6x6 is the free-space MRS's with Hasimoto's term for the grid box's images, ``1/R = 1/R_free - c/(6 pi mu
  L)`` on the translation, to Hasimoto's next term ``4.19 (a/L)^3`` (``1e-3`` here).

Control: the free box's subtraction omitted (the body's own field counted twice, the coupling spike's C1), which moves
the 6x6 by the order of itself."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import box_for, draw, pieces, plant, schwarz_body, sphere_member, tube_flow

from mime.nodes.flow.fvm._exchange import HASIMOTO, own_box
from mime.nodes.flow.fvm._grid import MACGrid
from mime.nodes.flow.fvm._schwarz import image_terms, own_response, prepared, stencils
from mime.nodes.hydro.mrs._solve import force_sums
from mime.nodes.hydro.near_field._gmres import REPORT_SIZE, Stopping

SPACING, RADIUS = 1e-4, 1.5e-4
OWN, GRID = 32, 24
#: The lines: the inertial step's screened images and Hasimoto's next term, each about a fifth of the line.
LINES = {"imex_euler": 1e-3, "steady": 3e-3}


def coupled(scheme: str, omit_box: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The handed 6x6 and the free-space MRS's own (with Hasimoto's term for the grid box in the steady limit); the
    body's place within a cell drawn from ``DETECTION_SEED``, the box omitted where ``MIME_PLANT_SCHWARZ=box``."""
    omit_box = omit_box or plant() == "box"
    offset = draw().uniform(-0.5, 0.5, 3) * SPACING
    with jax.enable_x64(True):
        member = sphere_member(RADIUS, 1e-4)
        box = box_for(RADIUS, SPACING, side=OWN)
        dt = 1e-3
        mu, rho = 1e-3, 1e3
        if scheme == "imex_euler":
            node = tube_flow(GRID, GRID, SPACING, 1.0, dt, scheme)  # a tube far wider than the box: no wall in it
            at = pieces(node, dt)
            grid, walled, shift, nu = at.grid, at.walled, at.shift, at.nu
        else:
            grid = MACGrid((GRID,) * 3, SPACING)
            free = own_box(GRID, SPACING, jnp.float64)
            shift, nu = 0.0, mu / rho

            def walled(rhs, guess=None):
                return own_response(free, tuple(rho * r for r in rhs), shift, nu, rho), jnp.zeros(REPORT_SIZE)

        centre = jnp.asarray([*(0.5 * GRID * SPACING + offset), 1.0, 0.0, 0.0, 0.0])
        body = schwarz_body(member, centre, mu)
        held = stencils(grid, box, body.placed.force, 1.0)
        now, _ = image_terms(box.side, box.spacing, shift, nu, mu)
        if omit_box:
            from mime.nodes.flow.fvm import _schwarz

            original = _schwarz.own_response
            _schwarz.own_response = lambda box, density, shift, nu, rho: tuple(jnp.zeros_like(d) for d in density)
        try:
            zero = jnp.zeros((OWN,) * 3)
            made = prepared(grid, box, held, body, lambda r: walled(r), grid.zeros(jnp.float64), (zero, zero, zero),
                            jnp.zeros(3), shift, nu, rho, now, Stopping(1e-11), jnp.zeros(2 * REPORT_SIZE))
        finally:
            if omit_box:
                _schwarz.own_response = original
        placed = body.placed
        free = np.asarray(force_sums(placed.wrench(), jnp.linalg.solve(body.matrix, placed.motions())))
        if scheme == "steady":
            mobility = np.linalg.inv(free)
            mobility[:3, :3] -= HASIMOTO / (6.0 * math.pi * mu * GRID * SPACING) * np.eye(3)
            free = np.linalg.inv(mobility)
        return np.asarray(made.resistance), free


@pytest.mark.parametrize("scheme", ["imex_euler", "steady"])
def test_one_body_coupled_through_a_grid_with_no_wall_is_the_body_uncoupled(scheme):
    found, wanted = coupled(scheme)
    scale = np.sqrt(np.abs(np.diag(wanted)))
    error = float(np.max(np.abs((found - wanted) / np.outer(scale, scale))))
    assert error < LINES[scheme], error


@pytest.mark.parametrize("scheme", ["imex_euler", "steady"])
def test_the_free_box_omitted_counts_the_body_twice(scheme):
    """Caught at the gate's own line: under the inertial step the grid's own response of the body is screened (1.6 %
    of the 6x6 measured locally, sixteen times the line), in the steady limit of the order of the 6x6 itself."""
    found, wanted = coupled(scheme, omit_box=True)
    scale = np.sqrt(np.abs(np.diag(wanted)))
    assert float(np.max(np.abs((found - wanted) / np.outer(scale, scale)))) > LINES[scheme]
