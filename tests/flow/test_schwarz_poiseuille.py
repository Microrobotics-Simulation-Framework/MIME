"""Gate 7: the read of a provided flow. An MRS sphere held, then free, in a straight tube's Poiseuille flow on the FVM
(D/h 32, Stokes, the steady limit; the base the walled response to a uniform axial force, the body's reflection
flux-free so that the flux stays the base's, an infinite tube's): held, the load at zero twist, ``A0``; free (no load,
overdamped), the twist ``R^-1 A0``.

**References.** On the axis, Haberman and Sayre's sphere in a tube's Poiseuille flow [recalled: W. L. Haberman and
R. M. Sayre, David Taylor Model Basin Report 1143, 1958], ``lambda = a/R``: held,
``F / (6 pi mu a V) = (1 - 2/3 lambda^2 - 0.20217 lambda^5) / (1 - 2.1050 lambda + 2.0865 lambda^3 - 1.7068 lambda^5 +
0.72603 lambda^6)``; free, ``U / V = (1 - 2/3 lambda^2 - 0.20217 lambda^5) / (1 - 0.75857 lambda^5)``, ``V`` the
centreline speed. Off the axis at small a/R, free-space Faxén's free sphere, ``U = u(b) + a^2/6 lap u =
V (1 - b^2/R^2 - 2/3 lambda^2)``, the walls' share a higher order there. ``V`` and ``R`` are the base's own, fitted to
its axial faces within 0.8 R (the band's wall sits a fraction of a cell from the nominal radius). The held force is
judged over the free sphere's own resistance (its discretisation's), as the walls' gate judges its 6x6.

**Lines** (percent): the exchange kernel's read of a curved flow is its average over the kernel, ``2 h^2 / (R^2 - b^2)``
of the local speed low for a parabola (0.78 % on the axis at D/h 32, 0.94 % at b/R 0.4: measured, the base gathered at a
point against the fitted profile); held, ``HELD_LINE``, that read and the grid's floor on the axis (the walls' gate's
band at the axis, -0.4 to -1.9 % from a/R 0.1 to 0.3); free, ``FREE_LINE``, that read where the resistance's and the
load's floors cancel; Faxén, ``FAXEN_LINE``, that read and the walls' share Faxén leaves out. The lattice alignment is
drawn from ``DETECTION_SEED``. **Control:** the read removed (``MIME_PLANT_SCHWARZ=unread``: the
base a medium at rest's zero), the held force and the free speed zero."""

import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import TUBE_CELLS, box_for, draw, flux_free, free_resistance, plant, schwarz_body, sphere_member, \
    steady_tube

from mime.nodes.flow.fvm._schwarz import image_terms, prepared, stencils
from mime.nodes.hydro.near_field._gmres import REPORT_SIZE, Stopping

TUBE = 1e-3
SPACING = 2.0 * TUBE / TUBE_CELLS
FORCE_SPACING = 0.2512
OUTER, INNER = 1e-6, 1e-8
#: The base's centreline speed (m/s), by its uniform force ``4 nu V / R^2``.
SPEED = 1e-3
HELD_LINE, FREE_LINE, FAXEN_LINE = 3.0, 1.5, 2.0
#: The cases: a/R, b/R and the reference judged.
CASES = [(0.1, 0.0, "axis"), (0.2, 0.0, "axis"), (0.3, 0.0, "axis"), (0.1, 0.4, "faxen")]


def held_over_stokes(lam: float) -> float:
    return (1.0 - 2.0 / 3.0 * lam ** 2 - 0.20217 * lam ** 5) / (
        1.0 - 2.1050 * lam + 2.0865 * lam ** 3 - 1.7068 * lam ** 5 + 0.72603 * lam ** 6)


def free_over_centreline(lam: float) -> float:
    return (1.0 - 2.0 / 3.0 * lam ** 2 - 0.20217 * lam ** 5) / (1.0 - 0.75857 * lam ** 5)


def fitted(at, base, centre: np.ndarray) -> tuple[float, float]:
    """The base's centreline speed and radius: ``u = V (1 - r^2 / R^2)`` fitted to its axial faces within 0.8 R."""
    cells = at.grid.cells
    x = (np.arange(cells[0]) + 0.5) * SPACING - centre[0]
    y = (np.arange(cells[1]) + 0.5) * SPACING - centre[1]
    squared = (x[:, None] ** 2 + y[None, :] ** 2)
    speed = np.asarray(base[2], dtype=np.float64).mean(axis=2)
    inside = squared < (0.8 * TUBE) ** 2
    slope, top = np.polyfit(squared[inside], speed[inside], 1)
    return float(top), float(np.sqrt(-top / slope))


def case(a_over_r: float, b_over_r: float) -> dict[str, float]:
    rng = draw()
    radius = a_over_r * TUBE
    across = rng.uniform(0.0, 1.0, 2) * SPACING
    node, at = steady_tube(TUBE, across, INNER)
    kind = jnp.zeros(()).dtype
    zeros = at.grid.zeros(kind)
    middle = 0.5 * (TUBE_CELLS + 8) * SPACING
    centre = np.array([middle + across[0], middle + across[1]])
    force = 4.0 * at.nu * SPEED / TUBE ** 2
    base, _ = at.walled((zeros[0], zeros[1], jnp.full_like(zeros[2], force)))
    centreline, wall = fitted(at, base, centre)
    if plant() == "unread":
        base = zeros
    member = sphere_member(radius, FORCE_SPACING * radius)
    pose = jnp.asarray([centre[0] + b_over_r * TUBE, centre[1], (TUBE_CELLS + rng.uniform(0.0, 1.0)) * SPACING,
                        1.0, 0.0, 0.0, 0.0], dtype=kind)
    body = schwarz_body(member, pose, at.mu)
    free = free_resistance(body)
    box = box_for(radius, SPACING)
    held = stencils(at.grid, box, body.placed.force, 1.0)
    now, _ = image_terms(box.side, box.spacing, 0.0, at.nu, at.mu)
    still = jnp.zeros((box.side,) * 3, dtype=kind)
    made = prepared(at.grid, box, held, body, flux_free(at), base, (still, still, still), jnp.zeros(3, dtype=kind),
                    0.0, at.nu, at.rho, now, Stopping(OUTER), jnp.zeros(2 * REPORT_SIZE, dtype=kind))
    load = np.asarray(made.load, dtype=np.float64)
    twist = np.linalg.solve(np.asarray(made.resistance, dtype=np.float64), load)
    lam = radius / wall
    faxen = centreline * (1.0 - (b_over_r * TUBE / wall) ** 2 - 2.0 / 3.0 * lam ** 2)
    return dict(held=100.0 * (load[2] / (free[2, 2] * centreline) / held_over_stokes(lam) - 1.0),
                free=100.0 * (twist[2] / centreline / free_over_centreline(lam) - 1.0),
                faxen=100.0 * (twist[2] / faxen - 1.0), across=float(np.hypot(twist[0], twist[1]) / centreline),
                centreline=centreline, wall=wall / TUBE, at_cap=float(np.asarray(made.report)[2]))


@pytest.mark.parametrize(("a_over_r", "b_over_r", "judge"), CASES, ids=[f"{a}-{b}" for a, b, _ in CASES])
def test_a_sphere_held_then_free_reads_the_tubes_poiseuille_flow(float64, a_over_r, b_over_r, judge):
    found = case(a_over_r, b_over_r)
    print(f"F5 Poiseuille [a/R {a_over_r}, b/R {b_over_r}]: {found}")
    assert found["at_cap"] == 0.0, found
    if judge == "axis":
        assert abs(found["held"]) <= HELD_LINE and abs(found["free"]) <= FREE_LINE, found
    else:
        assert abs(found["faxen"]) <= FAXEN_LINE, found
    assert found["across"] < 1e-2, found  # no migration across the flow in Stokes flow
