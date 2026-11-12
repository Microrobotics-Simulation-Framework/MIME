"""Gate 2 (C4): the walls carried by the grid. An MRS sphere in a straight tube at D/h 32 (Stokes, the steady limit;
the periodic tube flux-free, an infinite tube's stand-in), its near field in free space and the walls by the grid,
from about one coarse cell of gap to the axis, at a/R 0.1 to 0.34: the coupled 6x6's radial and axial resistance over
the free sphere's, against dev1's tube table read at the sphere's true position (``schwarz_cases.tube_table``) over
the same free sphere's.

**The line:** each pose's error ``K / K_table - 1`` reproduces the corrected band (the window spike's question 7, R2's
set-up with the sub-cell offset made its true position: ``Q7``) within ``2 pp + 0.1 |e_q7|``, its spread at a gap
(about a tenth of the band's size at one cell, and the floor's order at the axis). a/R 0.34 is beyond the band's
range: its band is the a/R 0.1 to 0.3 rows' trend carried on (``Q7``'s last row, an extrapolation). The grid's
lattice alignment (the tube's axis within a cell, the body's axial place) is drawn from ``DETECTION_SEED``.

**Controls** (``MIME_PLANT_SCHWARZ``): ``offset``, R2's own error, the reference read 0.37 of a cell farther from the
wall than the body (the table at the body's place before its sub-cell offset); ``image``, an image switch, the near
field carrying the tube's walls (its matrix plus the tube's image at the force points) while the grid carries them too,
the walls counted twice."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import TUBE_CELLS, box_for, draw, flux_free, free_resistance, plant, schwarz_body, sphere_member, \
    steady_tube, tube_table

from mime.nodes.flow.fvm._schwarz import SchwarzBody, image_terms, prepared, stencils
from mime.nodes.hydro.near_field._gmres import REPORT_SIZE, Stopping

TUBE = 1e-3
SPACING = 2.0 * TUBE / TUBE_CELLS
#: The force spacing over the radius: the window spike's 338-point sphere.
FORCE_SPACING = 0.2512
#: The columns' tolerance and the walled CG's (the inner share of it), by dtype: float32's CG above its reachable floor
#: in the steady limit at D/h 32 (at 1e-5 one pose took three times float64's time and matched it to 1e-4 pp), both far
#: under the band's percent.
TOLERANCES = {"float64": (1e-6, 1e-8), "float32": (1e-3, 1e-4)}
#: The band (the window spike's question 7: R2's set-up, the body at its true position), percent, by a/R: wall-normal
#: (radial) and wall-parallel (axial) by gap in coarse cells, and on the axis R2's floor (its gap the axis's). The
#: a/R 0.34 row carries the 0.1 to 0.3 rows' trend on linearly in a/R (an extrapolation, the band's range is to 0.3).
Q7 = {
    0.1: {"radial": {1: -32.17, 2: -12.44, 4: -3.15}, "axial": {1: -6.83, 2: -2.67, 4: -0.90}},
    0.2: {"radial": {1: -39.06, 2: -16.41, 4: -4.59, 8: -1.11, "axis": -0.59},
          "axial": {1: -6.86, 2: -3.07, 4: -1.43, 8: -0.99, "axis": -0.99}},
    0.3: {"radial": {1: -43.93, 2: -19.29, 4: -5.64}, "axial": {1: -6.99, 2: -3.55, 4: -2.12}},
    0.34: {"radial": {1: -45.88, 2: -20.44, 4: -6.06}, "axial": {1: -7.04, 2: -3.74, 4: -2.40}},
}
POSES = [(a, g) for a, rows in Q7.items() for g in rows["radial"]]
FLOAT32_POSES = [(0.2, 1), (0.2, 2), (0.34, 2)]
#: R2's sub-cell offset of the body from its reference, the planted control's (cells).
R2_OFFSET = 0.37


def line(wanted: float) -> float:
    return 2.0 + 0.1 * abs(wanted)


def doubled(body: SchwarzBody, centre: np.ndarray) -> SchwarzBody:
    """The image switch: the near field's matrix with the tube's own image at the force points added (the tube's wall
    kind's point reads, as the per-body node adds it far from the wall), preconditioned by its own inverse."""
    from mime.nodes.hydro.mrs._tube import TUBE_CORRECTION  # noqa: PLC0415

    placed = body.placed
    parameters = TUBE_CORRECTION.prepare({"centre": (float(centre[0]), float(centre[1]), 0.0), "axis": (0.0, 0.0, 1.0),
                                          "radius": TUBE}, placed.force.dtype)
    image = TUBE_CORRECTION.correction(placed.force, parameters, 1e-3).matrix * jnp.repeat(placed.area, 3)[None, :]
    matrix = body.matrix + image
    inverse = jnp.linalg.inv(matrix)  # its own inverse as the preconditioner, so that GMRES's cost stays the case's
    return body._replace(matrix=matrix, precondition=lambda v: inverse @ v, transposed=lambda v: inverse.T @ v)


def errors(a_over_r: float, gap: float | str, dtype: str) -> dict[str, float]:
    rng = draw()
    radius = a_over_r * TUBE
    across = rng.uniform(0.0, 1.0, 2) * SPACING
    offset = TUBE - radius - (gap * SPACING if gap != "axis" else TUBE - radius)
    outer, inner = TOLERANCES[dtype]
    node, at = steady_tube(TUBE, across, inner)
    kind = jnp.zeros(()).dtype
    walled = flux_free(at)
    member = sphere_member(radius, FORCE_SPACING * radius)
    middle = 0.5 * (TUBE_CELLS + 8) * SPACING
    centre = np.array([middle + across[0], middle + across[1]])
    pose = jnp.asarray([centre[0] + offset, centre[1], (TUBE_CELLS + rng.uniform(0.0, 1.0)) * SPACING,
                        1.0, 0.0, 0.0, 0.0], dtype=kind)
    body = schwarz_body(member, pose, at.mu)
    free = free_resistance(body)
    if plant() == "image":
        body = doubled(body, centre)
    box = box_for(radius, SPACING)
    held = stencils(at.grid, box, body.placed.force, 1.0)
    now, _ = image_terms(box.side, box.spacing, 0.0, at.nu, at.mu)
    still = jnp.zeros((box.side,) * 3, dtype=kind)
    made = prepared(at.grid, box, held, body, walled, at.grid.zeros(kind), (still, still, still),
                    jnp.zeros(3, dtype=kind), 0.0, at.nu, at.rho, now, Stopping(outer),
                    jnp.zeros(2 * REPORT_SIZE, dtype=kind))
    coupled = np.asarray(made.resistance, dtype=np.float64)
    read_at = offset - R2_OFFSET * SPACING if plant() == "offset" else offset
    radial, axial = tube_table(radius, FORCE_SPACING * radius, TUBE, read_at)
    report = np.asarray(made.report)
    return dict(radial=100.0 * (coupled[0, 0] / free[0, 0] / radial - 1.0),
                axial=100.0 * (coupled[2, 2] / free[2, 2] / axial - 1.0),
                products=float(report[0]), residual=float(report[1]), at_cap=float(report[2]))


def judged(a_over_r: float, gap: float | str, dtype: str) -> None:
    found = errors(a_over_r, gap, dtype)
    wanted = {key: Q7[a_over_r][key][gap] for key in ("radial", "axial")}
    print(f"F5 walls [{dtype}, a/R {a_over_r}, gap {gap}]: {found}, band {wanted}")
    assert found["at_cap"] == 0.0, found
    for key in ("radial", "axial"):
        assert abs(found[key] - wanted[key]) <= line(wanted[key]), (key, found, wanted)


@pytest.mark.parametrize(("a_over_r", "gap"), POSES, ids=[f"{a}-{g}" for a, g in POSES])
def test_the_walls_by_the_grid_reproduce_the_band(float64, a_over_r, gap):
    judged(a_over_r, gap, float64)


@pytest.mark.parametrize(("a_over_r", "gap"), FLOAT32_POSES, ids=[f"{a}-{g}" for a, g in FLOAT32_POSES])
def test_the_walls_by_the_grid_reproduce_the_band_in_float32(a_over_r, gap):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        judged(a_over_r, gap, "float32")
    finally:
        jax.config.update("jax_enable_x64", held)
