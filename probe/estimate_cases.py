"""The fifth gates probe's estimates (raw programs): the walls gate's step prepared at D/h 32 for its largest sphere
(a/R 0.34, a cell from the wall), the heaviest of the tube arms; and the far-field gate's helix built and scattered into
its 128-cell box."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "flow"))
sys.path.insert(0, str(ROOT / "tests"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def walls_step():
    from schwarz_cases import TUBE_CELLS, box_for, flux_free, schwarz_body, sphere_member, steady_tube

    from mime.nodes.flow.fvm._schwarz import image_terms, prepared, stencils
    from mime.nodes.hydro.near_field._gmres import REPORT_SIZE, Stopping

    tube = 1e-3
    spacing = 2.0 * tube / TUBE_CELLS
    radius = 0.34 * tube
    _, at = steady_tube(tube, (0.0, 0.0), 1e-8)
    walled = flux_free(at)
    member = sphere_member(radius, 0.2512 * radius)
    middle = 0.5 * (TUBE_CELLS + 8) * spacing
    pose = jnp.asarray([middle + tube - radius - spacing, middle, TUBE_CELLS * spacing, 1.0, 0.0, 0.0, 0.0])
    body = schwarz_body(member, pose, at.mu)
    box = box_for(radius, spacing)
    held = stencils(at.grid, box, body.placed.force, 1.0)
    now, _ = image_terms(box.side, box.spacing, 0.0, at.nu, at.mu)
    still = jnp.zeros((box.side,) * 3)

    def step(base):
        return prepared(at.grid, box, held, body, walled, base, (still, still, still), jnp.zeros(3), 0.0, at.nu,
                        at.rho, now, Stopping(1e-6), jnp.zeros(2 * REPORT_SIZE)).resistance

    return jax.jit(step), (at.grid.zeros(jnp.float64),)


def far_field():
    from test_schwarz_far_field import MU, RHO, SIDE, SPACING, helix_member
    from schwarz_cases import schwarz_body

    from mime.nodes.flow.fvm._exchange import own_box, scatter
    from mime.nodes.flow.fvm._grid import MACGrid
    from mime.nodes.flow.fvm._schwarz import own_response

    member = helix_member()
    centre = 0.5 * SIDE * SPACING
    body = schwarz_body(member, jnp.asarray([centre, centre, centre, 1.0, 0.0, 0.0, 0.0]), MU)
    grid = MACGrid((SIDE,) * 3, SPACING)
    box = own_box(SIDE, SPACING, jnp.float64)
    held = [grid.exchange_stencil(k, body.placed.force, 1.0) for k in range(3)]

    def scattered(matrix, motions):
        forces = jnp.linalg.solve(matrix, motions)[:, 3].reshape(-1, 3) * body.placed.area[:, None]
        return own_response(box, tuple(scatter(grid, k, forces[:, k], held[k]) for k in range(3)), 0.0, MU / RHO, RHO)

    return jax.jit(scattered), (body.matrix, body.placed.motions())
