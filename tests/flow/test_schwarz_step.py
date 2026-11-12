"""Gate 3's core, on the Schwarz functions themselves (``mime.nodes.flow.fvm._schwarz``): an MRS sphere in a small
walled tube, its step prepared (seven columns by GMRES, the walled CG inside each product), its body stepping through
the handed response, the step composed from its twist, against the monolithic step (one dense LU of the field's every
face, the tractions and the twist: ``schwarz_cases.monolithic_step``), in float64, with IMEX Euler and its steady limit,
with a history in the free box and its image term, the body off the axis and moving.

The case: the composed field, the tractions and the twist each within ``LINE`` of the monolithic step's, relative to
its own size; the near field's residual at the composed field under the outer tolerance. The control, GMRES capped at
one product: its report counts every column stopped at the cap, and its twist misses the monolithic one past ``LINE``,
and its composed residual is past the flag's line (the step is flagged, never silent)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import box_for, carried_field, decayed, monolithic_step, pieces, schwarz_body, sphere_member, \
    tube_flow

from mime.nodes.flow.fvm._schwarz import RESIDUAL_LINE, composed, image_terms, prepared, stencils
from mime.nodes.hydro.near_field._gmres import REPORT_SIZE, Stopping

SPACING = 1e-4
CELLS, LENGTH = 10, 8
TUBE = 4e-4
RADIUS = 1.5e-4
OUTER = 1e-10
#: The free box's side in cells: the sphere's four cells and the kernel's reach of eight each side inside it.
BOX = 32
#: The case's line: the solves' tolerances times the conditioning's margin.
LINE = 1e-7


def case(scheme: str, cap: int | None = None) -> dict[str, float]:
    with jax.enable_x64(True):
        dt = 1e-3
        node = tube_flow(CELLS, LENGTH, SPACING, TUBE, dt, scheme)
        at = pieces(node, dt)
        member = sphere_member(RADIUS, 1e-4)
        centre = np.array([0.5 * CELLS * SPACING + 0.4e-4, 0.5 * CELLS * SPACING - 0.2e-4, 0.5 * LENGTH * SPACING])
        pose = jnp.asarray([*centre, 1.0, 0.0, 0.0, 0.0])
        body = schwarz_body(member, pose, at.mu)
        box = box_for(RADIUS, SPACING, side=BOX)
        held = stencils(at.grid, box, body.placed.force, 1.0)
        rng = np.random.default_rng(7)
        grid = at.grid
        before = at.project(tuple(jnp.asarray(1e-3 * rng.standard_normal(grid.shape(k))) for k in range(3)))
        p = node.params_pytree()
        fields = node.fields(p, grid)
        drive = node.drive(fields, jnp.asarray([0.0, 0.0, 2.0]))
        rhs = tuple(at.shift * before[k] + drive[k] for k in range(3))
        base, _ = at.walled(rhs)
        carried = decayed(box, carried_field(box, 3, 1e-9), at)
        image = jnp.asarray([2e-6, -1e-6, 3e-6])
        now, _ = image_terms(box.side, box.spacing, at.shift, at.nu, at.mu)
        stopping = Stopping(OUTER) if cap is None else Stopping(OUTER, restart=1, cap=cap)
        step = prepared(grid, box, held, body, lambda r: at.walled(r), base, carried, image, at.shift, at.nu, at.rho,
                        now, stopping, jnp.zeros(2 * REPORT_SIZE))
        mass = np.diag([3e-8] * 3 + [5e-16] * 3) / dt
        twist0 = np.array([1e-4, 0.0, -2e-4, 0.3, -0.1, 0.2])
        right = mass @ twist0 + np.array([1e-9, 0.0, -2e-9, 0.0, 1e-16, 0.0])
        twist = np.linalg.solve(mass + np.asarray(step.resistance), right + np.asarray(step.load))
        done = composed(grid, box, held, body, step, jnp.asarray(twist), lambda r: at.walled(r), base, carried, image,
                        at.shift, at.nu, at.rho, now)
        reference = monolithic_step(at, box, held, body, rhs, carried, image, mass, right, None)

        def relative(found, wanted):
            found, wanted = np.asarray(found).ravel(), np.asarray(wanted).ravel()
            return float(np.linalg.norm(found - wanted) / np.linalg.norm(wanted))

        field = np.concatenate([np.ravel(c) for c in done.field])
        wanted = np.concatenate([np.ravel(c) for c in reference.velocity])
        report = np.asarray(step.report)
        return dict(field=relative(field, wanted), tractions=relative(done.tractions, reference.tractions),
                    twist=relative(twist, reference.twist), residual=float(done.residual),
                    at_cap=float(report[2]), gmres=float(report[0]), inner=float(report[REPORT_SIZE]))


@pytest.mark.parametrize("scheme", ["imex_euler", "steady"])
def test_the_schwarz_step_is_the_monolithic_step(scheme):
    found = case(scheme)
    assert found["field"] < LINE and found["tractions"] < LINE and found["twist"] < LINE, found
    assert found["residual"] < RESIDUAL_LINE * OUTER, found
    assert found["at_cap"] == 0.0 and found["inner"] >= found["gmres"], found


def test_gmres_capped_at_one_product_is_flagged_and_misses():
    found = case("imex_euler", cap=1)
    assert found["at_cap"] >= 1.0, found
    assert found["twist"] > LINE, found
    assert found["residual"] > RESIDUAL_LINE * OUTER, found
