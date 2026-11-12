"""Gate 4 (#351): a body in contact folds the handed response into its own step. An MRS sphere resting on a small
walled tube's wall (its surface ``MARGIN`` from the wall, on the confinement's shrunk surface), pressed into it and
pushed along it, moving along it: the flow prepares its step (seven columns by GMRES), the rigid body's own step
(``RigidInertialNode.stepped`` and ``held``, the confinement held inside its solve) takes the handed ``A0`` and ``R``
with its other loads, and the step is composed from the twist it holds.

The reference is the monolithic step with the active contact row held as an equality (``schwarz_cases.monolithic_step``
with the contact's normal: the field's every face, the tractions, the twist and the contact's multiplier as one dense
LU). The case: the composed field, the tractions and the twist within ``LINE`` of the monolithic step's, the normal
velocity held at the gap's closing, and the multiplier positive in both (the contact active, its premise). The body's
place along the wall within a cell, its twist along the wall and its loads are drawn from ``DETECTION_SEED``. The
control, the handed response replaced by its diagonal blocks (``MIME_PLANT_SCHWARZ=diagonal``: the coupling of
translation and rotation near the wall dropped, what turns a sphere pushed along a wall), misses past ``LINE``."""

import jax
import jax.numpy as jnp
import numpy as np
from schwarz_cases import box_for, carried_field, decayed, draw, monolithic_step, pieces, plant, schwarz_body, \
    sphere_member, tube_flow

from mime.nodes.body._rigid_inertial import RigidInertialNode
from mime.nodes.flow.fvm._schwarz import RESIDUAL_LINE, composed, image_terms, prepared, stencils
from mime.nodes.hydro.near_field._gmres import REPORT_SIZE, Stopping

SPACING = 1e-4
CELLS, LENGTH = 10, 8
TUBE = 4e-4
RADIUS = 1.5e-4
#: The sphere's surface from the wall (m): a quarter of a cell.
MARGIN = 2.5e-5
OUTER = 1e-10
BOX = 32
#: The line, gate 3's: the solves' tolerances times the conditioning's margin.
LINE = 1e-7
DT = 1e-3
MASS = 1.4e-8


def body_node(centre: np.ndarray, pose: np.ndarray, twist: np.ndarray) -> RigidInertialNode:
    surface = {"surface_centre": (float(centre[0]), float(centre[1]), 0.0), "surface_axis": (0.0, 0.0, 1.0),
               "surface_radius": TUBE}
    return RigidInertialNode("ball", DT, mass=MASS, inertia=(0.4 * MASS * RADIUS ** 2,) * 3,
                             initial_pose=tuple(float(c) for c in pose), initial_twist=tuple(float(c) for c in twist),
                             surface_shape="cylinder", extent=RADIUS, margin=MARGIN, velocity_rule="remove_outward",
                             **surface)


def case(diagonal: bool = False) -> dict[str, float]:
    diagonal = diagonal or plant() == "diagonal"
    with jax.enable_x64(True):
        node = tube_flow(CELLS, LENGTH, SPACING, TUBE, DT)
        at = pieces(node, DT)
        member = sphere_member(RADIUS, 1e-4)
        rng = draw()
        axis = np.array([0.5 * CELLS * SPACING, 0.5 * CELLS * SPACING])
        turn = rng.uniform(0.0, 2.0 * np.pi)
        outward = np.array([np.cos(turn), np.sin(turn), 0.0])
        along = np.array([-np.sin(turn), np.cos(turn), 0.0])
        reach = TUBE - RADIUS - MARGIN
        centre = np.array([*axis, 0.5 * LENGTH * SPACING + rng.uniform(-0.5, 0.5) * SPACING]) + reach * outward
        pose = np.array([*centre, 1.0, 0.0, 0.0, 0.0])
        body = schwarz_body(member, jnp.asarray(pose), at.mu)
        box = box_for(RADIUS, SPACING, side=BOX)
        held = stencils(at.grid, box, body.placed.force, 1.0)
        grid = at.grid
        before = at.project(tuple(jnp.asarray(1e-4 * rng.standard_normal(grid.shape(k))) for k in range(3)))
        p = node.params_pytree()
        drive = node.drive(node.fields(p, grid), jnp.asarray([0.0, 0.0, 2.0]))
        rhs = tuple(at.shift * before[k] + drive[k] for k in range(3))
        base, _ = at.walled(rhs)
        carried = decayed(box, carried_field(box, int(rng.integers(1 << 30)), 1e-9), at)
        image = jnp.asarray(rng.uniform(-3e-6, 3e-6, 3))
        now, _ = image_terms(box.side, box.spacing, at.shift, at.nu, at.mu)
        step = prepared(grid, box, held, body, lambda r: at.walled(r), base, carried, image, at.shift, at.nu, at.rho,
                        now, Stopping(OUTER), jnp.zeros(2 * REPORT_SIZE))
        # moving along the wall (no normal velocity), pressed into it (2 nN) and pushed along it and the axis
        twist0 = np.concatenate([rng.uniform(-1.0, 1.0) * 2e-4 * along + rng.uniform(-1.0, 1.0) * 2e-4 * np.eye(3)[2],
                                 rng.uniform(-1.0, 1.0, 3) * 0.3])
        applied = np.concatenate([2e-9 * outward + rng.uniform(-1.0, 1.0, 2) @ np.stack([along, np.eye(3)[2]]) * 2e-9,
                                  np.zeros(3)])
        ball = body_node(axis, pose, twist0)
        bp = ball.params_pytree()
        resistance = np.asarray(step.resistance)
        if diagonal:
            resistance = np.block([[resistance[:3, :3], np.zeros((3, 3))], [np.zeros((3, 3)), resistance[3:, 3:]]])
        given = {"resistance": jnp.asarray(resistance), "background_load": step.load, "applied_wrench": jnp.asarray(applied)}
        mass_matrix, system, moved = ball.stepped(bp, jnp.asarray(pose), jnp.asarray(twist0), given, DT)
        twist = np.asarray(ball.held(bp, jnp.asarray(centre), system, moved, DT))
        done = composed(grid, box, held, body, step, jnp.asarray(twist), lambda r: at.walled(r), base, carried, image,
                        at.shift, at.nu, at.rho, now)
        mass = np.asarray(mass_matrix) / DT
        normal = np.concatenate([outward, np.zeros(3)])
        reference = monolithic_step(at, box, held, body, rhs, carried, image, mass, mass @ twist0 + applied, None,
                                    normal=normal, closing=0.0)

        def relative(found, wanted):
            found, wanted = np.asarray(found).ravel(), np.asarray(wanted).ravel()
            return float(np.linalg.norm(found - wanted) / np.linalg.norm(wanted))

        field = np.concatenate([np.ravel(c) for c in done.field])
        wanted = np.concatenate([np.ravel(c) for c in reference.velocity])
        speed = float(np.linalg.norm(twist[:3]))
        return dict(field=relative(field, wanted), tractions=relative(done.tractions, reference.tractions),
                    twist=relative(twist, reference.twist), residual=float(done.residual),
                    normal=float(abs(normal @ twist)) / speed, outward=float(normal @ np.asarray(moved)) / speed,
                    multiplier=float(reference.multiplier), at_cap=float(np.asarray(step.report)[2]))


def test_a_body_in_contact_folds_the_handed_response_into_its_step():
    found = case()
    print(f"F5 contact: {found}")
    assert found["outward"] > 0.0 and found["multiplier"] > 0.0, found  # the contact active: the case's premise
    assert found["field"] < LINE and found["tractions"] < LINE and found["twist"] < LINE, found
    assert found["normal"] < LINE and found["residual"] < RESIDUAL_LINE * OUTER and found["at_cap"] == 0.0, found


def test_the_handed_response_cut_to_its_diagonal_blocks_misses():
    found = case(diagonal=True)
    print(f"F5 contact, diagonal blocks: {found}")
    assert found["twist"] > LINE, found
