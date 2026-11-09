"""Gate 8, the seam as a pure function of geometry (Bundle 2; C3's multi-contact report: W2's lift evaluates the coarse
MRS with the walls and the neighbours moved per contact): the joint node's grand resistance evaluated with a neighbour
moved along the line of centres, the tube's radius or the plane's offset moved, equals a fresh build at that geometry
to rounding (1e-12 of each entry's diagonal scale, float64), and its derivative through the move is the central
difference's to the difference's own resolution. The control: the moved geometry against the build it was moved from, far past the line,
so that the comparison sees the move."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import NODE, bodies, entrywise, joint_node, mrs, precision, spheres, start_inputs

from mime.build._maddening import parameter_leaves

SPACING = 0.6e-3
LINE = 1e-12


def grand_at(node, params, poses):
    found = node.solved(node.constants(params), poses)[0]
    return 0.5 * (found + found.T)


def poses_of(built):
    inputs = start_inputs(built)
    return jnp.stack([jnp.asarray(inputs[f"pose_{i}"]) for i in range(len(joint_node(built).members))])


@pytest.mark.parametrize("tube", [None, 6e-3], ids=["free_space", "tube"])
def test_a_neighbour_moved_along_the_line_of_centres_is_a_fresh_build_there(tube):
    law = mrs(SPACING, wall_kind="tube") if tube else mrs(SPACING)
    with precision("float64"):
        near = bodies("near", spheres(0.3e-3, 2), law, tube=tube).build()
        far = bodies("far", spheres(0.5e-3, 2), law, tube=tube).build()
        node, params = joint_node(near), parameter_leaves(near.graph)[NODE]
        moved = np.asarray(grand_at(node, params, poses_of(near).at[1, 0].add(0.2e-3)))
        fresh = np.asarray(grand_at(joint_node(far), parameter_leaves(far.graph)[NODE], poses_of(far)))
        before = np.asarray(grand_at(node, params, poses_of(near)))
    found, control = entrywise(moved, fresh), entrywise(before, fresh)
    print(f"\n{'tube' if tube else 'free space'}: moved against fresh {found:.3g} (line {LINE:g}); unmoved {control:.3g}")
    assert found <= LINE and control > 1e6 * LINE


def test_the_tubes_radius_moved_in_the_call_is_a_fresh_build_at_that_radius():
    law = mrs(SPACING, wall_kind="tube")
    with precision("float64"):
        narrow = bodies("narrow", spheres(0.4e-3, 2), law, tube=6e-3).build()
        wide = bodies("wide", spheres(0.4e-3, 2), law, tube=6.5e-3).build()
        node, params = joint_node(narrow), dict(parameter_leaves(narrow.graph)[NODE])
        params["wall_radius"] = jnp.asarray(6.5e-3)
        moved = np.asarray(grand_at(node, params, poses_of(narrow)))
        fresh = np.asarray(grand_at(joint_node(wide), parameter_leaves(wide.graph)[NODE], poses_of(wide)))
        before = np.asarray(grand_at(node, parameter_leaves(narrow.graph)[NODE], poses_of(narrow)))
    found, control = entrywise(moved, fresh), entrywise(before, fresh)
    print(f"\nthe radius moved against a fresh build at it {found:.3g} (line {LINE:g}); unmoved {control:.3g}")
    assert found <= LINE and control > 1e3 * LINE


def test_the_planes_offset_moved_in_the_call_is_a_fresh_build_at_that_offset():
    """The plane wall kind's arm of the moved wall (its offset is the node's leaf ``wall_offset``, which a contact's
    opening moves): two spheres a radius above the plane, the offset moved by a tenth of a millimetre in the call
    against a fresh build there, the wall's image between the bodies and each one's own regularised by its blob."""
    law = mrs(SPACING, wall_kind="plane")
    with precision("float64"):
        near = bodies("near", spheres(0.4e-3, 2), law, plane=-2e-3).build()
        far = bodies("far", spheres(0.4e-3, 2), law, plane=-2.1e-3).build()
        node, params = joint_node(near), dict(parameter_leaves(near.graph)[NODE])
        params["wall_offset"] = jnp.asarray(-2.1e-3)
        moved = np.asarray(grand_at(node, params, poses_of(near)))
        fresh = np.asarray(grand_at(joint_node(far), parameter_leaves(far.graph)[NODE], poses_of(far)))
        before = np.asarray(grand_at(node, parameter_leaves(near.graph)[NODE], poses_of(near)))
    found, control = entrywise(moved, fresh), entrywise(before, fresh)
    print(f"\nthe plane's offset moved against a fresh build at it {found:.3g} (line {LINE:g}); unmoved {control:.3g}")
    assert found <= LINE and control > 1e3 * LINE


def test_the_grand_resistance_is_differentiable_through_a_neighbours_move():
    with precision("float64"):
        built = bodies("moved", spheres(0.3e-3, 2), mrs(SPACING)).build()
        node, params = joint_node(built), parameter_leaves(built.graph)[NODE]
        poses = poses_of(built)

        def squeeze(shift):
            grand = grand_at(node, params, poses.at[1, 0].add(shift))
            return grand[0, 0] - grand[0, 6]

        exact = float(jax.grad(squeeze)(0.0))
        coarse, fine = 2e-6, 1e-6
        differences = [(float(squeeze(h)) - float(squeeze(-h))) / (2 * h) for h in (coarse, fine)]
    resolution = abs(differences[0] - differences[1])
    print(f"\nd(squeeze)/d(separation): autodiff {exact:.10g}, central {differences[1]:.10g} (its resolution "
          f"{resolution:.3g})")
    assert abs(exact - differences[1]) <= 10.0 * resolution + 1e-9 * abs(exact)
