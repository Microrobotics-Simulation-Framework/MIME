"""Gate 2, several bodies against a dense joint system (C1, 4: "identical to a dense joint system"): the joint node's
grand resistance against one assembled independently from dev1's pieces (S3's pipeline: each body's free part made
afresh and turned, every cross-block the source's kernel integrated over all its cells, or taken at its force points,
one dense solve), with the node's reach set so that every cell is integrated (or none is), so that the two assemble
the same blocks: to rounding, 1e-12 of each entry's diagonal scale in float64. The node at its default reach lies
between the two, within the reach's error (gate 4 judges that per hull kind). The control: the bodies uncoupled
(the joint system with no cross-blocks), whose error against the reference is the interaction's own, orders above the
line. The transposed preconditioner is W3b's (GMRES) and a cost check, not an accuracy control."""

import numpy as np
import pytest
from near_field_cases import (NODE, bodies, entrywise, joint_node, mrs, precision, rebuilt, reference_grand, spheres,
                              start_inputs)

from mime.build._maddening import parameter_leaves
from mime.nodes.hydro.near_field import grand_resistance, joint_matrix

SPACING = 0.4e-3
LINE = 1e-12


def built_cluster(gap: float, count: int):
    return bodies("cluster", spheres(gap, count), mrs(SPACING)).build()


def poses_of(built) -> np.ndarray:
    inputs = start_inputs(built)
    return np.stack([np.asarray(inputs[f"pose_{i}"]) for i in range(len(joint_node(built).members))])


def grand_of(node, built) -> np.ndarray:
    out = node.update(node.initial_state(), start_inputs(built), node.delta_t, params=node.params_pytree())
    return np.asarray(out["resistance"], dtype=np.float64)


@pytest.mark.parametrize(("gap", "count"), [(0.5e-3, 2), (0.2e-3, 2), (0.3e-3, 3)])
def test_the_joint_nodes_grand_resistance_is_a_dense_joint_systems_to_rounding(gap, count):
    with precision("float64"):
        built = built_cluster(gap, count)
        node = joint_node(built)
        poses = poses_of(built)
        every = grand_of(rebuilt(node, reach=1e9, cross_share=1.0), built)
        none = grand_of(rebuilt(node, reach=1e-12), built)
        default = grand_of(node, built)
        integrated, pointed = reference_grand(node, poses, every=True), reference_grand(node, poses, every=False)
    found = entrywise(every, integrated), entrywise(none, pointed)
    between = entrywise(default, integrated)
    reach = entrywise(pointed, integrated)
    print(f"\n{count} spheres {gap * 1e3:g} mm apart: integrated {found[0]:.3g}, at the force points {found[1]:.3g} "
          f"(line {LINE:g}); the default reach against integrated {between:.3g}, the point kernel's own {reach:.3g}")
    assert max(found) <= LINE
    assert between <= reach


@pytest.mark.parametrize(("gap", "count"), [(0.5e-3, 2), (0.3e-3, 3)])
def test_control_bodies_that_do_not_see_each_other_are_far_from_the_dense_system(gap, count):
    with precision("float64"):
        built = built_cluster(gap, count)
        node = joint_node(built)
        poses = poses_of(built)
        leaves = parameter_leaves(built.graph)[NODE]
        p = node.constants(leaves)
        wall = node._wall()  # noqa: SLF001 - the node's own wall, as its step reads it
        held = [node._held_value(i, p) for i in range(len(node.members))]  # noqa: SLF001
        alone = grand_resistance(joint_matrix(node._operators, [node._leaves(i, p) for i in range(len(node.members))],  # noqa: SLF001
                                              held, np.asarray(poses), p["viscosity"], wall, {}, p["reach"],
                                              coupled=False))
        reference = reference_grand(node, poses, every=True)
    found = entrywise(0.5 * (np.asarray(alone) + np.asarray(alone).T), reference)
    print(f"\nuncoupled against the dense system: {found:.3g} (line {LINE:g})")
    assert found > 1e6 * LINE


#: Gate 11's float32 line for the grand resistance against the float64 dense system: the joint matrix's LU in float32
#: on a conditioned matrix (the sphere's free part's condition number is about 3e3 at this spacing). Measured at these
#: three configurations on its first run (a provisional 2e-4 written before it): 9.7e-7 to 1.0e-6; set at ten times that.
FLOAT32_LINE = 1e-5


@pytest.mark.parametrize(("gap", "count"), [(0.5e-3, 2), (0.2e-3, 2), (0.3e-3, 3)])
def test_in_float32_the_grand_resistance_is_the_float64_dense_systems_to_its_rounding(gap, count):
    with precision("float64"):
        built = built_cluster(gap, count)
        node = rebuilt(joint_node(built), reach=1e9, cross_share=1.0)
        reference = reference_grand(node, poses_of(built), every=True)
    with precision("float32"):
        single = built_cluster(gap, count)
        found = grand_of(rebuilt(joint_node(single), reach=1e9, cross_share=1.0), single)
    error = entrywise(found, reference)
    print(f"\nfloat32, {count} spheres {gap * 1e3:g} mm apart: against the float64 dense system {error:.3g} "
          f"(line {FLOAT32_LINE:g})")
    assert error <= FLOAT32_LINE
