"""Gate 4, the cross-kernel's reach per registered hull kind (C1, 3): for each kind the mrs rung serves (a sphere, a
capsule, a screw, a cylinder, a box and an ellipsoid), two bodies of that kind side by side, the gap between them the
default reach (4) times their cells' mean size, where the joint node takes the kernel at the force points: the grand
resistance by the point kernel there against the integrated one within 0.5 % of each entry's diagonal scale. The
control, a reach planted short (0.5), judged at its own edge: every kind fails it, so the case fails a kind that needs a
longer reach than the one the node holds."""

import math

import numpy as np
import pytest
from near_field_cases import NODE, bodies, entrywise, joint_node, mrs, precision, rebuilt, start_inputs
from mrs_cases import BALL, PILL, SOLIDS
from script_a import fl9

from mime.build._maddening import parameter_leaves
from mime.nodes.hydro.near_field import REACH

KINDS = {"sphere": (BALL, 0.4e-3), "capsule": (PILL, 0.4e-3), "screw": (fl9(), 0.9e-3),
         "cylinder": (SOLIDS["rod"], 0.35e-3), "box": (SOLIDS["brick"], 0.35e-3), "ellipsoid": (SOLIDS["egg"], 0.35e-3)}
TOLERANCE = 5e-3
PLANT = 0.5


def edge_case(kind_name: str, reach: float) -> float:
    """The point kernel against the integrated one, the bodies ``reach`` mean cell sizes apart where nearest."""
    kind, spacing = KINDS[kind_name]
    with precision("float64"):
        probe = bodies("probe", {"a": (kind, (0.0, 0.0, 0.0)), "b": (kind, (0.05, 0.0, 0.0))}, mrs(spacing)).build()
        node = joint_node(probe)
        operator = node._operators[0]  # noqa: SLF001 - its cells' sizes
        p = node.constants(parameter_leaves(probe.graph)[NODE])
        inputs = start_inputs(probe)
        poses = np.stack([np.asarray(inputs[f"pose_{i}"]) for i in range(2)])
        placed = operator.placed(node._leaves(0, p), poses[0])  # noqa: SLF001
        size = float(np.mean(np.sqrt(np.asarray(placed.area))))
        force = np.asarray(placed.force)
        extent = float(force[:, 0].max() - force[:, 0].min())
        gap = reach * size
        poses[1, 0] = poses[0, 0] + extent + gap
        # Each arm's own reach in the leaves it is solved with: the reach is a leaf, so the built node's (``p``) would
        # hold both arms at the default.
        point = np.asarray(rebuilt(node, reach=1e-12).solved({**p, "reach": 1e-12}, poses)[0])
        integrated = np.asarray(rebuilt(node, reach=1e9, cross_share=1.0).solved({**p, "reach": 1e9}, poses)[0])
    return entrywise(0.5 * (point + point.T), 0.5 * (integrated + integrated.T))


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_each_kind_conforms_at_the_default_reach_and_fails_a_reach_planted_short(kind):
    found, planted = edge_case(kind, REACH), edge_case(kind, PLANT)
    print(f"\n{kind}: point against integrated at the reach's edge {found:.3g} (tolerance {TOLERANCE:g}); at a reach "
          f"of {PLANT}: {planted:.3g}")
    assert found <= TOLERANCE
    assert planted > TOLERANCE, "the control: a reach planted short fails the kind's case"
