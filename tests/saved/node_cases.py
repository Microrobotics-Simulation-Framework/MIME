"""The node cases the node walk requires: one per registered entry, saying how its nodes are verified (the options
MADDENING's ``verify_node`` draws with, and a refinement ladder for each order a node declares), or why it builds no
node. An entry registered without one fails the walk."""

import os

import rigid_oracles
from maddening.testing.mms import RefinementAxis

from mime.testing import NodeCase, OrderLadder, positive_semidefinite, unit_quaternion

#: A rigid body's drawn state is a pose with a unit quaternion, and its drawn resistance is positive semi-definite,
#: as every delivered resistance is, and of a body's order (entries up to 1e-3 before ``A A^T``: a millimetre body in
#: water is about 1e-5 N s/m): with the default box, ``A A^T`` reaches 1e8 in every block, ``M/dt + R`` is singular
#: to float64's resolution, and the battery would test the draw.
RIGID_BODY = NodeCase(
    options={"boundary_bounds": {"resistance": (-1e-3, 1e-3)},
             "constrain_state": lambda state: {**state, "pose": unit_quaternion(state["pose"])},
             "constrain_boundary": lambda given: {**given, "resistance": positive_semidefinite(given["resistance"])}},
    ladders={"": (OrderLadder(RefinementAxis.TIME, rigid_oracles.velocity_error, (25, 50, 100, 200)),)})

#: The drag's drawn pose is a body's: inside a box a little larger than its tube, its quaternion of unit length. In
#: the default box of 1e4 every body is far beyond the wall, where the clamp and the lubrication's gap floor hold every
#: output still.
DRAG = NodeCase(options={"boundary_bounds": {"pose": (-4e-3, 4e-3)},
                         "constrain_boundary": lambda given: {**given, "pose": unit_quaternion(given["pose"])}})

NODE_CASES: dict[tuple[str, str, str], NodeCase] = {
    ("dynamics", "toy.body", "0.1.0"): NodeCase(),
    ("dynamics", "toy.anchored_body", "0.1.0"): NodeCase(),
    ("dynamics", "toy.relaxation", "0.1.0"): NodeCase(),
    ("field provider", "toy.field", "0.1.0"): NodeCase(),
    ("interaction", "toy.spring", "0.1.0"): NodeCase(),
    ("interaction", "toy.drag", "0.2.0"): NodeCase(),
    ("interaction", "toy.lift", "0.1.0"): NodeCase(),
    ("interaction", "toy.hold", "0.1.0"): NodeCase(),
    ("interaction", "toy.couple", "0.1.0"): NodeCase(
        no_node="a registration with the medium's dynamics: the medium's node reads the body and loads it"),
    ("interaction", "toy.twist", "0.1.0"): NodeCase(
        no_node="refused at plan time: it delivers into a reserved interface"),
    ("observation", "toy.recorder", "0.1.0"): NodeCase(),
    ("interaction", "labx.nudge", "0.1.0"): NodeCase(),
    ("dynamics", "rigid_inertial", "0.2.0"): RIGID_BODY,
    ("field provider", "uniform_rotating_field", "0.1.0"): NodeCase(),
    ("interaction", "magnetic.torque", "0.1.0"): NodeCase(),
    ("interaction", "gravity", "0.1.0"): NodeCase(),
    ("interaction", "drag", "0.2.0"): DRAG,
    ("interaction", "contact", "0.1.0"): NodeCase(
        no_node="a projection: the body's dynamics takes its record at construction, and that node is "
                "rigid_inertial's, verified under its own case"),
    ("observation", "swim_speed", "0.1.0"): NodeCase(),
    ("drive", "drive.constant", "0.1.0"): NodeCase(),
    ("geometry", "tube", "0.1.0"): NodeCase(
        no_node="a geometry kind: the nodes that read its surfaces and directions hold their parameters"),
    **{("geometry", solid, "0.1.0"): NodeCase(
        no_node="a solid: the part resolver takes its volume and inertia, and the nodes that read the body's values "
                "derived from them hold those") for solid in ("solid_box", "solid_cylinder", "solid_ellipsoid")},
}

if os.environ.get("MIME_CI_PLANT_FAILURE") == "1":
    # The control that the walk fails CI when an entry's nodes have no case: the planted run drops toy.anchored_body's,
    # as cases.py drops the only round-trip case that uses it.
    del NODE_CASES[("dynamics", "toy.anchored_body", "0.1.0")]
