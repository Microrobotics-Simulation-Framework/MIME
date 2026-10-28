"""The node cases the node walk requires: one per registered entry, saying how its nodes are verified (the options
MADDENING's ``verify_node`` draws with, and a refinement ladder for each order a node declares), or why it builds no
node. An entry registered without one fails the walk."""

import os

import numpy as np
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

#: A chain segment's drawn mount poses hold unit quaternions, as every frame pose a segment publishes does.
def unit_poses(poses):
    held = np.asarray(poses)
    return np.stack([unit_quaternion(row) for row in held.reshape(-1, 7)]).reshape(held.shape).astype(held.dtype)


#: The arm's drawn joint positions lie within a turn either way, and its mount's poses hold unit quaternions: in the
#: default box of 1e4 a revolute joint turns through 1e4 rad, where float32 has no digits left for the angle's sine.
ARM = NodeCase(options={"boundary_bounds": {"joint_positions": (-7.0, 7.0), "mount_poses": (-1.0, 1.0)},
                        "bounds": {"joint_positions": (-7.0, 7.0), "frame_poses": (-1.0, 1.0)},
                        "constrain_boundary": lambda given: {**given, "mount_poses": unit_poses(given["mount_poses"])},
                        "constrain_state": lambda state: {**state, "frame_poses": unit_poses(state["frame_poses"])}})
MOTOR = NodeCase(options={"boundary_bounds": {"angle": (-7.0, 7.0), "velocity": (-100.0, 100.0),
                                              "mount_poses": (-1.0, 1.0)},
                          "bounds": {"angle": (-7.0, 7.0), "frame_poses": (-1.0, 1.0)},
                          "constrain_boundary": lambda given: {**given,
                                                               "mount_poses": unit_poses(given["mount_poses"])},
                          "constrain_state": lambda state: {**state, "frame_poses": unit_poses(state["frame_poses"])}})
#: The radius of the sphere about the rotor magnet of the first case that builds the dipole (A0's: 35 x 20 mm).
MAGNET_REACH = float(np.hypot(17.5e-3, 10e-3))


def outside_the_magnet(given):
    """A drawn mount and reading points as a reader can have them: quaternions of unit length, and each point outside
    the magnet, which sits on the mount's rotor frame (its second row): a point drawn inside is moved out along its
    direction from the magnet's centre (along x where it is the centre). Hypothesis shrinks a point onto the centre,
    where the field is singular, and in float32 its fifth power of a distance of 1e-15 m underflows."""
    held = {key: unit_poses(value) if key == "mount_poses" or key.startswith("points_") else value
            for key, value in given.items()}
    centre = np.asarray(held["mount_poses"], dtype=np.float64)[1, :3]
    for key, value in held.items():
        if not key.startswith("points_"):
            continue
        point = np.asarray(value, dtype=np.float64)
        offset = point[:3] - centre
        distance = float(np.linalg.norm(offset))
        if distance < 2 * MAGNET_REACH:
            direction = offset / distance if distance > 0 else np.array([1.0, 0.0, 0.0])
            point[:3] = centre + 2 * MAGNET_REACH * direction
            held[key] = point.astype(np.asarray(value).dtype)
    return held


#: The dipole's drawn mount lies within a metre of the origin, and its reading points outside the magnet.
DIPOLE = NodeCase(options={"boundary_bounds": {"mount_poses": (-1.0, 1.0)},
                           "constrain_boundary": outside_the_magnet})

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
    ("dynamics", "robot_arm", "0.1.0"): ARM,
    ("dynamics", "motor", "0.1.0"): MOTOR,
    ("dynamics", "labx.gantry", "0.1.0"): NodeCase(),
    ("field provider", "point_dipole_field", "0.1.0"): DIPOLE,
    ("interaction", "magnetic.force", "0.1.0"): NodeCase(),
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
