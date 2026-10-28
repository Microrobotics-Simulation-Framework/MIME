"""An apparatus chain built from its declaration, against ``mime.core.multibody`` (float64, per module): A0's AR4 arm,
a motor on its flange and a magnet on the rotor (``chain_cases.kinematic_rpm``).

- **The chain through the build**: over joint angles drawn within the AR4's limits, commanded through the arm's drive,
  every frame pose the arm publishes is its base's pose composed with the tree's forward kinematics, and the motor's
  base is its mount frame's pose. The control: the same build with the motor planted 1 mm off its mount fails on
  every draw.
- **The magnet follows the chain**: over steps of the motor turning, the provider's magnet is at the rotor's frame
  composed with its mount offset, and turns with it, with no edge written by hand: the edges carrying the mounts are
  the build's own, and lag accounting reads them this step.
- **The lab's gantry**, a segment with no URDF: a magnet on its carriage sits where the carriage is, so mounts and the
  field downstream work unchanged on it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import AR4, AR4_BASE, gantry_magnet, kinematic_rpm
from hypothesis import given, settings
from hypothesis import strategies as st

import multibody_cases as cases
from mime.build import Lag
from mime.build._maddening import run_scan
from mime.core import quaternion
from mime.core.multibody import compose, forward_kinematics, read_urdf
from mime.units import mm

EPS = float(np.finfo(np.float64).eps)
#: How many float64 roundings of its scale each pose allows: both sides run the same library, through different
#: compositions of a few dozen operations.
ROUNDINGS = 64
PROPERTY = settings(deadline=None, max_examples=100)
DT = 1e-3
TREE = read_urdf(AR4).tree
#: The AR4's reach, the scale its frames' positions are judged on.
REACH = 0.75
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def built():
    return kinematic_rpm().build()


@pytest.fixture(scope="module")
def planted():
    return kinematic_rpm("planted_rpm", motor_at=mm((0.0, 0.0, 1.0))).build()


def run(built, joints, speed=0.0, steps=1):
    graph = built.graph
    graph.reset_state()
    params = {"nodes": {"drive:rpm.arm.joint_positions": {"value": jnp.asarray(joints)},
                        "drive:rpm.motor.velocity": {"value": jnp.asarray(speed)}}}
    return run_scan(graph, steps, params)


def expected_frames(joints) -> np.ndarray:
    base = jnp.asarray((*AR4_BASE, 1.0, 0.0, 0.0, 0.0))
    local = forward_kinematics(TREE, jnp.asarray(joints))
    return np.stack([np.asarray(compose(base, row)) for row in local])


def poses_agree(found, expected) -> bool:
    found, expected = np.asarray(found, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    positions = np.abs(found[..., :3] - expected[..., :3]) <= ROUNDINGS * EPS * REACH
    turns = np.abs(found[..., 3:] - expected[..., 3:]) <= ROUNDINGS * EPS
    return bool(np.all(positions) and np.all(turns))


def chain_agrees(state, joints) -> bool:
    frames = expected_frames(joints)
    flange = frames[TREE.frame_names.index("link_6")]
    motor = np.asarray(state["rpm.motor"]["frame_poses"])
    return poses_agree(state["rpm.arm"]["frame_poses"], frames) and poses_agree(motor[0], flange)


@PROPERTY
@given(cases.positions(TREE))
def test_the_published_frame_poses_are_the_trees_forward_kinematics(built, joints):
    assert chain_agrees(run(built, joints), joints)


@PROPERTY
@given(cases.positions(TREE))
def test_control_a_motor_planted_off_its_mount_fails_on_every_draw(planted, joints):
    assert not chain_agrees(run(planted, joints), joints)


def test_the_magnet_follows_the_rotor_with_no_hand_written_edge(built):
    """Five steps at 6 rad/s: the magnet's centre is the rotor's origin (its mount offset is zero), and its moment's
    direction is the rotor's x axis, which has turned by the speed times the time about the flange's z."""
    joints = np.array([0.1, 0.2, -0.7, 1.4, 1.5, 0.3])
    speed, steps = 6.0, 5
    state = run(built, joints, speed, steps)
    rotor = np.asarray(state["rpm.motor"]["frame_poses"])[1]
    assert poses_agree(np.concatenate([state["rpm.magnetic"]["centre"], state["rpm.magnetic"]["orientation"]]),
                       rotor)
    flange = expected_frames(joints)[TREE.frame_names.index("link_6")]
    angle = speed * steps * DT
    turned = np.asarray(quaternion.multiply(jnp.asarray(flange[3:]),
                                            jnp.asarray((np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)))))
    assert poses_agree(rotor, np.concatenate([flange[:3], turned]))
    mounts = {(edge.source, edge.target): edge for edge in built.report.lags if edge.target[1] == "mount_poses"}
    assert set(mounts) == {(("rpm.arm", "frame_poses"), ("rpm.motor", "mount_poses")),
                           (("rpm.motor", "frame_poses"), ("rpm.magnetic", "mount_poses"))}
    assert all(edge.lag is Lag.THIS_STEP for edge in mounts.values())


def test_a_magnet_on_the_labs_gantry_sits_on_its_carriage():
    """The gantry 10 cm above the origin, its carriage commanded 2 cm along x: the magnet there."""
    built = gantry_magnet().build()
    built.graph.reset_state()
    state = run_scan(built.graph, 2)
    assert poses_agree(np.asarray(state["stage.gantry"]["frame_poses"])[1], (0.02, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0))
    assert np.allclose(np.asarray(state["stage.magnetic"]["centre"]), (0.02, 0.0, 0.1), rtol=0.0, atol=1e-15)


def test_the_x64_fixture_holds():
    assert jax.config.jax_enable_x64
