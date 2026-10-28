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

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import AR4, AR4_BASE, AR4_HOME, MAGNET_MOMENT, experiment, gantry_magnet, kinematic_rpm
from hypothesis import given, settings
from hypothesis import strategies as st
from pxr import Usd, UsdGeom

import mime.build._plan as plan_module
import multibody_cases as cases
from mime.build import Lag
from mime.build._maddening import node_state, run_scan
from mime.core import quaternion
from mime.core.multibody import compose, forward_kinematics, read_urdf
from mime.drives import Constant
from mime.dynamics import Motor, RobotArm
from mime.nodes.actuation import MountedNode
from mime.scene import STAGE_FILE, Z, asset
from mime.things import Apparatus
from mime.units import VACUUM_PERMEABILITY, m, mm, ms, rad, rad_per_s

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


@jax.jit
def tree_frames(joints):
    """Every frame of the tree at ``joints``, placed at the arm's base: compiled, since run eagerly it is most of a
    draw's time."""
    base = jnp.asarray((*AR4_BASE, 1.0, 0.0, 0.0, 0.0), dtype=joints.dtype)
    return jax.vmap(compose, in_axes=(None, 0))(base, forward_kinematics(TREE, joints))


def expected_frames(joints) -> np.ndarray:
    return np.asarray(tree_frames(jnp.asarray(joints)))


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


# -- a mount's composition: placements that are not the identity ----------------------------------------------------

#: The motor sits 10, 20 and 30 mm from the flange's origin, turned by a unit quaternion on no axis, and the magnet
#: 5 and 7 mm from the rotor's origin: no placement is zero or aligned, so a composition written the other way round
#: puts the pose far from where the right one does (0.1 m and a large angle on the AR4's home pose).
MOTOR_AT = np.array([0.010, 0.020, 0.030])
MOTOR_TURN = np.array([0.8, 0.3, -0.4, 0.3]) / math.sqrt(0.98)
MAGNET_AT = np.array([0.005, 0.0, 0.007])
MOUNTS = {"motor_at": mm((10.0, 20.0, 30.0)), "motor_orientation": tuple(float(c) for c in MOTOR_TURN),
          "magnet_at": mm((5.0, 0.0, 7.0))}
PLACEMENT = np.concatenate([MOTOR_AT, MOTOR_TURN])
PART_OFFSET = np.concatenate([MAGNET_AT, [1.0, 0.0, 0.0, 0.0]])
#: The rotor's speed and the steps taken: 0.1 rad. One step, so that the body is where it started when the field is
#: read: the field's reader takes its pose at the start of the step.
SPEED, STEPS = 100.0, 1
BODY_START = np.array([0.0, 0.0, -1e-3])
SLOT_FIELD = "flux_density_"
K = VACUUM_PERMEABILITY / (4 * np.pi)


@pytest.fixture(scope="module")
def mounted():
    return kinematic_rpm("mounted_rpm", **MOUNTS).build()


@jax.jit
def mounted_chain(joints, angle):
    """The motor's base, its rotor and the magnet, by ``flange o placement o rotor-turn o part-offset``: compiled, as
    ``tree_frames`` is."""
    flange = tree_frames(joints)[TREE.frame_names.index("link_6")]
    base = compose(flange, jnp.asarray(PLACEMENT))
    turn = jnp.stack([0.0 * angle, 0.0 * angle, 0.0 * angle, jnp.cos(angle / 2), 0.0 * angle, 0.0 * angle,
                      jnp.sin(angle / 2)])
    rotor = compose(base, turn)
    return jnp.stack([base, rotor, compose(rotor, jnp.asarray(PART_OFFSET))])


def pose_of(joints, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, rotor, magnet = np.asarray(mounted_chain(jnp.asarray(joints), jnp.asarray(angle)))
    return base, rotor, magnet


def dipole_field(moment: np.ndarray, centre: np.ndarray, point: np.ndarray) -> np.ndarray:
    r = point - centre
    distance = np.linalg.norm(r)
    return K * (3 * np.dot(moment, r) * r / distance**2 - moment) / distance**3


def mount_misfits(state, joints, angle: float) -> list[str]:
    """What differs between a built chain's state and the pose of ``pose_of``: the motor's frames, the magnet's centre
    and orientation, and the field the magnet gives at the body, whose moment is the declared one along the rotor's x
    turned by that orientation."""
    base, rotor, magnet = pose_of(joints, angle)
    found = []
    if not poses_agree(np.asarray(state["rpm.motor"]["frame_poses"]), np.stack([base, rotor])):
        found.append("the motor's frames")
    shown = np.concatenate([state["rpm.magnetic"]["centre"], state["rpm.magnetic"]["orientation"]])
    if not poses_agree(shown, magnet):
        found.append("the magnet's pose")
    return found


def field_misfit(state, magnet: np.ndarray) -> float:
    """How far, relative to its size, the dipole's field at the body is from that of the moment ``MAGNET_MOMENT`` along
    the rotor's x, turned by the magnet's orientation, at the body's start."""
    body = BODY_START
    moment = np.asarray(quaternion.rotate(jnp.asarray(magnet[3:]), jnp.asarray((MAGNET_MOMENT, 0.0, 0.0))))
    expected = dipole_field(moment, magnet[:3], body)
    slot = next(key for key in state["rpm.magnetic"] if key.startswith(SLOT_FIELD))
    return float(np.max(np.abs(np.asarray(state["rpm.magnetic"][slot]) - expected)) / np.max(np.abs(expected)))


@PROPERTY
@given(cases.positions(TREE), st.sampled_from([0.0, SPEED]))
def test_a_mounted_segment_and_part_sit_at_the_mount_then_the_placement(mounted, joints, speed):
    """The motor 10, 20 and 30 mm off the flange and turned, the magnet 5 and 7 mm off the rotor, at two rotor angles
    (0 and 0.1 rad): each pose is the mount's, then the placement's, then the rotor's turn, then the part's offset."""
    state = run(mounted, joints, speed, STEPS)
    angle = speed * STEPS * DT
    assert mount_misfits(state, joints, angle) == []
    assert field_misfit(state, pose_of(joints, angle)[2]) <= 1e-9


def test_the_start_poses_compose_in_the_same_order(mounted):
    """Before any step, the nodes' initial state and the plan's initial poses are the same composition."""
    mounted.graph.reset_state()
    state = {"rpm.motor": node_state(mounted.graph, "rpm.motor"), "rpm.magnetic": node_state(mounted.graph,
                                                                                            "rpm.magnetic")}
    assert mount_misfits(state, np.asarray(AR4_HOME), 0.0) == []


def swap_composition_in_the_nodes(patch: pytest.MonkeyPatch) -> None:
    """Plant: a mount composed the other way, ``placement o mount``, in the nodes' step and their initial state."""
    def base(self, p, given):
        placement = jnp.asarray(p["placement"])
        index = self.constants()["mount_index"]
        return placement if index < 0 else compose(placement, given["mount_poses"][index])

    def initial_base(self):
        held = self.constants()
        placement = jnp.asarray(held["placement"])
        return placement if held["mount_index"] < 0 else compose(placement, jnp.asarray(held["initial_mount_pose"]))

    patch.setattr(MountedNode, "base", base)
    patch.setattr(MountedNode, "initial_base", initial_base)


def test_control_a_mount_composed_the_other_way_in_the_nodes_fails_on_every_draw():
    with pytest.MonkeyPatch.context() as patch:
        swap_composition_in_the_nodes(patch)
        planted = kinematic_rpm("swapped_nodes_rpm", **MOUNTS).build()

        @PROPERTY
        @given(cases.positions(TREE), st.sampled_from([0.0, SPEED]))
        def every_draw_is_caught(joints, speed):
            state = run(planted, joints, speed, STEPS)
            assert mount_misfits(state, joints, speed * STEPS * DT) != []

        every_draw_is_caught()
        planted.graph.reset_state()
        state = {"rpm.motor": node_state(planted.graph, "rpm.motor"),
                 "rpm.magnetic": node_state(planted.graph, "rpm.magnetic")}
        assert mount_misfits(state, np.asarray(AR4_HOME), 0.0) != []


def test_control_a_plan_that_composes_the_start_poses_the_other_way_fails():
    """Plant: the plan's composition written ``b o a`` for ``a o b``, which hands every mounted node the wrong pose it
    starts from."""
    composed = plan_module.composed
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plan_module, "composed", lambda outer, inner: composed(inner, outer))
        planted = kinematic_rpm("swapped_plan_rpm", **MOUNTS).build()
    state = {"rpm.motor": node_state(planted.graph, "rpm.motor"), "rpm.magnetic": node_state(planted.graph,
                                                                                            "rpm.magnetic")}
    assert mount_misfits(state, np.asarray(AR4_HOME), 0.0) != []


def test_the_stage_stands_the_part_where_the_graph_starts_it(mounted, tmp_path):
    """The magnet's prim, below the motor's rotor frame, has the world pose the graph's magnet starts at."""
    mounted.save(tmp_path)
    stage = Usd.Stage.Open(str(tmp_path / STAGE_FILE))
    world = UsdGeom.Xformable(stage.GetPrimAtPath("/World/rpm/motor/rotor/rotor_magnet")).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    _, _, magnet = pose_of(np.asarray(AR4_HOME), 0.0)
    assert max(abs(a - b) for a, b in zip(world.ExtractTranslation(), magnet[:3], strict=True)) <= 1e-12
    for axis in np.eye(3):
        found = np.asarray(world.TransformDir(tuple(float(c) for c in axis)))
        expected = np.asarray(quaternion.rotate(jnp.asarray(magnet[3:]), jnp.asarray(axis)))
        assert np.max(np.abs(found - expected)) <= 1e-12


# -- the servo, the angle mode and the limits ------------------------------------------------------------------------

#: The joint positions the servo test commands: inside every limit of the AR4, away from the home pose.
SERVO_COMMAND = np.array([0.3, 0.6, -0.4, 1.0, 1.3, 0.5])
SERVO_TIME_CONSTANT, ANGLE = 10e-3, 1.2


@pytest.fixture(scope="module")
def servoed():
    return kinematic_rpm("servoed_rpm", servo=ms(SERVO_TIME_CONSTANT * 1e3), prescribed="angle", speed=ANGLE).build()


def run_servoed(built, steps: int):
    built.graph.reset_state()
    params = {"nodes": {"drive:rpm.arm.joint_positions": {"value": jnp.asarray(SERVO_COMMAND)},
                        "drive:rpm.motor.angle": {"value": jnp.asarray(ANGLE)}}}
    return run_scan(built.graph, steps, params)


@pytest.mark.parametrize("steps", [1, 5])
def test_a_servoed_arm_moves_by_one_minus_the_exponential_of_the_steps_over_the_time_constant(servoed, steps):
    """A first-order lag, exact over the step for a command held through it: from the home pose towards the command,
    ``1 - exp(-t / tau)`` of the way after ``t``. The planted sign of the exponent, ``exp(+t / tau)``, moves it the other
    way."""
    state = run_servoed(servoed, steps)
    fraction = 1.0 - math.exp(-steps * DT / SERVO_TIME_CONSTANT)
    expected = np.asarray(AR4_HOME) + fraction * (SERVO_COMMAND - np.asarray(AR4_HOME))
    assert np.allclose(np.asarray(state["rpm.arm"]["joint_positions"]), expected, rtol=0.0, atol=64 * EPS)


def test_a_motor_whose_angle_is_prescribed_goes_to_it_and_its_rotor_is_turned_that_far(servoed):
    """The command is the angle, not a speed: after any step the rotor has turned by it about the flange's z."""
    state = run_servoed(servoed, 2)
    assert float(state["rpm.motor"]["angle"]) == ANGLE
    flange_pose = np.asarray(state["rpm.motor"]["frame_poses"])
    stator, rotor = flange_pose
    turn = np.asarray(quaternion.multiply(quaternion.conjugate(jnp.asarray(stator[3:])), jnp.asarray(rotor[3:])))
    assert np.allclose(turn, (math.cos(ANGLE / 2), 0.0, 0.0, math.sin(ANGLE / 2)), rtol=0.0, atol=64 * EPS)


def test_a_command_beyond_a_limit_is_held_at_it(built):
    """Joint 1 commanded past its upper limit stands at the limit, whatever else is commanded."""
    joints = np.array([5.0, 0.2, -0.7, 1.4, 1.5, 0.3])
    state = run(built, joints)
    assert float(state["rpm.arm"]["joint_positions"][0]) == pytest.approx(2.792526803190927, abs=0.0)


def test_the_build_report_lists_the_limits_each_records_states(built):
    """The kinematic arm and motor state that a reaction on what they carry is not applied; the report says so, so
    that a reader of it is told."""
    text = built.report.text()
    assert "limits of rpm.arm (stated by its record)" in text and "limits of rpm.motor (stated by its record)" in text
    assert "the reaction of a field on a magnet mounted on it" in text
    assert "its reaction torque on the link it is mounted on is not delivered" in text


def test_the_build_lists_a_chains_segments_in_chain_order_not_by_name():
    """Named so that their alphabetical order is the reverse of the chain's, the motor on the arm's flange is still
    listed after the arm: the plan's order keeps a chain's segments where the chain has them."""
    exp = experiment("ordered")
    exp.add_apparatus(Apparatus("bench", chain={
        "zarm": RobotArm.Kinematic(description=asset(AR4), initial_positions=rad(AR4_HOME), at=m(AR4_BASE)),
        "amotor": Motor.Kinematic(axis=Z, prescribed="velocity", initial_angle=rad(0.0), mount="link_6")},
        commands={"zarm.joint_positions": Constant(rad(AR4_HOME)), "amotor.velocity": Constant(rad_per_s(1.0))}))
    owners = [entry.owner for entry in exp.plan().entries if entry.owner.startswith("bench.")]
    assert owners == ["bench.zarm", "bench.amotor"]


def test_a_magnet_on_the_labs_gantry_sits_on_its_carriage():
    """The gantry 10 cm above the origin, its carriage commanded 2 cm along x: the magnet 3 and 5 mm off the carriage's
    origin is there, on a segment that is no URDF's."""
    built = gantry_magnet(magnet_at=mm((3.0, 0.0, 5.0))).build()
    built.graph.reset_state()
    state = run_scan(built.graph, 2)
    assert poses_agree(np.asarray(state["stage.gantry"]["frame_poses"])[1], (0.02, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0))
    assert np.allclose(np.asarray(state["stage.magnetic"]["centre"]), (0.023, 0.0, 0.105), rtol=0.0, atol=1e-15)


def test_the_x64_fixture_holds():
    assert jax.config.jax_enable_x64
