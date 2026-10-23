"""The canonical tree and its articulation; the maths' contracts (``jit``, ``grad``, ``vmap``, finite at zero rotation,
dtype-preserving); and loop closures, held as data and refused by every algorithm, naming them."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import multibody_cases as cases
from mime.core.multibody import ROOT, Body, ClosedChainRefused, ClosureKind, Frame, Inertial, InvalidDescription, \
    Joint, JointKind, LoopClosure, MultibodyTree, RigidTransform, bias_forces, compose, forward_dynamics, \
    forward_kinematics, frame_jacobian, frame_pose, frame_poses, gravity_forces, inverse, inverse_dynamics, \
    inverse_kinematics, mass_matrix, pose_error, rotation_vector, transform_point

GRAVITY = (0.0, 0.0, -9.81)


@pytest.fixture
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def chain(*joints: Joint, bodies: tuple[Body, ...] | None = None, **fields) -> MultibodyTree:
    names = {j.parent for j in joints} | {j.child for j in joints} | {"base"}
    held = bodies if bodies is not None else tuple(Body(n, Inertial(1.0, inertia=((0.1, 0, 0), (0, 0.1, 0),
                                                                                   (0, 0, 0.1)))) for n in sorted(names))
    return MultibodyTree("t", "base", held, joints, **fields)


def revolute(name: str, parent: str, child: str, **fields) -> Joint:
    return Joint(name, JointKind.REVOLUTE, parent, child, axis=(0.0, 0.0, 1.0), **fields)


# The tree -----------------------------------------------------------------------------------------------------------

def test_a_tree_holds_what_it_was_given_and_compares_by_value():
    tree = cases.branched()
    assert tree == cases.branched() and hash(tree) == hash(cases.branched())
    assert tree.frame_names == ("base", "upper", "fore", "carriage", "tool", "hand", "side_arm", "marker", "tip")
    assert tree.joint("slide").kind is JointKind.PRISMATIC
    assert tree.joint("elbow").axis == pytest.approx(np.array([0.2, 1.0, 0.1]) / np.linalg.norm([0.2, 1.0, 0.1]))
    assert tree.welded() == {"base": "base", "upper": "upper", "fore": "fore", "carriage": "carriage",
                             "tool": "carriage", "hand": "hand", "marker": "hand", "side_arm": "side_arm"}


def test_the_articulation_numbers_moving_joints_depth_first_and_welds_fixed_ones():
    articulation = cases.branched().articulation
    assert articulation.dof_names == ("shoulder", "elbow", "slide", "wrist", "side")
    assert articulation.parents == (ROOT, 0, 1, 2, 0)
    assert articulation.revolute == (True, True, False, True, True)
    assert articulation.frame_bodies == (ROOT, 0, 1, 2, 2, 3, 4, 3, 3)
    assert list(articulation.armatures) == [0.0, 0.0, 0.05, 0.01, 0.02]


def test_welded_inertials_combine_about_their_common_centre_of_mass():
    tree = cases.branched()
    carriage, tool = tree.body("carriage").inertial, tree.body("tool").inertial
    bracket = tree.joint("bracket").origin
    a = tree.articulation
    k = a.dof_names.index("slide")
    tool_centre = bracket.matrix() @ np.asarray(tool.centre_of_mass) + np.asarray(bracket.translation)
    mass = carriage.mass + tool.mass
    centre = (carriage.mass * np.asarray(carriage.centre_of_mass) + tool.mass * tool_centre) / mass
    assert a.masses[k] == pytest.approx(mass)
    assert a.centres_of_mass[k] == pytest.approx(centre)
    total = a.inertias[k]
    assert np.allclose(total, total.T) and np.all(np.linalg.eigvalsh(total) > 0)


@pytest.mark.parametrize(("build", "says"), [
    (lambda: chain(revolute("a", "base", "x"), revolute("a", "x", "y")), "joint name 'a' is used twice"),
    (lambda: chain(revolute("a", "base", "x"), revolute("b", "base", "x")), "child of two joints"),
    (lambda: chain(revolute("a", "x", "base")), "root 'base' is the child"),
    (lambda: chain(revolute("a", "base", "x"), revolute("b", "y", "z"), revolute("c", "z", "y")), "not connected"),
    (lambda: chain(revolute("a", "base", "nowhere"), bodies=(Body("base"),)), "not one of the bodies"),
    (lambda: chain(revolute("a", "base", "x"), frames=(Frame("f", "nowhere"),)), "fixed to 'nowhere'"),
    (lambda: chain(revolute("a", "base", "x"), frames=(Frame("x", "base"),)), "'x' is used twice"),
    (lambda: chain(revolute("a", "base", "x"), closures=(LoopClosure("c", ClosureKind.CONNECT, "x", "nowhere"),)),
     "names 'nowhere'"),
    (lambda: Joint("a", JointKind.REVOLUTE, "p", "c", axis=(0.0, 0.0, 0.0)), "zero axis"),
    (lambda: Joint("a", JointKind.REVOLUTE, "p", "c", damping=-1.0), "damping"),
    (lambda: Inertial(-1.0), "mass"),
    (lambda: Inertial(1.0, inertia=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))), "principal moments"),
    (lambda: Inertial(1.0, inertia=((1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))), "symmetric"),
    (lambda: RigidTransform(rotation=(0.0, 0.0, 0.0, 0.0)), "zero"),
])
def test_a_tree_that_is_not_one_is_refused_saying_why(build, says):
    with pytest.raises(InvalidDescription, match=says):
        build()


def test_a_tree_with_no_moving_joint_has_frames_and_no_degrees_of_freedom():
    tree = chain(Joint("weld", JointKind.FIXED, "base", "x", RigidTransform((1.0, 0.0, 0.0))))
    q = jnp.zeros(0)
    assert np.asarray(frame_pose(tree, q, "x"))[:3].tolist() == [1.0, 0.0, 0.0]
    assert mass_matrix(tree, q).shape == (0, 0) and inverse_dynamics(tree, q, q, q, GRAVITY).shape == (0,)
    assert frame_jacobian(tree, q, "x").shape == (6, 0) and forward_dynamics(tree, q, q, q, GRAVITY).shape == (0,)


# Poses --------------------------------------------------------------------------------------------------------------

def test_poses_compose_invert_and_measure_their_difference(x64):
    a = jnp.concatenate([jnp.asarray([0.1, -0.2, 0.3]), jnp.asarray(cases.turned((1.0, 2.0, 3.0), 0.7))])
    b = jnp.concatenate([jnp.asarray([-0.4, 0.5, 0.2]), jnp.asarray(cases.turned((0.0, 1.0, -1.0), -1.1))])
    point = jnp.asarray([0.3, 0.2, -0.1])
    assert np.allclose(transform_point(compose(a, b), point), transform_point(a, transform_point(b, point)))
    assert np.allclose(compose(a, inverse(a)), [0, 0, 0, 1, 0, 0, 0])
    assert np.allclose(pose_error(a, a), np.zeros(6))
    turn = jnp.asarray(cases.turned((0.0, 0.0, 1.0), 0.25))
    assert np.allclose(rotation_vector(turn), [0.0, 0.0, 0.25])
    assert np.allclose(rotation_vector(-turn), [0.0, 0.0, 0.25])  # -q is the same rotation
    single, point32 = a.astype(jnp.float32), point.astype(jnp.float32)
    for found in (compose(single, single), inverse(single), transform_point(single, point32),
                  pose_error(single, single), rotation_vector(single[3:]), rotation_vector(-single[3:])):
        assert found.dtype == jnp.float32


# The maths' contracts -----------------------------------------------------------------------------------------------

#: The branched tree's tip at zero joint positions, for the inverse kinematics run on every tree.
TARGET = np.asarray([0.66718252, -0.03994220, 0.24876900, 0.93298359, 0.00736444, 0.24080424, -0.26739616])


def algorithms(tree: MultibodyTree):
    """Every algorithm on ``tree``, each a function of the joint positions alone."""
    frame, target = "tip", jnp.asarray(TARGET)

    def solve(q):
        return inverse_kinematics(tree, frame, target.astype(q.dtype), q, position_tolerance=1e-3,
                                  rotation_tolerance=1e-3, iterations=5)

    return {
        "forward_kinematics": lambda q: forward_kinematics(tree, q),
        "frame_pose": lambda q: frame_pose(tree, q, frame),
        "frame_poses": lambda q: frame_poses(tree, q, ("upper", frame)),
        "frame_jacobian": lambda q: frame_jacobian(tree, q, frame),
        "mass_matrix": lambda q: mass_matrix(tree, q),
        "inverse_dynamics": lambda q: inverse_dynamics(tree, q, q, q, GRAVITY),
        "bias_forces": lambda q: bias_forces(tree, q, q, GRAVITY),
        "gravity_forces": lambda q: gravity_forces(tree, q, GRAVITY),
        "forward_dynamics": lambda q: forward_dynamics(tree, q, q, q, GRAVITY),
        "inverse_kinematics": lambda q: solve(q).q,
    }


ALGORITHMS = sorted(algorithms(cases.branched()))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_every_algorithm_keeps_the_dtype_of_the_joint_positions(x64, algorithm, dtype):
    """Under x64, float32 in gives float32 out: the tree's float64 constants are cast to the positions' dtype; and the
    float32 result is the float64 one to within float32's rounding of its scale (64 roundings: a few dozen operations,
    and a solve with a mass matrix whose condition number here is about 65)."""
    run = algorithms(cases.branched())[algorithm]
    q = jnp.asarray([0.3, -0.2, 0.04, 0.5, 0.1], dtype=dtype)
    found = run(q)
    assert found.dtype == dtype
    if dtype == jnp.float32:
        exact = np.asarray(run(q.astype(jnp.float64)))
        assert np.max(np.abs(np.asarray(found) - exact)) <= 64 * np.finfo(np.float32).eps * np.max(np.abs(exact))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_is_finite_with_finite_derivatives_at_zero_rotation(x64, algorithm):
    run = algorithms(cases.branched())[algorithm]
    q = jnp.zeros(5)
    value, gradient = jax.jit(jax.value_and_grad(lambda x: jnp.sum(run(x))))(q)
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(gradient)))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_compiles_and_maps_over_a_batch(x64, algorithm):
    run = algorithms(cases.branched())[algorithm]
    batch = jnp.asarray([[0.0, 0.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.04, 0.5, 0.1], [-1.0, 1.2, -0.03, 2.0, -0.7]])
    compiled = jax.jit(run)
    one_by_one = np.stack([np.asarray(compiled(q)) for q in batch])
    mapped = np.asarray(jax.jit(jax.vmap(run))(batch))
    assert np.max(np.abs(mapped - one_by_one)) <= 64 * np.finfo(np.float64).eps * np.max(np.abs(one_by_one))


def test_inverse_kinematics_is_differentiable_at_a_reached_target(x64):
    """At a reached target the rotation error is zero, where a rotation vector's closed form divides by zero."""
    tree = cases.planar()
    q = jnp.asarray([0.3, -0.2, 0.5])
    target = jax.jit(lambda x: frame_pose(tree, x, "link_3"))(q)

    def solved(t):
        return inverse_kinematics(tree, "link_3", t, q, position_tolerance=1e-9, rotation_tolerance=1e-9,
                                  iterations=2)

    joints, rotation_error = jax.jit(jax.jacfwd(lambda t: (solved(t).q, solved(t).rotation_error)))(target)
    assert np.all(np.isfinite(np.asarray(joints))) and np.all(np.isfinite(np.asarray(rotation_error)))


def test_a_mass_or_an_axis_can_be_differentiated_through_the_articulation(x64):
    articulation = cases.branched().articulation
    q = jnp.asarray([0.3, -0.2, 0.04, 0.5, 0.1])
    gradient = jax.jit(jax.grad(lambda a: jnp.sum(gravity_forces(a, q, GRAVITY))))(articulation)
    assert np.all(np.isfinite(np.asarray(gradient.masses))) and np.any(np.asarray(gradient.masses) != 0)


# Loop closures ------------------------------------------------------------------------------------------------------

def closed(tree: MultibodyTree) -> MultibodyTree:
    return dataclasses.replace(tree, closures=(LoopClosure("four_bar", ClosureKind.CONNECT, "tip", "side_arm"),))


def test_a_tree_holds_a_loop_closure_as_data():
    tree = closed(cases.branched())
    assert tree.closures == (LoopClosure("four_bar", ClosureKind.CONNECT, "tip", "side_arm"),)
    assert tree.articulation.closures == ("four_bar",)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_refuses_a_tree_with_a_closure_naming_it(algorithm):
    run = algorithms(cases.branched())[algorithm]
    run(jnp.zeros(5))  # the control: the open tree is accepted
    tree = closed(cases.branched())
    planted = algorithms(tree)[algorithm]
    with pytest.raises(ClosedChainRefused, match="'four_bar'") as refused:
        planted(jnp.zeros(5))
    assert refused.value.algorithm == algorithm
    assert refused.value.closures == ("four_bar",)
    assert "out of scope" in str(refused.value) and "register_dynamics" in str(refused.value)
    with pytest.raises(ClosedChainRefused, match="'four_bar'"):
        mass_matrix(tree.articulation, jnp.zeros(5))


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.3, math.pi])
def test_a_rotation_vector_is_finite_with_finite_derivatives_from_rest_to_a_half_turn(x64, angle):
    turned = jnp.asarray(cases.turned((0.0, 1.0, 0.0), angle))
    assert np.allclose(np.asarray(rotation_vector(turned)), [0.0, angle, 0.0], rtol=1e-14, atol=0.0)
    assert np.all(np.isfinite(np.asarray(jax.jacobian(rotation_vector)(turned))))
