"""The trees the rigid-body library is tested on, and joint states a real arm can have on each.

- ``planar()``: the three-link planar arm (``descriptions/three_link_planar.urdf``), whose forward kinematics has a
  closed form;
- ``ar4()``: A0's arm (``descriptions/ar4.urdf``, as the 0.2 line ships it), a six-joint serial chain whose visual
  links are welded on by fixed joints;
- ``coupled_planar()``: two strongly coupled links built in code, for the dynamic arm's stops and friction;
- ``branched()``: a tree built in code that the two files do not exercise: a branch, a prismatic joint, a fixed joint
  welding a body with mass between two moving ones, tilted axes and origins, products of inertia, armature, and an
  extra frame.

Joint states are drawn inside each joint's limits, at speeds and accelerations of the arm's own scale, with components
far below that scale set to zero rather than left subnormal.
"""

import math
import warnings
from pathlib import Path

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mime.core.multibody import Body, Frame, Inertial, Joint, JointKind, JointLimits, MultibodyTree, RigidTransform, \
    read_urdf

DESCRIPTIONS = Path(__file__).parent / "descriptions"
PLANAR_URDF, AR4_URDF = DESCRIPTIONS / "three_link_planar.urdf", DESCRIPTIONS / "ar4.urdf"
#: The planar arm's link lengths, from its file.
PLANAR_LENGTHS = (1.0, 1.0, 0.5)
EPS = float(np.finfo(np.float64).eps)


def planar() -> MultibodyTree:
    return read_urdf(PLANAR_URDF).tree


def ar4() -> MultibodyTree:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_urdf(AR4_URDF).tree


def turned(axis: tuple[float, float, float], angle: float) -> tuple[float, float, float, float]:
    n = math.sqrt(sum(a * a for a in axis))
    s = math.sin(angle / 2) / n
    return (math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def box(mass: float, x: float, y: float, z: float, centre: tuple[float, float, float],
        rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)) -> Inertial:
    """A uniform box's inertial, its principal axes turned by ``rotation`` in the body's frame."""
    principal = mass / 12 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])
    r = RigidTransform(rotation=rotation).matrix()
    tensor = r @ principal @ r.T
    return Inertial(mass, centre, tuple(tuple(float(v) for v in row) for row in tensor))


def branched() -> MultibodyTree:
    tilt = turned((1.0, 2.0, -0.5), 0.4)
    bodies = (
        Body("base", box(5.0, 0.2, 0.2, 0.1, (0.0, 0.0, 0.05))),
        Body("upper", box(1.2, 0.3, 0.05, 0.04, (0.15, 0.01, 0.0), turned((0.0, 0.0, 1.0), 0.1))),
        Body("fore", box(0.8, 0.25, 0.04, 0.03, (0.12, 0.0, 0.01), tilt)),
        Body("carriage", box(0.3, 0.06, 0.05, 0.04, (0.02, 0.0, 0.0))),
        Body("tool", box(0.2, 0.04, 0.03, 0.08, (0.0, 0.01, 0.03), turned((0.0, 1.0, 0.0), 0.7))),
        Body("hand", box(0.1, 0.05, 0.02, 0.02, (0.03, 0.0, 0.0))),
        Body("side_arm", box(0.5, 0.2, 0.03, 0.03, (0.0, 0.1, 0.0))),
        Body("marker"),
    )
    joints = (
        Joint("shoulder", JointKind.REVOLUTE, "base", "upper", RigidTransform((0.0, 0.0, 0.3), turned((1, 0, 0), 0.2)),
              (0.0, 0.0, 1.0), JointLimits(-2.5, 2.5, velocity=2.0)),
        Joint("elbow", JointKind.REVOLUTE, "upper", "fore", RigidTransform((0.3, 0.0, 0.02), tilt), (0.2, 1.0, 0.1),
              JointLimits(-2.0, 2.0, velocity=2.0), damping=0.1),
        Joint("side", JointKind.REVOLUTE, "upper", "side_arm", RigidTransform((0.1, 0.1, 0.0)), (1.0, 0.0, 0.0),
              JointLimits(-1.5, 1.5, velocity=2.0), armature=0.02),
        Joint("slide", JointKind.PRISMATIC, "fore", "carriage", RigidTransform((0.25, 0.0, 0.0)), (1.0, 0.0, 0.3),
              JointLimits(-0.05, 0.1, velocity=0.2), armature=0.05, friction=0.3),
        Joint("bracket", JointKind.FIXED, "carriage", "tool",
              RigidTransform((0.05, 0.02, -0.01), turned((0.3, 0.0, 1.0), -0.9))),
        Joint("wrist", JointKind.REVOLUTE, "tool", "hand", RigidTransform((0.0, 0.0, 0.08)), (0.0, 1.0, 0.0),
              JointLimits(-3.0, 3.0, velocity=3.0), armature=0.01),
        Joint("marker_mount", JointKind.FIXED, "hand", "marker", RigidTransform((0.06, 0.0, 0.0))),
    )
    frames = (Frame("tip", "hand", RigidTransform((0.08, 0.01, 0.0), turned((0.0, 0.0, 1.0), 0.3))),)
    return MultibodyTree("branched", "base", bodies, joints, frames)


def coupled_planar(limit: float | None = None) -> MultibodyTree:
    """Two links turning about z, the second's mass and length large beside the first's own inertia: a strongly
    coupled tree (normalised coupling 0.86 at an elbow of 0.5 rad), each joint bounded at ``+-limit`` where given."""
    m1, m2, l1, lc1, lc2, i1, i2 = 2.0, 1.5, 1.0, 0.45, 0.6, 0.17, 0.11
    bound = JointLimits(-limit, limit) if limit is not None else JointLimits()
    b1 = Body("l1", Inertial(m1, (lc1, 0, 0), ((0.01, 0, 0), (0, 0.02, 0), (0, 0, i1))))
    b2 = Body("l2", Inertial(m2, (lc2, 0, 0), ((0.01, 0, 0), (0, 0.03, 0), (0, 0, i2))))
    j1 = Joint("j1", JointKind.REVOLUTE, "base", "l1", axis=(0, 0, 1), limits=bound)
    j2 = Joint("j2", JointKind.REVOLUTE, "l1", "l2", RigidTransform((l1, 0, 0)), axis=(0, 0, 1), limits=bound)
    return MultibodyTree("coupled_planar", "base", (Body("base"), b1, b2), (j1, j2))


def _component(scale: float):
    return st.floats(-scale, scale, allow_nan=False).map(lambda x: 0.0 if abs(x) < 1e-6 * scale else x)


def _within(lower: float, upper: float):
    lower, upper = max(lower, -math.pi), min(upper, math.pi)
    scale = max(abs(lower), abs(upper))
    return st.floats(lower, upper, allow_nan=False).map(lambda x: 0.0 if abs(x) < 1e-6 * scale else x)


def positions(tree: MultibodyTree):
    """Joint positions inside every joint's limits (an unlimited joint within a turn either way)."""
    a = tree.articulation
    return st.tuples(*(_within(float(lo), float(hi)) for lo, hi in zip(a.lower, a.upper, strict=True))).map(
        lambda values: np.asarray(values, dtype=np.float64))


def states(tree: MultibodyTree, speed: float, acceleration: float):
    """``(q, qd, qdd)``: positions inside the limits, and velocities and accelerations up to the given scales."""
    dofs = tree.articulation.dofs
    return st.tuples(positions(tree), arrays(np.float64, (dofs,), elements=_component(speed)),
                     arrays(np.float64, (dofs,), elements=_component(acceleration)))


#: Each tree with the speed and acceleration scales its states are drawn at, and the size of its quantities: its
#: reach (m), from the joint origins and frame offsets, which scales positions and Jacobians.
TREES = {"planar": (planar, 10.0, 20.0, 2.5), "ar4": (ar4, 1.0, 5.0, 0.75), "branched": (branched, 2.0, 10.0, 1.0)}


def joint_commands(node, values, *, dynamic: bool = True) -> dict:
    """A value per moving joint as a chain node takes it, typed by the joint's kind: a dynamic node's revolute joints'
    torques under ``joint_torques`` and its prismatic joints' forces under ``joint_forces``; a kinematic node's angles
    under ``joint_positions`` and its slides' displacements under ``joint_displacements``."""
    revolute, prismatic = node.kinds()
    names = ("joint_torques", "joint_forces") if dynamic else ("joint_positions", "joint_displacements")
    return {name: values[np.asarray(indices)] for name, indices in zip(names, (revolute, prismatic), strict=True)
            if indices}
