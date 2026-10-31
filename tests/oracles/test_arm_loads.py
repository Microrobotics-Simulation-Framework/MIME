"""The dynamic arm's loads and its weight through a turned base, against their closed forms (float64, per module):
the AR4's node placed at a drawn offset and turned by a drawn rotation (0.3 to 2.8 rad about an axis at least 26
degrees off the vertical), one step of 1 us from rest at a drawn pose. Its acceleration under its weight is
``-M^-1 g(q; R^T g_world)``, and a drawn wrench on a drawn frame (world axes, about the frame's origin) changes it by
``M^-1 J^T R^T W`` (``J`` the frame's geometric Jacobian in the base's axes, by forward kinematics, not the recursion
the node's loads take). The controls, each failing on every draw a thousand times above the oracle's rounding: the
weight turned the wrong way into the base (``R g``), the loads not turned, and the loads' sign flipped."""

import math
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import AR4_HOME
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import multibody_cases as cases
from mime.core import quaternion
from mime.core.multibody import frame_forces, frame_jacobian, gravity_forces, mass_matrix
from mime.nodes.actuation import DynamicArmNode

PROPERTY = settings(deadline=None, max_examples=100)
EPS = float(np.finfo(np.float64).eps)
#: An oracle's tolerance, in units in the last place of its scale.
ROUNDING = 64


# -- its loads and its weight, through a turned base ------------------------------------------------------------------

LOAD_STEP = 1e-6
AR4_TREE = cases.ar4()
#: The AR4's mass matrix, gravity torques and a frame's Jacobian (the frame's name static), compiled once: the
#: strategy and the closed forms call them per draw.
AR4_INERTIA = jax.jit(lambda q: mass_matrix(AR4_TREE, q))
AR4_WEIGHT = jax.jit(lambda q, g: gravity_forces(AR4_TREE, q, g))
AR4_JACOBIAN = jax.jit(lambda q, frame: frame_jacobian(AR4_TREE, q, frame), static_argnums=1)


class WeightTurnedTheWrongWay(DynamicArmNode):
    """The plant: the world's gravity turned into the base by the base's rotation, not its inverse."""

    def weight(self, p, base):
        return quaternion.rotate(base[3:], jnp.asarray(p["gravity"], dtype=base.dtype))


class LoadsNotTurned(DynamicArmNode):
    """The plant: the loads taken in the base's axes as given, in the world's."""

    def loads(self, articulation, q, base, wrenches):
        return frame_forces(articulation, q, wrenches)


class LoadsReversed(DynamicArmNode):
    """The plant: the loads' sign flipped where the node gathers them."""

    def loads(self, articulation, q, base, wrenches):
        return -super().loads(articulation, q, base, wrenches)


@cache
def turned_arm(kind=DynamicArmNode) -> DynamicArmNode:
    """The AR4's node with no stops: a drawn pose may lie on a limit, where a stop would hold the joint its weight
    pushes into it, and the closed forms are the free arm's."""
    a = AR4_TREE.articulation
    described = {**DynamicArmNode.described(a, [0.0] * a.dofs, [0.0] * a.dofs), "bounded": (), "lower": (),
                 "upper": ()}
    return kind("arm", LOAD_STEP, **described, placement=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                gravity=(0.0, 0.0, -9.80665), initial_joint_positions=AR4_HOME)


@cache
def turned_start(kind=DynamicArmNode) -> dict:
    """Its initial state, once: each draw replaces its joints' positions and velocities."""
    return dict(turned_arm(kind).initial_state())


@st.composite
def turned_loads(draw):
    """A base placed within half a metre of the origin and turned 0.3 to 2.8 rad about an axis at least 26 degrees off
    the vertical (so that the weight turned either way differs by at least a tenth of it), a pose inside the AR4's
    limits away from its wrist's singularity (the last link's Jacobian's smallest singular value a twentieth of its
    largest or more), a frame on that link, and a wrench on it of up to a newton and a tenth of a newton metre across
    the turn's axis, each with at most as much again along it (a vector along the axis is the same turned or not). The
    axis and the wrench are built in place rather than filtered, so that the only draw refused is the singular pose."""
    unit = st.floats(-1.0, 1.0, allow_nan=False)
    height, bearing = draw(st.floats(-0.9, 0.9)), draw(st.floats(0.0, 2.0 * math.pi))
    axis = np.array([math.sqrt(1.0 - height**2) * math.cos(bearing), math.sqrt(1.0 - height**2) * math.sin(bearing),
                     height])
    across = np.cross(axis, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)
    beside = np.cross(axis, across)

    def vector(scale: float) -> np.ndarray:
        size, turn, along = draw(st.floats(0.1, 1.0)), draw(st.floats(0.0, 2.0 * math.pi)), draw(unit)
        return scale * size * (math.cos(turn) * across + math.sin(turn) * beside + along * axis)

    angle = draw(st.floats(0.3, 2.8))
    offset = 0.5 * np.asarray(draw(st.tuples(unit, unit, unit)))
    placement = np.concatenate([offset, [math.cos(angle / 2)], math.sin(angle / 2) * axis])
    q = draw(cases.positions(AR4_TREE))
    a = AR4_TREE.articulation
    frame = draw(st.sampled_from([k for k, body in enumerate(a.frame_bodies) if body == a.dofs - 1]))
    jacobian = np.linalg.svd(np.asarray(AR4_JACOBIAN(jnp.asarray(q), a.frame_names[frame])), compute_uv=False)
    assume(jacobian[-1] >= 0.05 * jacobian[0])
    return placement, q, frame, np.concatenate([vector(1.0), vector(0.1)])


def turned_misfits(drawn, kind=DynamicArmNode) -> tuple[float, float, float]:
    """The departures, each relative to its expected value's largest component, of the weight's acceleration and of
    the load's change of it from their closed forms; and the rounding of the solves either takes, ``ROUNDING`` units
    in the last place times the mass matrix's condition number."""
    placement, q, frame, wrench = drawn
    node = turned_arm(kind)
    a = AR4_TREE.articulation
    params = {"placement": jnp.asarray(placement), "gravity": jnp.asarray((0.0, 0.0, -9.80665))}
    state = {**turned_start(kind), "joint_positions": jnp.asarray(q), "joint_velocities": jnp.zeros(a.dofs)}
    loads = np.zeros((len(a.frame_names), 6))
    loads[frame] = wrench
    rest = np.asarray(node.update(state, {}, LOAD_STEP, params=params)["joint_velocities"]) / LOAD_STEP
    loaded = np.asarray(node.update(state, {"link_wrench": jnp.asarray(loads)}, LOAD_STEP,
                                    params=params)["joint_velocities"]) / LOAD_STEP
    rotation = np.asarray(quaternion.rotate(jnp.asarray(placement[3:]), jnp.eye(3)))
    m = np.asarray(AR4_INERTIA(jnp.asarray(q)))
    weight = np.asarray(AR4_WEIGHT(jnp.asarray(q), jnp.asarray(rotation.T @ [0.0, 0.0, -9.80665])))
    falling = -np.linalg.solve(m, weight)
    jacobian = np.asarray(AR4_JACOBIAN(jnp.asarray(q), a.frame_names[frame]))
    pushed = np.linalg.solve(m, jacobian.T @ np.concatenate([rotation.T @ wrench[:3], rotation.T @ wrench[3:]]))
    rounding = ROUNDING * EPS * float(np.linalg.cond(m))
    return (float(np.max(np.abs(rest - falling)) / np.max(np.abs(falling))),
            float(np.max(np.abs(loaded - rest - pushed)) / np.max(np.abs(pushed))), rounding)


@PROPERTY
@given(turned_loads())
def test_the_weight_and_a_load_move_the_arm_through_its_turned_base_as_their_closed_forms(drawn):
    weight, load, rounding = turned_misfits(drawn)
    assert weight <= rounding and load <= rounding, (weight, load, rounding)


@PROPERTY
@given(turned_loads())
@pytest.mark.parametrize(("kind", "fails"), [(WeightTurnedTheWrongWay, "weight"), (LoadsNotTurned, "load"),
                                             (LoadsReversed, "load")])
def test_control_the_weight_or_the_loads_turned_wrongly_or_reversed_fail_on_every_draw(kind, fails, drawn):
    weight, load, rounding = turned_misfits(drawn, kind)
    # The oracle fails above its rounding; the plant stands a thousand times above it on every draw.
    assert {"weight": weight, "load": load}[fails] > 1e3 * rounding, (kind.__name__, weight, load, rounding)
