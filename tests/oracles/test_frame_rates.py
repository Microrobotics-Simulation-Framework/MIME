"""The frames' angular velocities a chain segment publishes (phase 7a4, ``apparatus.frame_angular_velocities``), the
rates' oracle: on both rungs of the AR4, its base turned by a drawn rotation and its joints moving, every frame's
published angular velocity is the central finite difference of the node's own frame poses along the joint rates it
was stepped at (the kinematic arm's ``(theta_{n+1} - theta_n) / dt``, the dynamic arm's new velocity), at the new
positions. Float64, the step ``1e-6`` of the difference: its truncation is some ``1e-12`` of the rate and its rounding
``eps / 1e-6`` of a pose's unit scale, some ``2e-10``; the bound is ``1e-8`` of the largest rate.

The controls, each failing on every draw: the rates' sign flipped, and the rates not turned into the world (the base's
turn dropped), which no other oracle sees (the arm's goldens hold its pose, so their rates are zero, and no other case
turns an arm's base). The golden outputs' rates are a regression guard, this file their oracle."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import AR4
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.core import quaternion
from mime.core.multibody import forward_kinematics, read_urdf
from mime.nodes.actuation import DynamicArmNode, KinematicArmNode

PROPERTY = settings(deadline=None, max_examples=100)
DT, STEP = 1e-3, 1e-6
BOUND = 1e-8


def signed(self, base, mount, local):
    return mount - quaternion.rotate(base[3:], local.T).T


def unturned(self, base, mount, local):
    return mount + local


@pytest.fixture(scope="module")
def nodes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        articulation = read_urdf(AR4).tree.articulation
    dofs = articulation.dofs
    held = {"placement": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), "initial_joint_positions": (0.0,) * dofs}
    built = {"kinematic": (KinematicArmNode, dict(KinematicArmNode.described(articulation), **held)),
             "dynamic": (DynamicArmNode, dict(DynamicArmNode.described(articulation, (0.0,) * dofs, (0.0,) * dofs),
                                              gravity=(0.0, 0.0, 0.0), **held))}
    found = {}
    for rung, (cls, settings_) in built.items():
        for plant, rates in (("", None), ("signed", signed), ("unturned", unturned)):
            kind = cls if rates is None else type(f"{plant}_{cls.__name__}", (cls,), {"rates_in_world": rates})
            found[(rung, plant)] = kind(f"rates_{rung}_{plant}", DT, **settings_)
    return found, articulation


@st.composite
def motions(draw, articulation):
    """A base turned by 0.5 to 3 rad about an axis drawn over the sphere, joint positions within the middle half of
    their limits, and joint rates of 0.2 to 1 rad/s either way."""
    polar, azimuth = draw(st.floats(0.0, math.pi)), draw(st.floats(0.0, 2.0 * math.pi))
    axis = np.asarray([math.sin(polar) * math.cos(azimuth), math.sin(polar) * math.sin(azimuth), math.cos(polar)])
    angle = draw(st.floats(0.5, 3.0))
    turn = (math.cos(angle / 2.0), *(math.sin(angle / 2.0) * axis))
    lower, upper = np.asarray(articulation.lower), np.asarray(articulation.upper)
    middle, half = 0.5 * (lower + upper), 0.25 * (upper - lower)
    q = np.asarray([middle[i] + half[i] * draw(st.floats(-1.0, 1.0)) for i in range(articulation.dofs)])
    rates = np.asarray([draw(st.floats(0.2, 1.0)) * draw(st.sampled_from((-1.0, 1.0)))
                        for _ in range(articulation.dofs)])
    placement = (draw(st.floats(-0.5, 0.5)), draw(st.floats(-0.5, 0.5)), draw(st.floats(-0.5, 0.5)), *turn)
    return np.asarray(placement), q, rates


_COMPILED: dict[tuple, object] = {}


def stepped(node, rung: str):
    """The node's step, compiled: its published rates and new positions, and the rates it was stepped at."""
    key = ("step", id(node))
    if key not in _COMPILED:
        def step(p, q, rates):
            if rung == "kinematic":
                state = {"frame_poses": jnp.zeros((len(node.constants()["frame_names"]), 7)), "joint_positions": q,
                         "frame_angular_velocities": jnp.zeros((len(node.constants()["frame_names"]), 3))}
                out = node.advance(p, state, {"joint_positions": q + DT * rates}, DT)
                return out["frame_angular_velocities"], out["joint_positions"], rates
            state = {"frame_poses": jnp.zeros((len(node.constants()["frame_names"]), 7)), "joint_positions": q,
                     "joint_velocities": rates, "frame_angular_velocities": jnp.zeros(
                         (len(node.constants()["frame_names"]), 3))}
            out = node.advance(p, state, {"joint_torques": jnp.zeros_like(q)}, DT)
            return out["frame_angular_velocities"], out["joint_positions"], out["joint_velocities"]
        _COMPILED[key] = jax.jit(step)
    return _COMPILED[key]


def differenced(node):
    """Every frame's angular velocity by the central difference of the node's own world poses along ``rates``."""
    key = ("difference", id(node))
    if key not in _COMPILED:
        def difference(p, q, rates):
            base = jnp.asarray(p["placement"])
            ahead = node.in_world(base, forward_kinematics(node.articulation(p, q.dtype), q + STEP * rates))
            behind = node.in_world(base, forward_kinematics(node.articulation(p, q.dtype), q - STEP * rates))
            turned = jax.vmap(quaternion.multiply)(ahead[:, 3:], jax.vmap(quaternion.conjugate)(behind[:, 3:]))
            return 2.0 * jnp.sign(turned[:, :1]) * turned[:, 1:] / (2.0 * STEP)
        _COMPILED[key] = jax.jit(difference)
    return _COMPILED[key]


def misfit(node, rung: str, drawn) -> float:
    placement, q, rates = drawn
    p = {**node._step_constants(None), "placement": jnp.asarray(placement)}
    published, q_next, moved = stepped(node, rung)(p, jnp.asarray(q), jnp.asarray(rates))
    expected = differenced(node)(p, q_next, moved)
    return float(np.max(np.abs(np.asarray(published) - np.asarray(expected)))) / float(np.max(np.abs(expected)))


@pytest.mark.parametrize("rung", ("kinematic", "dynamic"))
@PROPERTY
@given(data=st.data())
def test_every_frames_published_rate_is_the_difference_of_its_poses_with_the_base_turned(nodes, rung, data):
    found, articulation = nodes
    drawn = data.draw(motions(articulation))
    assert misfit(found[(rung, "")], rung, drawn) <= BOUND


@pytest.mark.parametrize("rung", ("kinematic", "dynamic"))
@PROPERTY
@given(data=st.data())
def test_control_the_rates_sign_flipped_or_not_turned_into_the_world_fails_on_every_draw(nodes, rung, data):
    found, articulation = nodes
    drawn = data.draw(motions(articulation))
    for plant in ("signed", "unturned"):
        assert misfit(found[(rung, plant)], rung, drawn) > 1e3 * BOUND, plant
