"""Gravity's class, settled on the law alone (F0-1 M6.4): ``GravityNode``'s wrench is the negative gradient of its
family's energy, ``U = -(m g + F_b) . x - F_b . (R_q c_b)`` (``GravityNode.energy``), the force by the position's
gradient and the couple by the gradient on the rotation group (the pose turned by ``exp(theta)`` in the world, at
``theta = 0``: a quaternion's four numbers are not independent). Judged within 16 eps of each part's arithmetic in
magnitude (``|m g| + |F_b|`` for the force, ``|c_b| |F_b|`` for the couple), on draws of the law and the pose in both
dtypes: a body 0.1 to 10 times the fluid's density, of FL-9's mass to a factor of two, gravity in a drawn direction,
its centre of buoyancy off the centre of mass by up to a millimetre in a drawn direction, or on it.

**The plant**, on every draw: the weight's term times 1.01 (F0's spike FU1's), not the buoyancy's, so that it is a hundredth of
the weight whatever the body's density: caught far above the floor (``tests/energy/predictions.json``)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from ledger_cases import MASS, in_dtype, unit, vector

from mime.core import quaternion
from mime.nodes.body import GravityNode

ULPS = 16.0
DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])


class HeavierWeight(GravityNode):
    """A plant: the weight's term times 1.01, the buoyancy as it is."""

    def wrench(self, p, pose):
        return super().wrench(p, pose) + jnp.concatenate([0.01 * p["mass"] * jnp.asarray(p["gravity"]),
                                                          jnp.zeros(3, dtype=jnp.asarray(p["gravity"]).dtype)])


@st.composite
def laws(draw):
    """A gravity law and a pose: the body's mass, its density over the fluid's, gravity's direction, the centre of
    buoyancy (or none), and the body's pose."""
    mass = MASS * draw(st.floats(0.5, 2.0))
    ratio = draw(st.floats(0.1, 10.0))
    density = 1000.0 * draw(st.floats(0.8, 1.2))
    centre = tuple(vector(draw, 1e-5, 1e-3)) if draw(st.booleans()) else None
    held = {"mass": mass, "buoyant_volume": mass / (ratio * density), "fluid_density": density,
            "gravity": tuple(9.81 * unit(draw, 3)), "centre_of_buoyancy": centre}
    pose = np.concatenate([vector(draw, 1e-5, 1e-2), unit(draw, 4)])
    return held, pose


def judged(node_class: type[GravityNode], held: dict[str, Any], pose: np.ndarray, dtype: Any) -> float:
    """How far the law's wrench stands from the negative gradient of its energy, in units of the floor."""
    node = node_class("g", 5e-4, **held)
    p = {k: jnp.asarray(v, dtype=dtype) for k, v in node._step_constants(None).items()}
    at = jnp.asarray(pose, dtype=dtype)

    def energy(offset):
        moved = jnp.concatenate([at[:3] + offset[:3], quaternion.multiply(quaternion.exponential(offset[3:]), at[3:])])
        return node.energy(p, moved)

    expected = -np.asarray(jax.grad(energy)(jnp.zeros(6, dtype=dtype)), dtype=np.float64)
    found = np.asarray(node.wrench(p, at if node.constants()["centre_of_buoyancy"] is not None else None),
                       dtype=np.float64)
    g = np.asarray(held["gravity"])
    buoyancy = held["fluid_density"] * held["buoyant_volume"] * np.linalg.norm(g)
    eps = float(jnp.finfo(dtype).eps)
    force_scale = held["mass"] * np.linalg.norm(g) + buoyancy
    centre = held["centre_of_buoyancy"]
    couple_scale = buoyancy * (np.linalg.norm(centre) if centre is not None else 0.0)
    over = np.linalg.norm(found[:3] - expected[:3]) / (ULPS * eps * force_scale)
    apart = np.linalg.norm(found[3:] - expected[3:])
    if couple_scale > 0.0:
        over = max(over, apart / (ULPS * eps * couple_scale))
    elif apart != 0.0:
        over = np.inf
    return float(over)


@DTYPES
@settings(deadline=None)
@given(drawn=laws())
def test_gravity_is_the_negative_gradient_of_its_family_s_energy(name, drawn):
    with in_dtype(name) as dtype:
        assert judged(GravityNode, *drawn, dtype) <= 1.0


@DTYPES
@settings(deadline=None)
@given(drawn=laws())
def test_a_weight_one_percent_heavier_is_caught_on_every_draw(name, drawn):
    with in_dtype(name) as dtype:
        assert judged(HeavierWeight, *drawn, dtype) > 50.0
