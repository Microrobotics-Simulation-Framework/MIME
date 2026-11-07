"""The drag's class, dissipative (F0-1 M6.4): its resistance positive semidefinite. On the surrogate rung it is so by
construction (its network's ``L L^T``, carried through congruences, the lubrication's semidefinite part added), checked
here once on the law alone: script A's surrogate drag node, as the build makes it, its delivered resistance at drawn
poses in the tube (the centre's offset from the axis anywhere in its room, past the clamp included, in a drawn
direction; rolled about its own axis; tilted up to the 5 degrees it is declared valid to; either end first), definite
at the floor by ``mime.core.ledger.definite_at_floor``, both dtypes. On the MRS rung definiteness depends on the
state, and the body checks it at run time: here, the body's check counts a step whose resistance is indefinite and
none whose resistance is definite.

**The plant**, on every draw: F0's spike FU1's indefinite part, 1.1 times the least eigenvalue of the resistance's scaled
symmetric part taken off along its eigenvector. The sign of the drag's work alone caught it on none of that spike's 50 draws:
definiteness is the check that sees it. And a semidefinite resistance (one that does not resist one turning) passes at
the floor, where a plain Cholesky factorisation fails on it."""

from __future__ import annotations

import math
from functools import cache

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from ledger_cases import CHECKED_DRAG, LOAD_LAW, free_states, in_dtype, metered_body, one_step

from mime.core import ledger as books

DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])
#: Script A's tube and the room its body's centre has: the tube's radius less the body's extent and the margin.
TUBE_RADIUS, EXTENT, MARGIN = 3.175e-3, 1.33 * 1.56e-3, 1e-4
ROOM = TUBE_RADIUS - EXTENT - MARGIN
TILT = math.radians(5.0)


@cache
def drag_node():
    from script_a import reproduction_arm

    return reproduction_arm().build().graph.get_node("drag:umr")


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def about(angle: float, axis) -> np.ndarray:
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * np.asarray(axis, dtype=np.float64)])


@st.composite
def poses(draw) -> np.ndarray:
    """A pose of script A's body in its tube (along x through the origin)."""
    towards, share = draw(st.floats(0.0, 2.0 * math.pi)), draw(st.floats(0.0, 1.0))
    roll, tilt, tilted = draw(st.floats(0.0, 2.0 * math.pi)), draw(st.floats(0.0, TILT)), draw(st.floats(0, 2 * math.pi))
    orientation = quaternion_product(about(tilt, (0.0, math.cos(tilted), math.sin(tilted))), about(roll, (1, 0, 0)))
    if draw(st.booleans()):
        orientation = quaternion_product(about(math.pi, (0.0, 0.0, 1.0)), orientation)
    across = np.array([0.0, math.cos(towards), math.sin(towards)])
    return np.concatenate([[draw(st.floats(-0.05, 0.05))] + list(share * ROOM * across[1:]), orientation])


def delivered(pose: np.ndarray, dtype) -> jnp.ndarray:
    node = drag_node()
    p = {k: jnp.asarray(v, dtype=dtype) for k, v in node._step_constants(None).items()}
    state = {k: jnp.asarray(v, dtype=dtype) for k, v in node.initial_state().items()}
    return node.advance(p, state, {"pose": jnp.asarray(pose, dtype=dtype)}, 5e-4)["resistance"]


def indefinite(resistance):
    """FU1's plant: 1.1 times the least eigenvalue of the scaled symmetric part taken off along its eigenvector."""
    symmetric = 0.5 * (resistance + resistance.T)
    scale = jnp.sqrt(jnp.diag(symmetric))
    least, vectors = jnp.linalg.eigh(symmetric / (scale[:, None] * scale[None, :]))
    along = vectors[:, 0] * scale
    return resistance - 1.1 * least[0] * jnp.outer(along, along)


def scaled_least(resistance) -> float:
    symmetric = np.asarray(0.5 * (resistance + resistance.T), dtype=np.float64)
    scale = np.sqrt(np.diag(symmetric))
    return float(np.linalg.eigvalsh(symmetric / np.outer(scale, scale))[0])


@DTYPES
@settings(deadline=None)
@given(pose=poses())
def test_the_surrogate_s_resistance_is_definite_wherever_the_body_is(name, pose):
    with in_dtype(name) as dtype:
        resistance = delivered(pose, dtype)
        assert bool(books.definite_at_floor(resistance)), scaled_least(resistance)
    assert scaled_least(resistance) > 0.5


@DTYPES
@settings(deadline=None)
@given(pose=poses())
def test_an_indefinite_part_in_the_resistance_is_caught_on_every_draw(name, pose):
    with in_dtype(name) as dtype:
        assert not bool(books.definite_at_floor(indefinite(delivered(pose, dtype))))


BODY = metered_body(laws=(LOAD_LAW, CHECKED_DRAG))


@DTYPES
@settings(deadline=None)
@given(drawn=free_states(), planted=st.booleans())
def test_the_body_counts_each_step_whose_checked_resistance_is_indefinite(name, drawn, planted):
    with in_dtype(name) as dtype:
        resistance = jnp.asarray(drawn["resistance"], dtype=dtype)
        found = one_step(BODY, {**drawn, "resistance": indefinite(resistance) if planted else resistance}, dtype)
    row = books.laws(books.deliveries(BODY.constants())).index(CHECKED_DRAG.law)
    assert float(found["ledger_indefinite_steps"][row]) == (1.0 if planted else 0.0)


@DTYPES
@settings(deadline=None)
@given(drawn=free_states(), axis=st.integers(3, 5))
def test_a_semidefinite_resistance_passes_where_a_plain_factorisation_fails(name, drawn, axis):
    with in_dtype(name) as dtype:
        held = np.asarray(drawn["resistance"], dtype=np.float64)
        held[axis, :] = held[:, axis] = 0.0
        resistance = jnp.asarray(held, dtype=dtype)
        assert bool(books.definite_at_floor(resistance))
        assert not bool(jnp.all(jnp.isfinite(jnp.linalg.cholesky(resistance))))
