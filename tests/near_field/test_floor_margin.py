"""The floor stop's margin, ``FLOOR_MARGIN``, pinned in value: how far above ``eps || |A| |x| + |b| ||`` a stagnated
column's least true residual may lie and still be named at its rounding floor.

The stall tests plant a stall at ``1 / eps`` times the floor, so a margin a hundred times looser names it the same; this
file plants the residual's floor *near* the margin, on both sides, and takes the name from the recheck's own naming
(``mime.study._gradient.cause`` on the reports of the member's solve and the recheck's, the recheck's stopping
``recheck_stopping``).

**The plant**: a small system (a diagonally dominant matrix, a Krylov length of 6) solved through the real :func:`gmres`,
its operator ``v -> A v + s eps u(v) (|A| |v|)`` with ``u(v)`` in [-1, 1] a hash of the bits of ``v`` (seeded by the hash's
constants): the same iterate gets the same error and a different iterate an unrelated one, so the residual cannot fall
below about ``s eps || |A| |x| ||`` however many restarts are taken, and ``s`` sets where. (The right-hand sides are scaled
to a largest entry of 1, so that the solve's own column scaling changes no bit and the test's re-evaluation sees the
solve's error.) Each case asserts its measured ratio, from the returned iterate, before the name: (a) at about twice the
margin, a stall above the floor, which the recheck runs on past to the cap, the member's stagnation named ``stagnated``;
(b) at about half the margin, at the floor, named ``floor-limited``. A margin 100 times looser (400) names (a)
``floor-limited``; one 4 times tighter (1) names (b) ``stagnated``. Both dtypes."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import precision

from mime.nodes.hydro.near_field import JointDragNode
from mime.nodes.hydro.near_field._gmres import REPORT_FIELDS, Stopping, gmres, recheck_stopping
from mime.results._gradient import SolveReport
from mime.study._gradient import FLOOR_LIMITED, STAGNATED, cause

SIZE, COLUMNS, KRYLOV = 24, 4, 6
#: The error's size in eps (``s``) that lands each case, found by measurement in either dtype; the ratio asserted is the
#: measured one, so a wrong ``s`` fails the landing, never the name.
SCALES = {"above": 24.0, "at": 5.0}
#: The margin's value the cases are planted around: a literal, not ``FLOOR_MARGIN``, so that a moved margin changes the
#: name and not the plant (the cases' ratios are in units of the floor estimate).
MARGIN = 4.0
#: The margin's multiples each case plants at.
TARGETS = {"above": 2.0, "at": 0.5}
BAND = 1.5


def hashed(v: jax.Array) -> jax.Array:
    """``v``'s error shape: a hash of each entry's bits into [-1, 1], the same for the same entry, unrelated for a
    neighbouring float."""
    bits = jax.lax.bitcast_convert_type(v, jnp.uint64 if v.dtype == jnp.float64 else jnp.uint32)
    h = (bits ^ (bits >> 32)).astype(jnp.uint32) if v.dtype == jnp.float64 else bits
    for multiplier in (0x7FEB352D, 0x846CA68B):
        h = (h ^ (h >> 15)) * jnp.uint32(multiplier)
    h = h ^ (h >> 16)
    return (h >> 8).astype(v.dtype) / (2.0 ** 23) - 1.0


def planted(dtype: str, scale: float):
    """The matrix, the right-hand sides, the perturbed operator (jitted) and the operator's magnitude."""
    generator = np.random.default_rng(20261111)
    matrix = generator.uniform(-1.0, 1.0, (SIZE, SIZE)) + SIZE * 0.35 * np.eye(SIZE)
    rhs = generator.uniform(-1.0, 1.0, (SIZE, COLUMNS))
    rhs = rhs / np.max(np.abs(rhs), axis=0)  # largest entry exactly 1: the solve's column scaling is then exact (by 1)
    a, b = jnp.asarray(matrix, dtype), jnp.asarray(rhs, dtype)
    eps = float(jnp.finfo(a.dtype).eps)

    def operator(v):
        return a @ v + scale * eps * hashed(v) * (jnp.abs(a) @ jnp.abs(v))

    return b, jax.jit(operator), jax.jit(lambda v: jnp.abs(a) @ v), eps


def solved(dtype: str, scale: float) -> dict:
    """The member's solve (the node's defaults to a window and a floor test off) and the recheck's, through
    :func:`gmres`, and each column's measured ratio of its returned iterate's true residual to the floor estimate."""
    b, operator, magnitude, eps = planted(dtype, scale)
    member = Stopping(1e-30, restart=KRYLOV, cap=400)
    again = recheck_stopping(dataclasses.replace(member, cap=100))
    first = jax.jit(lambda rhs: gmres(operator, rhs, lambda v: v, member))(b)
    second = jax.jit(lambda rhs: gmres(operator, rhs, lambda v: v, again, magnitude))(b)
    x = second[0]
    residual = np.linalg.norm(np.asarray(b - operator(x), np.float64), axis=0)
    floor = eps * np.linalg.norm(np.asarray(magnitude(jnp.abs(x)) + jnp.abs(b), np.float64), axis=0)
    return {"member": np.asarray(first[1]), "recheck": np.asarray(second[1]), "ratio": residual / floor}


class _Stub:
    """The one thing ``cause`` reads of a node, ``floor_stops``, taken from the near field's own."""

    floor_stops = JointDragNode.floor_stops


def named(run: dict) -> str:
    own, again = (SolveReport("near_field", run[key], None, None, "forward") for key in ("member", "recheck"))
    return cause(_Stub(), own, again)  # type: ignore[arg-type]


def lands(run: dict, case: str) -> None:
    """The case's planted ratio, measured, before it is named: every column within ``BAND`` of its target margin times
    the margin (and so on the side of the margin the case says)."""
    target = TARGETS[case] * MARGIN
    ratio = run["ratio"]
    assert np.all(ratio >= target / BAND) and np.all(ratio <= target * BAND), (case, ratio)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_stagnation_at_twice_the_margin_above_the_floor_is_named_stagnated_never_floor_limited(dtype):
    """(a) The least true residual of every column lies at about twice the margin (4) times ``eps || |A| |x| + |b| ||``
    (measured, from the iterate returned): a stall above the floor, so the recheck's floor stop does not stop it, each
    column runs on to the lifted cap (counted at it, none stagnated), and the member's stagnation is named
    ``stagnated``. A margin 100 times looser (400) stops it at the floor and names it ``floor-limited``."""
    with precision(dtype):
        run = solved(dtype, SCALES["above"])
    lands(run, "above")
    print(f"\n{dtype}: ratios to the floor {np.round(run['ratio'], 2)} (margin {MARGIN:g}), member "
          f"{run['member']}, recheck {run['recheck']}")
    stagnated, at_cap = REPORT_FIELDS.index("stagnated"), REPORT_FIELDS.index("at_cap")
    assert run["member"][stagnated] == COLUMNS, "the control: the member's own solve stagnated"
    assert run["recheck"][stagnated] == 0 and run["recheck"][at_cap] == COLUMNS
    assert named(run) == STAGNATED


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_stagnation_at_half_the_margin_above_the_floor_is_named_floor_limited(dtype):
    """(b) The same plant with the error set so that the least true residual of every column lies at about half
    the margin (4) times the floor (measured): the recheck's solve stops stagnated there and is named
    ``floor-limited``. A margin 4 times tighter (1) runs it on to the cap and names it ``stagnated``."""
    with precision(dtype):
        run = solved(dtype, SCALES["at"])
    lands(run, "at")
    print(f"\n{dtype}: ratios to the floor {np.round(run['ratio'], 2)} (margin {MARGIN:g}), member "
          f"{run['member']}, recheck {run['recheck']}")
    stagnated = REPORT_FIELDS.index("stagnated")
    assert run["member"][stagnated] == COLUMNS and run["recheck"][stagnated] == COLUMNS
    assert named(run) == FLOOR_LIMITED
