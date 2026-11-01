"""The MRS rung's force sums, ``M^T X`` (``mime.nodes.hydro.mrs.force_sums``), sum over the samples in pieces of at
most ``DEPTH`` terms (MIME-ANO-125). XLA:CPU takes that product, over both operands' first axis, through Eigen, which
blocks the sum by the L1 data cache it reads off the core it first runs on, so on a CPU with two sizes of core a
longer sum was added in one order or the other by where the process started. A saved experiment then failed its
bitwise reproduction in a fresh process: the registry walk's ``mrs_solids`` round trip (its ellipsoid's 297 unknowns),
2 of 10 runs on such a laptop.

- **Every product in the compiled program** of the node's step on that ellipsoid, and of ``resistance`` at 297 and
  6,000 unknowns, sums at most ``DEPTH`` terms, read from the program XLA compiled, not from what the code asks for.
- **A sum no longer than** ``DEPTH`` **is the product it was**, bit for bit, so no result whose sum is that short
  moved (every round-trip case but that ellipsoid, and the saved-form fixtures).
- **A longer one equals the product to rounding**: within the worst-case bound of a sum in any order, ``2 (K + 1) u``
  of the sum of the terms' magnitudes.
- **The control**, failing on every run: the product taken whole at 297 unknowns is found by the same scan.

CI's runners have one size of core, so nothing here can show that two processes' bits differ; that was measured on a
laptop with both, in fresh processes on each core type (in the pull request)."""

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mrs_cases import SOLIDS, drag_node, free_space, mrs

from mime.build._maddening import node_state, parameter_leaves
from mime.nodes.hydro.mrs import DEPTH, PRECISION, force_sums, resistance
from mime.units import mm

DEFINED = re.compile(r"%(?P<name>[\w.\-]+) = [a-z]+\d*\[(?P<dims>[\d,]*)\]")
DOT = re.compile(r" dot\((?:[a-z]+\d*\[[\d,]*\](?:\{[\d,]*\})? )?%(?P<lhs>[\w.\-]+),[^\n]*?"
                 r"lhs_contracting_dims=\{(?P<axes>[\d,]*)\}")


def depths(compiled: str) -> list[int]:
    """How many terms each product of a compiled program sums: its left operand's length along its contracted axes."""
    shapes = {m["name"]: [int(d) for d in m["dims"].split(",") if d] for m in DEFINED.finditer(compiled)}
    return [math.prod(shapes[m["lhs"]][int(a)] for a in m["axes"].split(",") if a) for m in DOT.finditer(compiled)]


def whole(motions, forces):
    return jnp.matmul(motions.T, forces, precision=PRECISION)


def drawn(count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((count, 6)), rng.standard_normal((count, 6))


def ball(count: int) -> tuple[np.ndarray, float]:
    """``count`` points on a unit sphere (Fibonacci) and the blob squared for half their mean spacing."""
    k = np.arange(count) + 0.5
    polar, around = np.arccos(1.0 - 2.0 * k / count), math.pi * (1.0 + math.sqrt(5.0)) * k
    points = np.stack([np.sin(polar) * np.cos(around), np.sin(polar) * np.sin(around), np.cos(polar)], axis=-1)
    return points, 0.25 * 4.0 * math.pi / count


def test_every_product_in_the_compiled_mrs_step_sums_at_most_depth_terms():
    built = free_space("egg", {"egg": SOLIDS["egg"]}, mrs(mm(0.35))).build()
    node = drag_node(built, "egg")
    assert 3 * node.sample_count > DEPTH
    held = node_state(built.graph, "drag:egg")
    step = jax.jit(lambda p, pose: node.update(held, {"pose": pose}, 1e-3, params=p)["resistance"])
    found = depths(step.lower(parameter_leaves(built.graph)["drag:egg"],
                              jnp.asarray(node_state(built.graph, "egg")["pose"])).compile().as_text())
    assert found and max(found) <= DEPTH, found


@pytest.mark.parametrize("count", [99, 2000])
def test_every_product_in_the_compiled_resistance_sums_at_most_depth_terms(count):
    points, blob = ball(count)
    program = jax.jit(resistance).lower(points, blob, 1e-3, np.zeros(3)).compile().as_text()
    found = depths(program)
    assert found and max(found) <= DEPTH and sum(d for d in found if d > 6) >= 3 * count, found


@pytest.mark.parametrize("count", [45, 237, DEPTH])
def test_a_sum_no_longer_than_depth_is_the_product_it_was_bit_for_bit(count):
    motions, forces = drawn(count)
    assert np.array_equal(np.asarray(jax.jit(force_sums)(motions, forces)), np.asarray(jax.jit(whole)(motions, forces)))


@pytest.mark.parametrize("count", [DEPTH + 1, 297, 6000])
def test_a_longer_sum_equals_the_product_to_rounding(count):
    motions, forces = drawn(count, seed=count)
    found = np.asarray(jax.jit(force_sums)(motions, forces))
    bound = 2.0 * (count + 1) * np.finfo(np.float64).eps / 2.0 * (np.abs(motions).T @ np.abs(forces))
    assert np.all(np.abs(found - motions.T @ forces) <= bound)


def test_control_the_product_taken_whole_is_found_by_the_scan():
    motions, forces = drawn(297)
    assert 297 in depths(jax.jit(whole).lower(motions, forces).compile().as_text())
