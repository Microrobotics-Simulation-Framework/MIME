"""Reciprocity of the corrected unit-tube table (MIME-ANO-117), checked where the evaluator does not hide it.

The image system is reciprocal, ``G(x, y) = G(y, x)^T``. The tube's evaluator blends every pair's read with its
reciprocal's (``tube_correction``, the rule on the axis), and the blend acts at every pair, not only near the axis, so
the matrix it returns is reciprocal to rounding whatever the table holds: a reciprocity measured through it cannot fail.
Reciprocity of the table is therefore measured in two places that are not behind the blend, on the shipped asset:

- **the file's knots** (``scripts/tube_table/build.py: knot_reciprocity``, also the gate's): the block ``G(t, s)``
  against ``G(s, t)^T``, at every knot. Measured on the shipped table (2500 modes): at most 8.4e-10, at (0.997, 0.997,
  dphi 2.19, dz 0), and none of the 2,903,181 knots above the gate's declared 1e-8; for both radii below 0.95 at most
  3.7e-14. The table it superseded (1667 modes) missed 2.0e-7 there, at 129 knots above 1e-8: the truncation at the most
  nearly singular pairs. The bound here is the declared 1e-8, twelve times the measured maximum.
- **the pre-blend reads at the table's knots** (``tube_reads``): ``world[t, s]`` against ``world[s, t]^T``, for pairs
  that lie on knots, where the cubic read reproduces the table to rounding, so that the pair's asymmetry is the file's
  knots' plus the turn into the source's frame and the parities' (an error in either is of order one). Off the knots
  the two reads differ by their own interpolation errors, each within the read's 0.23% of the builder: 160 drawn
  point sets measured the asymmetry up to 1.4e-3 (the review's sets, 8.9e-5), which cannot tell a table non-reciprocal
  by 2e-3 from the table, so no test is made there.

Controls, each failing on every run (20 of 20): a table planted non-reciprocal by 2e-3 (the entries with the target's
radius knot above the source's scaled by 1.002) fails the knot check, fails the pre-blend check (each run on pairs
of its own), and passes the check through the blend at 1e-12, which is why that check is no check."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tube_asset
from tube_cases import builder

from mime.nodes.hydro.bem import frame_across, tube_correction, tube_reads, tube_table_from, unit_tube_table

#: The gate's declared tolerance (``scripts/tube_table/full.py``), which the file meets at every knot: the measured
#: maximum, 8.4e-10, is twelve times under it.
DECLARED = 1e-8
KNOT_BOUND = DECLARED
#: The pre-blend reads' asymmetry at knots: the file's own bound, since the reads reproduce its knots.
READ_BOUND = KNOT_BOUND
PLANT = 2e-3
RUNS = 20


@pytest.fixture(scope="module")
def held() -> dict[str, np.ndarray]:
    with np.load(tube_asset.path()) as file:
        return {name: file[name] for name in ("rho_grid", "dphi_grid", "dz_grid", "data", "readable_rho")}


@pytest.fixture(scope="module")
def knot_miss(held) -> np.ndarray:
    return builder().knot_reciprocity(held["data"])


def planted(data: np.ndarray, eps: float) -> np.ndarray:
    """A copy with the entries whose target radius knot is above the source's scaled by ``1 + eps``."""
    out = np.array(data, dtype=np.float64, copy=True)
    for i in range(out.shape[0]):
        out[i, :i] *= 1.0 + eps
    return out


def test_the_files_knots_are_reciprocal_within_the_declared_tolerance_at_every_knot(held, knot_miss):
    radii = held["rho_grid"]
    print(f"\nknot reciprocity: max {knot_miss.max():.3e}, {int((knot_miss > DECLARED).sum())} knots over {DECLARED:g}, "
          f"{int((knot_miss > 1e-12).sum())} over 1e-12")
    assert knot_miss.size == 2_903_181 and knot_miss.max() <= DECLARED
    inner = radii < 0.95
    assert knot_miss[np.ix_(inner, inner)].max() < 1e-12


def test_control_a_table_planted_non_reciprocal_fails_the_knot_check_every_run(held):
    block = held["data"][:, :, :3, :3]
    found = [builder().knot_reciprocity(planted(block, PLANT)).max() for _ in range(RUNS)]
    assert min(found) > 100 * KNOT_BOUND, found


def unit_frame_parameters(table_arrays):
    return {"table": table_arrays, "frame": frame_across((0.0, 0.0, 1.0)), "radius": 1.0, "centre": (0.0, 0.0, 0.0)}


def knot_pairs(held, count: int = 40, seed: int = 5) -> list[np.ndarray]:
    """``count`` pairs of samples in the unit tube, each ``[t, s]`` with ``s`` at a radius knot and ``t`` at a radius,
    angle and axial offset knot from it, the pair turned about the axis and moved along it by a drawn amount; the offsets to 1.5, so that the pairs read between
    different pairs (all the pairs are read together) stay within the table's axial reach of 5. Radii from
    the first off the axis (a sample on it has no angle) to the last but one: a sample at exactly 0.997 can be refused
    by rounding."""
    rng = np.random.default_rng(seed)
    radii, angles, offsets = held["rho_grid"], held["dphi_grid"], held["dz_grid"]
    reach = offsets[offsets <= 1.5]
    pairs = []
    while len(pairs) < count:
        i, j = rng.integers(1, len(radii) - 1, 2)
        if i == j:
            continue
        turn, shift = rng.uniform(-math.pi, math.pi), rng.uniform(-0.5, 0.5)
        dphi, dz = angles[rng.integers(len(angles))], reach[rng.integers(len(reach))]
        sign = rng.choice([-1.0, 1.0], 2)
        source = [radii[j] * math.cos(turn), radii[j] * math.sin(turn), shift]
        target = [radii[i] * math.cos(turn + sign[0] * dphi), radii[i] * math.sin(turn + sign[0] * dphi),
                  shift + sign[1] * dz]
        pairs.append(np.array([target, source]))
    return pairs


@jax.jit
def reads_of(points, table_arrays):
    return tube_reads(points, unit_frame_parameters(table_arrays))


@jax.jit
def correction_of(points, table_arrays):
    return tube_correction(points, unit_frame_parameters(table_arrays), 1.0)


def read_asymmetry(table_arrays, pairs: list[np.ndarray]) -> float:
    """The largest, over the pairs, of ``world[t, s]`` against ``world[s, t]^T`` over the pair's largest entry: all the
    pairs' samples read together, the cross pairs between different pairs left unread."""
    reads = reads_of(jnp.asarray(np.concatenate(pairs)), table_arrays)
    assert bool(reads.accepted)
    world = np.asarray(reads.world)
    return max(float(np.max(np.abs(world[2 * k, 2 * k + 1] - world[2 * k + 1, 2 * k].T))
                     / np.max(np.abs(world[2 * k, 2 * k + 1]))) for k in range(len(pairs)))


def blended_asymmetry(table_arrays, pairs: list[np.ndarray]) -> float:
    """The same through ``tube_correction``, whose blend makes it reciprocal by construction."""
    matrix = np.asarray(correction_of(jnp.asarray(np.concatenate(pairs)), table_arrays).matrix)
    return float(np.max(np.abs(matrix - matrix.T)) / np.max(np.abs(matrix)))


@pytest.fixture(scope="module")
def shipped():
    return tube_table_from(str(tube_asset.path())).arrays(jnp.float64)


@pytest.fixture(scope="module")
def planted_table(held):
    rho, angle, along = held["rho_grid"], held["dphi_grid"], held["dz_grid"]
    table = unit_tube_table(1.0, rho, angle, along, planted(held["data"], PLANT),
                            readable_rho=tuple(float(r) for r in held["readable_rho"]))
    return table.arrays(jnp.float64)


def test_the_pre_blend_reads_at_knots_are_reciprocal_within_the_files_bound(held, shipped):
    found = read_asymmetry(shipped, knot_pairs(held, 20))
    print(f"\npre-blend asymmetry at 20 knot pairs: {found:.3e} against {READ_BOUND:g}")
    assert found <= READ_BOUND


def test_control_a_planted_table_fails_the_pre_blend_check_and_passes_the_blend_every_run(held, planted_table):
    runs = []
    for run in range(RUNS):
        pairs = knot_pairs(held, 3, seed=100 + run)
        runs.append((read_asymmetry(planted_table, pairs), blended_asymmetry(planted_table, pairs)))
    print(f"\nplanted {PLANT:g}: pre-blend {runs[0][0]:.3e}, through the blend {runs[0][1]:.3e}")
    assert all(pre > 100 * READ_BOUND for pre, _ in runs)
    assert all(blend < 1e-12 for _, blend in runs)


def test_control_the_check_through_the_blend_cannot_tell_the_table_from_the_planted_one(held, shipped, planted_table):
    pairs = knot_pairs(held, 10)
    for table in (shipped, planted_table):
        assert blended_asymmetry(table, pairs) < 1e-12
