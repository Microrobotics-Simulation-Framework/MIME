"""Gate 1 in float32: each member of a batched sweep of script A (eight locked points over the field's frequency and the
body's mass, 1 s) against its own run, ``run(override=...)`` of its point on a fresh build: every field of the final
state and the recorded series within the tolerance committed in ``predictions.json`` before this ran, in eps of each
field's scale. Not bit for bit: a batched member is another compiled program (probe B). The controls fail on every
run: one member's frequency leaf moved by 1e-6 relative, and every member read against its neighbour's run."""

from __future__ import annotations

import jax
import pytest
from compare import worst
from members import gate_one


@pytest.fixture(scope="module")
def gate():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield gate_one("float32")
    finally:
        jax.config.update("jax_enable_x64", held)


def test_every_member_is_within_the_committed_tolerance_of_its_own_run(gate, record_property):
    tolerance = gate["tolerance"]
    worst_each = [worst(found) for found in gate["found"]]
    record_property("largest_miss_eps", float(max(miss for _, miss in worst_each)))
    print({i: (where, round(miss, 2)) for i, (where, miss) in enumerate(worst_each)})
    assert len(gate["result"]) == 8 and len(gate["plan"].groups) == 1
    for index, (where, miss) in enumerate(worst_each):
        assert miss <= tolerance, f"member {index}: {where} is {miss:.1f} eps from its own run, past {tolerance}"


def test_a_member_moved_by_one_part_in_a_million_fails(gate, record_property):
    """Its field turns 2 pi f t 1e-6 further by the run's end: at least twice the tolerance (float32: about 480 eps
    measured on the field's flux density, against 150)."""
    where, miss = worst(gate["planted"])
    record_property("planted_miss_eps", float(miss))
    assert miss > 2 * gate["tolerance"], (where, miss)


def test_every_member_read_against_its_neighbours_run_fails(gate):
    assert all(miss > 10 * gate["tolerance"] for miss in gate["shifted"]), gate["shifted"]
