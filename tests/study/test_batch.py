"""The batched scan's own claims, each with the comparison it can fail: a batch of one member is bit for bit its own
unbatched pure scan; chunks of two steps and more are bit for bit the unchunked batch; a leaf every member shares enters
the step once, with no member axis; ``_meta`` (the multi-rate counter, the coupling groups' warm starts) is carried per
member through the chunks; and members of an experiment with a coupling group and a node at a third of the base rate
are within gate 1's tolerance of their own runs. Bits are claimed at the usable CPU count and OpenBLAS thread count the
test runs at (MIME-ANO-140, -141)."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest
from compare import TOLERANCE_EPS, misses, worst
from script_a import reproduction_arm

from mime import Study
from mime.study import grid
from mime.study._study import run_members
from mime.testing.toys import ToyRecorder
from mime.testing.toys._experiments import full_toy
from mime.units import Hz, N, kg, m, s


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    try:
        yield request.param
    finally:
        jax.config.update("jax_enable_x64", held)


def _plan(points, duration):
    exp = reproduction_arm()
    field = exp.handle("field")
    return Study(exp).sweep(grid({field.commands.frequency: [Hz(f) for f in points], "umr.mass": [kg(8.5e-5)]}),
                            duration=duration).plan()


def _bits(a, b):
    return all(np.array_equal(a[n][f], b[n][f]) for n in a for f in a[n])


def test_a_batch_of_one_is_bit_for_bit_its_own_unbatched_scan(dtype):
    plan = _plan([20.0], s(0.05))
    group = plan.groups[0]
    tests = group.built.flag_tests()
    batched, tallies, _ = run_members(group.built, plan.points, group.members, group.varying, group.chunks, tests, ())
    alone, alone_tallies, _ = run_members(group.built, plan.points, group.members, group.varying, group.chunks, tests,
                                          (), batched=False)
    assert _bits(batched, alone)
    assert all(np.array_equal(a.count, b.count) for a, b in zip(tallies, alone_tallies, strict=True))


def test_a_batch_of_one_moved_by_one_ulp_is_not_bit_for_bit(dtype):
    """The control: the comparison sees one ulp of one leaf."""
    plan = _plan([20.0], s(0.05))
    group = plan.groups[0]
    moved = [dataclasses.replace(p) for p in plan.points]
    tree = {n: dict(leaves) for n, leaves in moved[0].tree.items()}
    held = tree["drive:field.frequency"]["value"]
    tree["drive:field.frequency"]["value"] = np.nextafter(held, np.inf).astype(held.dtype)
    moved[0].tree = tree
    tests = group.built.flag_tests()
    batched, _, _ = run_members(group.built, plan.points, group.members, group.varying, group.chunks, tests, ())
    planted, _, _ = run_members(group.built, moved, group.members, group.varying, group.chunks, tests, (), batched=False)
    assert not _bits(batched, planted)


def test_chunks_of_two_steps_and_more_are_bit_for_bit_the_unchunked_batch(dtype):
    plan = _plan([10.0, 20.0, 30.0], s(0.1))
    group = plan.groups[0]
    tests = group.built.flag_tests()
    runs = {chunks: run_members(group.built, plan.points, group.members, group.varying, chunks, tests, ())[0]
            for chunks in ((200,), (100, 100), (10,) * 20, (2,) * 100)}
    whole = runs[(200,)]
    for chunks, final in runs.items():
        assert _bits(whole, final), chunks


def test_a_leaf_every_member_shares_enters_the_step_once_with_no_member_axis():
    """Members differing only in a command: the surrogate's weights and every other leaf but the drive's value are the
    first member's own arrays, marked unbatched (``None``), never stacked per member."""
    from mime.study._study import stacked_tree

    plan = _plan([10.0, 20.0, 30.0, 40.0], s(0.01))
    group = plan.groups[0]
    tree, axes = stacked_tree(plan.points, group.members, group.varying)
    first = plan.points[group.members[0]].tree
    batched = [(node, leaf) for node, leaf, axis in axes if axis == 0]
    assert batched == [("drive:field.frequency", "value")]
    for node, leaf, axis in axes:
        if axis is None:
            assert tree["nodes"][node][leaf] is first[node][leaf], (node, leaf)
    weights = max(((n, k) for n, k, a in axes if a is None), key=lambda nk: first[nk[0]][nk[1]].nbytes)
    assert tree["nodes"][weights[0]][weights[1]].shape == first[weights[0]][weights[1]].shape
    # Every point holds the first point's own array for each leaf they share: a table is held once, not per point.
    for index in group.members[1:]:
        own = plan.points[index].tree
        assert all(own[n][k] is first[n][k] for n, k, a in axes if a is None)
        assert own["drive:field.frequency"]["value"] is not first["drive:field.frequency"]["value"]


def _toy(rate: float):
    exp = full_toy()
    exp.remove("toy.recorder:probe")
    exp.observe(ToyRecorder(body="probe", rate=s(rate)))
    return exp


def _toy_gate(rate: float, chunks: tuple[int, ...] | None = None):
    exp = _toy(rate)
    spring = exp.handle("toy.spring:probe")
    values = [N(k) / m(1.0) for k in (1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2)]
    study = Study(exp).sweep(grid({spring.k: values}), duration=s(1.0))  # type: ignore[union-attr]
    plan = study.plan()
    group = plan.groups[0]
    if chunks is not None:
        group.chunks = chunks
    result = study.run()
    references = [_toy(rate).build().run(s(1.0), history=True, override={spring.k: value})  # type: ignore[union-attr]
                  for value in values]
    return [worst(misses(references[i], result.member(i))) for i in range(len(values))]


def test_members_of_a_coupled_multi_rate_experiment_are_within_the_tolerance_of_their_own_runs():
    """The full toy: a coupling group of the medium and the body, and an observation at a third of the base rate, so
    ``_meta`` holds the multi-rate counter and the group's warm start; 50 steps in chunks of two, so a counter not
    carried per member through the chunks would put the observation's steps elsewhere."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        found = _toy_gate(0.06, (2,) * 25)
    finally:
        jax.config.update("jax_enable_x64", held)
    print(found)
    assert all(miss <= TOLERANCE_EPS["float64"] for _, miss in found), found
