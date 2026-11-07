"""The batched scan's own claims, each with the comparison it can fail: a batch of one member is bit for bit its own
unbatched pure scan; chunks of two steps and more are bit for bit the unchunked batch; a leaf every member shares reaches
the compiled step once, with no member axis, as the step's own arguments show; ``_meta`` (the multi-rate counter, the coupling groups' warm starts) is carried per
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


def _handed(monkeypatch) -> list[dict]:
    """What each call of a batched run hands its compiled step, read from the program MADDENING hands out
    (``GraphManager._cached_scan``), below every MIME caller: the lowered program's argument shapes, by node and leaf,
    the batch size, and its argument bytes."""
    from maddening import GraphManager

    seen: list[dict] = []
    real = GraphManager._cached_scan

    def handing(self, key, build):
        program = real(self, key, build)
        if key[0] != "mime_batched_scan" or not key[2] or key[-1]:  # the batched run's own scan, not a measurement
            return program

        def called(*args):
            info = program.lower(*args).args_info
            (state, _, params), _ = info
            leaves = jax.tree.leaves(info)
            seen.append({
                "batch": next(iter(jax.tree.leaves(state))).shape[0],
                "leaves": {(node, leaf): info.shape for node, held in params["nodes"].items()
                           for leaf, info in held.items()},
                "state": sum(int(np.prod(a.shape)) * np.dtype(a.dtype).itemsize for a in jax.tree.leaves(state)),
                "bytes": sum(int(np.prod(a.shape)) * np.dtype(a.dtype).itemsize for a in leaves)})
            return program(*args)

        return called

    monkeypatch.setattr(GraphManager, "_cached_scan", handing)
    return seen


def test_a_leaf_every_member_shares_reaches_the_compiled_step_once_with_no_member_axis(monkeypatch):
    """Members differing only in a command: what the compiled step is handed, at one member and at three. Every leaf
    but the drive's value arrives with its own shape, no member axis, and the arguments of three members exceed one's
    by exactly two members' own bytes (their states and the drive's value): no shared leaf is copied per member. A
    regression anywhere between the plan and the step that broadcast the shared leaves would add their bytes (here 236
    a member) and a leading axis, both of which this sees."""
    plan = _plan([10.0, 20.0, 30.0, 40.0], s(0.01))
    group = plan.groups[0]
    first = plan.points[group.members[0]].tree
    varying = set(group.varying)
    assert varying == {("drive:field.frequency", "value")}
    shared = [(n, k) for n in first for k in first[n] if (n, k) not in varying]
    shared_bytes = sum(first[n][k].nbytes for n, k in shared)
    assert shared and shared_bytes > 0, "a broadcast of the shared leaves would add nothing this could see"
    seen = _handed(monkeypatch)
    tests = group.built.flag_tests()
    for size in (1, 3):
        run_members(group.built, plan.points, group.members[:size], group.varying, group.chunks, tests, ())
    calls = {call["batch"]: call for call in reversed(seen)}
    assert sorted(calls) == [1, 3], sorted(calls)
    for size, call in calls.items():
        for node, leaf in shared:
            assert call["leaves"][(node, leaf)] == first[node][leaf].shape, (size, node, leaf, call["leaves"][(node, leaf)])
        for node, leaf in varying:
            assert call["leaves"][(node, leaf)] == (size, *first[node][leaf].shape), (size, node, leaf)
    own = calls[1]["state"] + sum(first[n][k].nbytes for n, k in varying)
    assert calls[3]["bytes"] - calls[1]["bytes"] == 2 * own, (calls[3]["bytes"], calls[1]["bytes"], own)


def test_points_and_the_batchs_tree_hold_the_first_points_own_array_for_each_leaf_they_share():
    """On the host: the batch's tree marks every leaf but the drive's value unbatched (``None``) and holds the first
    member's own array for it, and every point holds the first point's array for each leaf they share, so a table is
    held once, not per point."""
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
