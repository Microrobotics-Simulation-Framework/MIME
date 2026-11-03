"""A run that keeps no step's state knows each flag over the whole run: the flags are reduced inside the run's compiled
scans (how many steps raised each, and the first and the last), so that "raised at a step and clear at the end" is
reported as raised without keeping a history, and "unknown" is left for a flag the run could not reduce.

The rig is the toy recorder with a pulse: a flag on a field that is nonzero at steps 3, 4 and 9 of a run and clear
everywhere else, so that a run keeping only the last step cannot see it. **What this bounds**: that the reduction is the
per-step one (the same run with its history is the oracle) over chunks, in a run folder read back, and that the
summary of a run says nothing is unknown where the reduction holds. The controls plant the failure: with the carry
removed (``BuiltExperiment._flag_tests`` returning nothing) every case above reads unknown, on every run."""

from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import numpy as np
import pytest

from mime.build import ObservationBuilder, _maddening
from mime.core import NodeFields, node_fields, typed_view
from mime.experiment import BuiltExperiment
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Flag, Observation, Output, OutputKind, Read, RoleKind, register_observation
from mime.results import RunResult
from mime.testing.toys import POSITION, coupled_toy
from mime.testing.toys._experiments import DT
from mime.testing.toys._nodes import ToyRecorderNode, ToyRecorderState
from mime.units import DIMENSIONLESS, LENGTH, s
from mime.vocabulary import register_namespace

register_namespace("pulsing", "0.0.1")

#: The steps (counted from one) at which the pulse is raised.
PULSES = (3, 4, 9)
FLAG = "pulsing.recorder:probe.last"


class PulseNode(ToyRecorderNode):
    """The recorder, its ``last`` raised only at the steps in :data:`PULSES`: the flag a run sees at its last step only
    is clear there."""

    @override
    def update(self, state: NodeFields, boundary_inputs: Any, dt: float, *, params: Any = None) -> NodeFields:
        held = typed_view(ToyRecorderState, state)
        count = held["samples"] + 1.0
        fired = jnp.any(jnp.asarray([count == float(step) for step in PULSES]))
        return node_fields(ToyRecorderState(last=jnp.where(fired, jnp.ones(3, count.dtype), jnp.zeros(3, count.dtype)),
                                            samples=count))


@register_observation
@dataclass(frozen=True, kw_only=True)
class PulsingRecorder(Observation):
    """A recorder whose ``last`` is a flag."""

    name = "pulsing.recorder"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    builds = {"": (PulseNode,)}
    roles = {"body": RoleKind.BODY}
    reads = {"position": Read("body", POSITION.name)}
    outputs = (Output("last", LENGTH, "the pulse", shape=(3,)),
               Output("samples", DIMENSIONLESS, "how many steps it has taken", kind=OutputKind.AT_END))
    raises = (Flag("last", "the pulse was raised, for the test"),)
    body: Any = None

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        node = b.add_node(PulseNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("position"), into=node["position"])


def pulsing(steps: int = 12) -> Any:
    experiment = coupled_toy()
    experiment.observe(PulsingRecorder(body="probe"))
    return experiment.build(), s(steps * DT.value)


def without_the_carry(monkeypatch):
    monkeypatch.setattr(BuiltExperiment, "_flag_tests", lambda self: ())


def test_a_flag_raised_at_middle_steps_and_clear_at_the_end_is_reported_raised_without_a_history():
    built, duration = pulsing()
    result = built.run(duration)
    assert result.history is None and result.tracks[FLAG].raised is None
    (read,) = result.flags
    assert (read.was_raised, read.raised, read.steps) == (True, 3, 12)
    assert (read.first, read.last, read.at_last) == (pytest.approx(3 * DT.value), pytest.approx(9 * DT.value), False)
    assert "raised at 3 of 12 steps" in read.text() and "unknown" not in read.text()


def test_the_reduction_is_the_per_step_one_over_chunks_of_every_size():
    """The oracle is the same run keeping every step: its flag at each step, reduced on the host. A hook splits the run
    into chunks, with the pulses on both sides of a boundary and one at it (steps 4 and 8, 4 and 9)."""
    kept = pulsing()[0].run(pulsing()[1], history=True)
    steps = np.flatnonzero(kept.tracks[FLAG].raised) + 1
    assert list(steps) == list(PULSES)
    for chunk in (1, 2, 3, 4, 6, 12):
        built, duration = pulsing()
        result = built.run(duration, hook=lambda _: None, chunk=s(chunk * DT.value))
        (read,) = result.flags
        assert (read.raised, read.first, read.last) == (len(steps), pytest.approx(steps[0] * DT.value),
                                                        pytest.approx(steps[-1] * DT.value)), chunk


def test_a_run_continued_from_another_counts_its_own_steps_and_says_where_they_begin():
    built, _ = pulsing()
    first = built.run(s(5 * DT.value))
    second = built.run(s(7 * DT.value))
    assert [(r.raised, r.steps) for r in first.flags] == [(2, 5)] and [(r.raised, r.steps) for r in second.flags] == [(1, 7)]
    assert second.flags[0].first == pytest.approx(9 * DT.value) and second.start_step == 5


def test_the_manifest_holds_the_reduction_and_a_result_read_back_reports_the_same(tmp_path):
    built, duration = pulsing()
    result = built.run(duration)
    (entry,) = result.manifest()["flags"]
    assert entry["recorded"] == "the whole run" and entry["array"] is None and entry["was_raised"] is True
    assert entry["whole_run"] == {"raised_steps": 3, "first_step": 3, "last_step": 9}
    read_back = RunResult.read(result.to_npz(tmp_path / "run.npz"))
    (read,) = read_back.flags
    assert (read.was_raised, read.raised, read.steps) == (True, 3, 12) and read.first == pytest.approx(3 * DT.value)
    assert read_back.manifest()["flags"][0]["whole_run"] == entry["whole_run"]


def test_the_reduction_changes_nothing_the_run_computes():
    built, duration = pulsing()
    plain = pulsing()[0]
    with_carry = built.run(duration)
    original = BuiltExperiment._flag_tests
    BuiltExperiment._flag_tests = lambda self: ()  # type: ignore[method-assign]
    try:
        without = plain.run(duration)
    finally:
        BuiltExperiment._flag_tests = original  # type: ignore[method-assign]
    for node, fields in without.final.items():
        for name, value in fields.items():
            assert np.asarray(with_carry.final[node][name]).tobytes() == np.asarray(value).tobytes(), (node, name)


def test_control_with_the_carry_removed_the_same_runs_read_unknown(monkeypatch):
    """Every case above, with ``_flag_tests`` returning nothing: the flag is unknown, so each assertion that it was
    raised fails, on every run (this reads the state of the flag and nothing else)."""
    without_the_carry(monkeypatch)
    built, duration = pulsing()
    result = built.run(duration)
    (read,) = result.flags
    assert (read.was_raised, read.raised, read.at_last) == (None, None, False)
    assert "unknown, not recorded per step" in read.text()
    assert result.manifest()["flags"][0]["recorded"] == "the last step" and result.manifest()["flags"][0]["whole_run"] is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_threshold_in_the_fields_type_judges_a_value_as_the_host_does_float64_limit(dtype):
    """A limit has no exact float32: a value just above it in float64 must be above in float32 too. 0.1 rounds up in
    float32, so the nearest float32 to it is above the limit by the host's comparison, and the threshold in float32 must
    be below it."""
    limit = 0.1
    kind = np.dtype(dtype)
    threshold = _maddening._threshold(limit, kind)
    near = np.asarray([np.nextafter(kind.type(threshold), kind.type(np.inf)), kind.type(threshold),
                       np.nextafter(kind.type(threshold), kind.type(-np.inf)), kind.type(limit)], dtype=kind)
    host = near.astype(np.float64) > limit
    assert list(near > kind.type(threshold)) == list(host)
    if kind == np.float32:
        naive = near > kind.type(limit)
        assert list(naive) != list(host), "the control: comparing against the rounded limit misjudges the value at it"


def test_script_a_ends_in_the_same_bits_with_and_without_the_reduction(monkeypatch):
    """Script A's reproduction arm, run without a history: its final state is the same bytes whether or not the scans
    carry the reduction. The control is the asserts on ``whole``: the two arms did take different programs, so the
    equality is not of one run with itself."""
    from script_a import reproduction_arm

    with_carry = reproduction_arm().build().run(s(0.025))
    assert with_carry.tracks[next(iter(with_carry.tracks))].whole is not None, "the carry ran"
    without_the_carry(monkeypatch)
    plain = reproduction_arm().build().run(s(0.025))
    assert all(track.whole is None for track in plain.tracks.values()), "the carry was removed"
    for node, fields in plain.final.items():
        for name, value in fields.items():
            assert np.asarray(with_carry.final[node][name]).tobytes() == np.asarray(value).tobytes(), (node, name)
