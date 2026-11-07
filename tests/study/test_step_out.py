"""Gate 5, the sweep's acceptance gate: step-out from rest on script A's reproduction arm, float64, dt 0.5 ms.

Protocol ``FromRest``, 5 s, ``PhaseLag``'s window from 2.5 s; a point is unlocked where its mean spin about the field's
axis over the window, over the field's frequency, is below 0.99; the grid runs from 39 to 48 Hz every 0.25 Hz. The
bracket the sweep finds must be the design's measurement's, last locked 43.25 Hz and first unlocked 43.5 Hz, and the
one two unbatched runs at its ends give, each on a fresh build (``tests/study/predictions_drives.json``, committed
before this ran). The control: the field's amplitude times 0.9 must move the bracket below 43.25 Hz by more than a
grid step (predicted near 41.0 / 41.25 Hz, the square root of the amplitude's scaling of an inertia-dominated capture).
No drag mode moved this bracket in the design's measurement, so a drag plant cannot serve as its control.

**Its claim.** An acceptance test of the sweep and the inertial body, not an accuracy claim about step-out: comparing
with a measured step-out is a separate study. A member past step-out rocks and nearby trajectories separate there, so a
member is read by the study's output (its mean spin over the window), never trajectory by trajectory."""

from __future__ import annotations

from collections.abc import Iterator

import jax
import pytest
from script_a import reproduction_arm

from mime import Experiment, Study
from mime.observations import PhaseLag
from mime.results import RunResult
from mime.scene import X
from mime.study import StepOut, grid, step_out
from mime.study._axes import value_text
from mime.units import Hz, mT, s

FREQUENCIES = tuple(39.0 + 0.25 * i for i in range(37))
AMPLITUDE, CONTROL = 1.2, 1.2 * 0.9
DURATION, AFTER, THRESHOLD = s(5.0), s(2.5), 0.99
LAG = "phase_lag:umr"
#: The design's measurement (and the committed prediction): the bracket from rest at 1.2 mT.
LAST_LOCKED, FIRST_UNLOCKED = 43.25, 43.5


def observed(experiment: Experiment) -> Experiment:
    experiment.observe(PhaseLag("umr", field="field.magnetic", about=X, after=AFTER))
    return experiment


@pytest.fixture(scope="module")
def float64() -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


@pytest.fixture(scope="module")
def brackets(float64: None) -> dict[float, StepOut]:
    """The sweep over the frequency at the amplitude and at its control, one study of 74 members, one group."""
    experiment = observed(reproduction_arm())
    field = experiment.handle("field")
    study = Study(experiment).sweep(grid({field.commands.amplitude: [mT(AMPLITUDE), mT(CONTROL)],  # type: ignore[union-attr]
                                          field.commands.frequency: [Hz(f) for f in FREQUENCIES]}),  # type: ignore[union-attr]
                                    duration=DURATION)
    assert len(study.plan().groups) == 1 and not study.plan().refused
    result = study.run()
    found = step_out(result, LAG, field.commands.frequency, threshold=THRESHOLD)  # type: ignore[union-attr]
    by_amplitude = {dict(held.at)["field.commands.amplitude"]: held for held in found}
    for held in found:
        print(f"\n{held.text()}; ratios {[round(r, 5) for r in held.ratios]}")
    return {AMPLITUDE: by_amplitude[value_text(mT(AMPLITUDE))], CONTROL: by_amplitude[value_text(mT(CONTROL))]}


def alone(frequency: float) -> RunResult:
    return observed(reproduction_arm(frequency=Hz(frequency))).build().run(DURATION)


def ratio_of(run: RunResult, frequency: float) -> float:
    return float(run[LAG].mean_spin.to(Hz)) / frequency


def test_the_bracket_from_rest_is_the_designs_measurement(brackets, record_property):
    held = brackets[AMPLITUDE]
    record_property("bracket_hz", [held.last_locked, held.first_unlocked])
    assert held.quantity == "from rest"
    assert (held.last_locked, held.first_unlocked) == (LAST_LOCKED, FIRST_UNLOCKED), held.text()
    assert not held.relocks, "no member above step-out locks again"


def test_the_bracket_equals_two_unbatched_runs_at_its_ends(brackets, float64):
    """Each end run alone on a fresh build, not batched: locked at 43.25 Hz and unlocked at 43.5 Hz, as the sweep found."""
    below, above = ratio_of(alone(LAST_LOCKED), LAST_LOCKED), ratio_of(alone(FIRST_UNLOCKED), FIRST_UNLOCKED)
    print(f"\nalone: {LAST_LOCKED} Hz ratio {below:.6f}; {FIRST_UNLOCKED} Hz ratio {above:.6f}")
    assert below >= THRESHOLD > above
    held = brackets[AMPLITUDE]
    assert held.ratios[FREQUENCIES.index(LAST_LOCKED)] >= THRESHOLD > held.ratios[FREQUENCIES.index(FIRST_UNLOCKED)]


def test_control_nine_tenths_of_the_amplitude_moves_the_bracket_below_by_more_than_a_grid_step(brackets,
                                                                                                record_property):
    control = brackets[CONTROL]
    record_property("control_bracket_hz", [control.last_locked, control.first_unlocked])
    assert control.first_unlocked is not None and control.last_locked is not None, control.text()
    assert control.last_locked < LAST_LOCKED - 0.25, control.text()


def test_a_member_is_locked_where_its_mean_spin_follows_the_field(brackets):
    """The reading itself: below the bracket every member turns with the field to within a percent, and above it
    none turns at more than half the field's rate (it rocks, its mean spin a few hertz)."""
    held = brackets[AMPLITUDE]
    cut = FREQUENCIES.index(FIRST_UNLOCKED)
    assert all(abs(r - 1.0) < 0.01 for r in held.ratios[:cut])
    assert all(r < 0.5 for r in held.ratios[cut:])
