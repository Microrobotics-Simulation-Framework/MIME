"""The ramp down, gated: script A's reproduction arm, float64, dt 0.5 ms, the field held at 100 Hz for 3 s from rest
(the body cannot be captured there: it rocks), then ramped down at 0.25 Hz/s towards 5 Hz (protocol ``Ramp``, the
integrating field 0.2.0). Read from ``PhaseLag``'s spin recorded every step: the windowed mean spin over the mean
frequency the ramp commanded, over 0.2 s windows, against 0.99; the first window centred after the hold at or above it
is where the body is caught, read at the frequency commanded at the window's centre (``ramp_crossings``).

The design's measurement found 31.73 Hz at this rate, step and criterion, good to the rate times half the window, 0.03
Hz; the prediction committed before this ran (``tests/study/predictions_drives.json``) is that figure within that
resolution. The quantity is the ramp down from an unlocked body, never mixed with the step-out from rest (43.25 / 43.5
Hz, ``test_step_out.py``) or the ramp up from a locked one (reported, not gated: far beyond the drag's declared Reynolds
number). The criterion is the design's; it is not adjusted to match.

Its claim, as the step-out from rest's: an acceptance test of the sweep, the ramp and the inertial body, not an
accuracy claim about where a body is caught."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from script_a import reproduction_arm

from mime import Study
from mime.observations import PhaseLag
from mime.scene import X
from mime.study import Ramp, grid, ramp_crossings
from mime.units import Hz, s

#: The design's measurement of the ramp down at 0.25 Hz/s, and its resolution (the rate times half the window).
CAUGHT, RESOLUTION = 31.73, 0.03
RATE, HOLD, WINDOW = 0.25, 3.0, 0.2
#: Long enough for the ramp to reach 29.75 Hz, two hertz below the measured figure.
DURATION = s(284.0)
LAG = "phase_lag:umr"


@pytest.fixture(scope="module")
def crossing():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        experiment = reproduction_arm(frequency=Hz(100.0))
        experiment.observe(PhaseLag("umr", field="field.magnetic", about=X))
        field = experiment.handle("field")
        frequency = field.commands.frequency  # type: ignore[union-attr]
        study = Study(experiment).sweep(
            grid({frequency: [Hz(100.0)]}), duration=DURATION, record=[f"{LAG}.spin"],
            protocol=Ramp(frequency, to=Hz(5.0), rate=Hz(RATE) / s(1.0), hold=s(HOLD)))
        result = study.run()
        (found,) = ramp_crossings(result, LAG, window=s(WINDOW))
        spin = result.member(0)[LAG].spin
        return found, np.asarray(spin.raw, dtype=np.float64)
    finally:
        jax.config.update("jax_enable_x64", held)


def test_the_ramp_down_catches_the_body_where_the_design_measured_it(crossing, record_property):
    found, _ = crossing
    record_property("caught_hz", found.at)
    print(f"\n{found.text()}; the design's measurement {CAUGHT} Hz, its resolution {RESOLUTION} Hz")
    assert found.quantity == "ramp down"
    assert found.at is not None and abs(found.at - CAUGHT) <= RESOLUTION


def test_the_body_is_unlocked_at_the_end_of_the_hold_and_locked_at_the_end_of_the_run(crossing):
    """The reading's two ends: rocking at 100 Hz from rest (its mean spin a small part of the field's), turning with
    the field at the run's end, two hertz below the catch."""
    _, spin = crossing
    count = round(WINDOW / 5e-4)
    held = spin[round(HOLD / 5e-4) - count:round(HOLD / 5e-4)].mean() / 100.0
    end = spin[-count:].mean() / (100.0 - RATE * (DURATION.value - HOLD - WINDOW / 2))
    assert held < 0.1 and end > 0.99
