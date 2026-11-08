"""The aliasing warning: a recording whose samples are as far apart as half the period of a declared drive's frequency
(or farther) warns, and one recorded often enough does not. A warning, never a refusal. Both cases in one test, so that a
warning that fires for both or for neither fails."""

from __future__ import annotations

import warnings

import pytest
from script_a import reproduction_arm

from mime.experiment._aliasing import aliasing_text
from mime.units import Hz, ms, s


def aliasing(caught) -> list[str]:
    return [str(w.message) for w in caught if "aliasing" in str(w.message)]


def test_ten_milliseconds_against_a_fifty_hertz_drive_warns_and_one_millisecond_does_not():
    built = reproduction_arm(frequency=Hz(50.0), dt=ms(1.0)).build()
    with warnings.catch_warnings(record=True) as sparse:
        warnings.simplefilter("always")
        built.run(s(0.04), record=["swim_speed:umr.speed"], every=ms(10.0))
    with warnings.catch_warnings(record=True) as dense:
        warnings.simplefilter("always")
        built.run(s(0.04), record=["swim_speed:umr.speed"], every=ms(1.0))
    assert len(aliasing(sparse)) == 1 and aliasing(dense) == []
    message = aliasing(sparse)[0]
    assert "50 Hz" in message and "one sample every 0.01 s" in message and "every=s(0.005)" in message


def test_the_warning_is_a_function_of_the_spacing_and_the_fastest_drive_and_is_never_a_refusal():
    import math

    rates = [(2 * math.pi * 10.0, "a 10 Hz drive"), (2 * math.pi * 50.0, "a 50 Hz drive")]
    assert aliasing_text(rates, 0.010, "x") is not None       # at half the period: not enough to show the waveform
    assert aliasing_text(rates, 0.0099, "x") is None          # just under it
    assert aliasing_text(rates, 0.020, "x") is not None and "50 Hz" in aliasing_text(rates, 0.020, "x")
    assert aliasing_text([], 1.0, "x") is None and aliasing_text(rates, 0.0, "x") is None


def test_a_study_warns_for_the_stage_it_keeps_as_for_the_series_it_records():
    from mime import Study
    from mime.study import grid

    exp = reproduction_arm(frequency=Hz(50.0), dt=ms(1.0))
    field = exp.handle("field")
    study = Study(exp).sweep(grid({field.commands.frequency: [Hz(50.0)]}), duration=s(0.04),  # type: ignore[union-attr]
                             record=["swim_speed:umr.speed"], every=ms(1.0), stage_every=ms(10.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study.run()
    messages = aliasing(caught)
    assert len(messages) == 1 and "the study's stage samples" in messages[0]
    with pytest.warns(UserWarning, match="aliasing"):
        reproduction_arm(frequency=Hz(50.0), dt=ms(1.0)).build().run(s(0.04), history=True, every=ms(10.0))
