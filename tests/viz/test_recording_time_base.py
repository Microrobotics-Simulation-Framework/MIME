"""The recording layer's time base, on both writers (``run_to(..., stage_every=)`` and a live session's ``record``), at
two spacings each. The invariant: a sample's time code is its index, ``timeCodesPerSecond`` and ``framesPerSecond`` are
the samples per simulated second, and so, within one uninterrupted stretch of samples, **a time code over
``timeCodesPerSecond`` is the simulated seconds since the layer's first sample**, which the layer data names (``steps``
holds every sample's step). The first sample is the state where recording began: the run's start for ``run_to``, the
step the ``record`` command took effect at for a live session, so the codes give seconds from the run's start only
where recording began there. A session that turns recording off and on joins its stretches, the codes staying
consecutive: across the gap the codes are not seconds, and ``steps`` is what names each sample's step. The control: the
same layer with its rate planted wrong is found by the check every time, as is the joined stretches' gap."""

import math

import pytest
from live_cases import built
from pxr import Sdf

from mime.units import ms
from mime.viz import RECORDING_FILE

DT = 0.001  # the rig's base step (s), which live_cases.built() has


def layer_of(folder) -> Sdf.Layer:
    layer = Sdf.Layer.FindOrOpen(str(folder / RECORDING_FILE))
    assert layer is not None
    return layer


def time_base_problems(layer: Sdf.Layer, dt: float = DT) -> list[str]:
    """Every way the layer's time base differs from the invariant: the two rates equal, one time sample at each index,
    and each code over the rate the simulated seconds since the first sample's step."""
    data = layer.customLayerData["mime:recording"]
    steps = list(data["steps"])
    rate = layer.timeCodesPerSecond
    found: list[str] = []
    if layer.framesPerSecond != rate:
        found.append(f"framesPerSecond {layer.framesPerSecond} is not timeCodesPerSecond {rate}")
    if data["samples"] != len(steps):
        found.append(f"{data['samples']} samples named, {len(steps)} steps")
    if sorted(layer.ListAllTimeSamples()) != [float(code) for code in range(len(steps))]:
        found.append("the time samples are not at the indices 0..n-1")
    for code, step in enumerate(steps):
        if not math.isclose(code / rate, (step - steps[0]) * dt, rel_tol=1e-9, abs_tol=1e-12):
            found.append(f"code {code} is {code / rate} s, step {step} is {(step - steps[0]) * dt} s after the first")
    return found


def with_rate(layer: Sdf.Layer, factor: float) -> Sdf.Layer:
    """A copy of ``layer`` with both rates times ``factor``, which leaves the layer on disk as it is."""
    copy = Sdf.Layer.CreateAnonymous(".usda")
    copy.TransferContent(layer)
    copy.timeCodesPerSecond = copy.framesPerSecond = layer.timeCodesPerSecond * factor
    return copy


@pytest.mark.parametrize("every", [6, 12])
def test_a_run_tos_codes_over_the_rate_are_seconds_from_the_runs_start(every, tmp_path):
    written = built().run_to(tmp_path / "run", ms(120), stage_every=ms(every))
    layer = layer_of(tmp_path / "run")
    assert written.recording is not None
    steps = list(layer.customLayerData["mime:recording"]["steps"])
    assert steps == list(range(0, 121, every)) and layer.timeCodesPerSecond == pytest.approx(1.0 / (every * DT))
    assert time_base_problems(layer) == []
    assert time_base_problems(with_rate(layer, 2.0)) != [] and time_base_problems(with_rate(layer, 0.5)) != []


@pytest.mark.parametrize("chunk", [4, 6])
def test_a_live_sessions_codes_over_the_rate_are_seconds_from_where_it_began_recording(chunk, tmp_path):
    session = built().live(chunk=ms(chunk), pace=None, out=tmp_path / "live")
    session.step(2)
    session.record(True)
    session.step(5)
    session.stop()
    layer = layer_of(tmp_path / "live")
    steps = list(layer.customLayerData["mime:recording"]["steps"])
    # From the step recording began at, not from the run's start.
    assert steps == [chunk * index for index in range(2, 8)]
    assert layer.timeCodesPerSecond == pytest.approx(1.0 / (chunk * DT))
    assert time_base_problems(layer) == []
    assert time_base_problems(with_rate(layer, 2.0)) != [] and time_base_problems(with_rate(layer, 0.5)) != []


def test_a_session_that_turns_recording_off_and_on_names_the_gap_in_its_steps_not_its_codes(tmp_path):
    chunk = 4
    session = built().live(chunk=ms(chunk), pace=None, out=tmp_path / "live")
    session.step(1)
    session.record(True)
    session.step(2)
    session.record(False)
    session.step(2)
    session.record(True)
    session.step(1)
    session.stop()
    layer = layer_of(tmp_path / "live")
    steps = list(layer.customLayerData["mime:recording"]["steps"])
    assert steps == [4, 8, 12, 20, 24]
    problems = time_base_problems(layer)
    assert len(problems) == 2 and problems[0].startswith("code 3 ") and problems[1].startswith("code 4 "), problems
