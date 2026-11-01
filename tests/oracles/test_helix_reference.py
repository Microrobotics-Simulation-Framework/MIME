"""The measured effect of quasi-uniform sampling on de Jongh's FL-9, against a converged free-space reference shipped as
data (``references/helix_fl9_free_space.json``, pinned by its SHA-256, its method, runs and band in the file).

**What it bounds and what it estimates.** The reference bounds the model's free-space limit only under its
extrapolations, by the rule preregistered before any of its solves: 3.060 +- 0.147 mm/s. The rung at the frozen
spacing, 0.16 mm (5,984 samples), and blob 0.5 is held inside that band on the screw 0.3.0's quasi-uniform sampling,
and the screw 0.2.0's quadrature, what the rung took before 7m-a, lies outside it: the control, run beside it, which
fails every run. The rung's error against the post hoc point (3.0573 mm/s) is an estimate, not a bound: -3.43% on the
quasi-uniform sampling, -8.48% on the quadrature, each recorded in the file and reproduced here within
``REPRODUCED``, so that a change of the sampler shows here as a change of that number. Nothing here compares the
model with a measured speed.

The run's precision is the default, float32: the rung's speed there is within ``REPRODUCED`` of the float64 values the
file records (the helix score measured float32 within 1e-6 of float64 on the same body)."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from mrs_cases import free_space, mrs, resistance_at
from script_a import fl9

from mime.scene import KEPT
from mime.units import mm

REFERENCE = Path(__file__).parent / "references" / "helix_fl9_free_space.json"
REFERENCE_SHA256 = "52ff663fa2743e177a6231f0373712c23627802e5a21ae14e71632a1c3f34538"
#: The rung's speed reproduced within this share of the recorded one: float32's rounding over a 17,952-unknown solve
#: and the samples' placement (the sampling's choice is made in float64 whatever the run's precision).
REPRODUCED = 1e-4


@pytest.fixture(scope="module")
def recorded():
    data = REFERENCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == REFERENCE_SHA256, "the reference file is not the one pinned"
    return json.loads(data)


def axial_speed(kind, spacing_mm: float, blob: float, frequency_hz: float) -> float:
    """The rung's axial speed of ``kind`` in free space, its axis along x, turned at ``frequency_hz`` about it."""
    built = free_space("helix", {"screw": kind}, mrs(mm(spacing_mm), blob=blob)).build()
    r = resistance_at(built, "screw", pose=np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    return float(-r[0, 3] * 2.0 * math.pi * frequency_hz / r[0, 0])


def test_the_reference_is_the_bodys_and_its_band_holds_its_post_hoc_point(recorded):
    body = recorded["body"]
    kind = fl9()
    assert (body["radius_m"], body["length_m"], body["wavenumber"], body["modulation"], body["starts"]) == (
        pytest.approx(kind.radius.value), pytest.approx(kind.length.value), kind.wavenumber.value,
        kind.modulation.value, kind.starts)
    low = recorded["preregistered"]["speed_m_per_s"] - recorded["preregistered"]["band_m_per_s"]
    high = recorded["preregistered"]["speed_m_per_s"] + recorded["preregistered"]["band_m_per_s"]
    assert low < recorded["reference"]["speed_m_per_s"] < high


def test_quasi_uniform_sampling_brings_the_frozen_point_inside_the_reference_band_and_the_quadrature_does_not(
        recorded):
    rung = recorded["on_the_rung"]
    spacing_mm, blob, frequency = rung["spacing_m"] * 1e3, rung["blob"], recorded["frequency_hz"]
    low = recorded["preregistered"]["speed_m_per_s"] - recorded["preregistered"]["band_m_per_s"]
    high = recorded["preregistered"]["speed_m_per_s"] + recorded["preregistered"]["band_m_per_s"]
    current = fl9()
    held, _ = KEPT[("screw", "0.2.0")]
    grid = held(radius=current.radius, length=current.length, wavenumber=current.wavenumber,
                modulation=current.modulation, starts=current.starts)
    sampled = axial_speed(current, spacing_mm, blob, frequency)
    gridded = axial_speed(grid, spacing_mm, blob, frequency)
    assert sampled == pytest.approx(rung["quasi_uniform_m_per_s"], rel=REPRODUCED)
    assert gridded == pytest.approx(rung["quadrature_m_per_s"], rel=REPRODUCED)
    assert low <= sampled <= high, f"{sampled * 1e3:.4f} mm/s, outside the reference's preregistered band"
    assert not low <= gridded <= high, "the control: the quadrature's speed lies inside the band, so it cannot tell"
    point = recorded["reference"]["speed_m_per_s"]
    print(f"\nquasi-uniform {sampled * 1e3:.4f} mm/s ({sampled / point - 1:+.2%}), quadrature {gridded * 1e3:.4f} mm/s "
          f"({gridded / point - 1:+.2%}), against {point * 1e3:.4f} mm/s")
