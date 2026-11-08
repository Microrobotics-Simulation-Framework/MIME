"""The MRS rung's rule over a declared range, measured: at one spacing, the force points its sampling gives are
monotone in each number its rule declares monotone (``MONOTONE_NUMBERS``), so that the range's corners hold the most
(``Monotone``); and not in the screw's radius, which its rule samples (``Sampled``) instead.

Measured before the rule was written, on 31 values from 0.8 to 1.4 times each number of the ball, the pill and FL-9 at
two spacings (0.3 and 0.6 mm; FL-9 at 1.0 and 1.5 mm): non-decreasing in the sphere's radius, the capsule's radius and
height, and FL-9's length, wavenumber and modulation; FL-9's radius was not monotone at either spacing. Here the ball and
the pill at 0.6 mm on a grid of their own, and a pair of FL-9's radii whose count falls (at 1.5 mm, from 106 at 2.153 mm
to 91 at 2.184 mm), a witness the rule rests on."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from mrs_cases import BALL, PILL
from script_a import fl9

from mime.interactions.hydrodynamic import (
    FORCE_POINTS_MONOTONE,
    FORCE_POINTS_SAMPLED,
    MONOTONE_NUMBERS,
    force_points,
    force_points_rule,
)
from mime.registry import Hull
from mime.units import Quantity

FRAME = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def hull_of(kind: object, ranges: dict[str, tuple[float, float]] | None = None, **settings: float) -> Hull:
    held = {f.name: (getattr(kind, f.name).value if isinstance(getattr(kind, f.name), Quantity) else
                     getattr(kind, f.name)) for f in dataclasses.fields(kind)}
    return Hull("body", "shell", kind, {**held, **settings}, (0.0, 0.0, 0.0), FRAME, ranges or {})


@pytest.mark.parametrize(("kind", "number"), [(BALL, "radius"), (PILL, "radius"), (PILL, "height")],
                         ids=["ball-radius", "pill-radius", "pill-height"])
def test_the_force_points_grow_with_each_number_the_rule_declares_monotone(kind, number):
    assert number in MONOTONE_NUMBERS[kind.name]
    own = float(hull_of(kind).settings[number])
    counts = [force_points(hull_of(kind, **{number: own * factor}), 0.6e-3, tuple(kind.patches))
              for factor in np.linspace(0.8, 1.4, 7)]
    assert np.all(np.diff(counts) >= 0), counts
    assert counts[-1] > counts[0], "the grid spans a change of count, so it could see a fall"


#: Two radii of FL-9 a fiftieth of its radius apart where its force points at 1.5 mm fall (106, then 91; measured).
FALLING = (2.1528e-3, 2.184e-3)


def test_the_screws_radius_is_not_monotone_and_its_rule_samples_it():
    screw = fl9()
    below, above = (force_points(hull_of(screw, radius=radius), 1.5e-3, tuple(screw.patches)) for radius in FALLING)
    assert (below, above) == (106.0, 91.0), "the witness: a larger screw with fewer force points"
    assert "radius" not in MONOTONE_NUMBERS["screw"]
    assert force_points_rule(hull_of(screw, {"radius": (1e-3, 2e-3)})) is FORCE_POINTS_SAMPLED
    assert force_points_rule(hull_of(screw, {"length": (5e-3, 9e-3)})) is FORCE_POINTS_MONOTONE
    assert force_points_rule(hull_of(BALL, {"radius": (1e-3, 2e-3)})) is FORCE_POINTS_MONOTONE
    assert force_points_rule(hull_of(PILL, {"radius": (1e-3, 2e-3), "height": (1e-3, 2e-3)})) is FORCE_POINTS_MONOTONE
