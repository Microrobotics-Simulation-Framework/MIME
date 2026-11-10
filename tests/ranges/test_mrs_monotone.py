"""The MRS rung's rules over a declared range, measured on dense scans. The force points at one spacing never fall as a
number its kind declares monotone grows (``MONOTONE_NUMBERS``), so that the range's corners hold the most
(``ChoiceRule.Monotone``); every other number is sampled inside the range as well (``ChoiceRule.Sampled``). The
family's default spacing is sampled in every number (``SPACING_SAMPLED``): its scans fall, which is why.

The force points' scans, each over a window of the kind's own value (the ball, a 1 mm sphere; the pill, a capsule of
radius 0.5 mm and height 2 mm; FL-9) at two spacings (:data:`FORCE_SCANS`). The claim is the windows': outside them the
rule is declared, not measured, and a value whose own choice is more demanding than the range's is refused as the
entry's defect at every point a study or an override plans (``tests/ranges/test_range_checks.py``), never run coarser.

**The controls**, deterministic (a fixed grid, no draw), so each fails on every run: the same scan over a number the
force points' rule samples finds its fall (FL-9's radius), and the default spacing's scans find theirs (the ball's
radius, where drag 0.4.0's rounding steps ten units under a power of ten, 0.999 to 0.99 mm, a step 0.5.0 does not take;
FL-9's modulation, where the spacing itself falls), so a rule that declared either monotone would fail here; a number declared monotone with no
scan fails the scans' census."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from mrs_cases import BALL, PILL, mrs, uncoupled
from script_a import fl9

from mime.interactions.hydrodynamic import (
    FORCE_POINTS_MONOTONE,
    FORCE_POINTS_SAMPLED,
    MONOTONE_NUMBERS,
    SPACING_SAMPLED,
    force_points,
    force_points_rule,
    spacing_rule,
)
from mime.registry import Hull
from mime.units import Quantity, mm

FRAME = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
KINDS: dict[str, Any] = {"sphere": BALL, "capsule": PILL, "screw": fl9()}
#: The force points' scans, by kind and number: the window (times the kind's own value), the spacings and the values.
FORCE_SCANS: dict[tuple[str, str], tuple[tuple[float, float], tuple[float, ...], int]] = {
    ("sphere", "radius"): ((0.5, 2.0), (0.3e-3, 0.6e-3), 61),
    ("capsule", "radius"): ((0.5, 2.0), (0.3e-3, 0.6e-3), 31),
    ("capsule", "height"): ((0.5, 2.0), (0.3e-3, 0.6e-3), 31),
    ("screw", "length"): ((0.7, 1.6), (1.0e-3, 1.5e-3), 21),
    ("screw", "wavenumber"): ((0.7, 1.6), (1.0e-3, 1.5e-3), 21),
    ("screw", "modulation"): ((0.7, 1.6), (1.0e-3, 1.5e-3), 21),
}
#: The controls' scans: FL-9's radius in the force points at 1.5 mm; the ball's radius and FL-9's modulation in the
#: default spacing (each window, values), with the first fall each finds: its value and the spacings either side, mm.
FORCE_CONTROL = (("screw", "radius"), (0.7, 1.6), 1.5e-3, 21)
SPACING_CONTROLS = {("sphere", "radius"): ((0.2, 5.0), 400, (3.9654, 0.996, 0.990)),
                    ("screw", "modulation"): ((0.7, 1.6), 200, (0.2310, 0.817, 0.816))}


def hull_of(kind: Any, ranges: dict[str, tuple[float, float]] | None = None, **settings: float) -> Hull:
    held = {f.name: (getattr(kind, f.name).value if isinstance(getattr(kind, f.name), Quantity) else
                     getattr(kind, f.name)) for f in dataclasses.fields(kind)}
    return Hull("body", "shell", kind, {**held, **settings}, (0.0, 0.0, 0.0), FRAME, ranges or {})


def scanned(kind_name: str, number: str, window: tuple[float, float], count: int, measure: Any) -> np.ndarray:
    """``measure`` of the kind at ``count`` values of ``number`` evenly over ``window`` times its own."""
    kind = KINDS[kind_name]
    own = float(hull_of(kind).settings[number])
    return np.asarray([measure(hull_of(kind, **{number: float(own * factor)}))
                       for factor in np.linspace(window[0], window[1], count)])


def falls(values: np.ndarray) -> list[int]:
    """The indices at which the scanned measure falls from one value to the next."""
    return [int(i) for i in np.nonzero(np.diff(values) < 0)[0]]


def force_scan(kind_name: str, number: str, window: tuple[float, float], spacing: float, count: int) -> np.ndarray:
    patches = tuple(KINDS[kind_name].patches)
    return scanned(kind_name, number, window, count, lambda hull: force_points(hull, spacing, patches))


def spacing_scan(kind_name: str, number: str, window: tuple[float, float], count: int,
                 entry: Any | None = None) -> np.ndarray:
    entry = mrs(mm(0.6)) if entry is None else entry
    return scanned(kind_name, number, window, count, entry._family_spacing)


def declared(numbers: Any) -> list[tuple[str, str]]:
    return sorted((kind, number) for kind, held in numbers.items() for number in held)


def test_every_number_declared_monotone_has_its_scan_and_every_scan_a_declaration():
    assert sorted(FORCE_SCANS) == declared(MONOTONE_NUMBERS)


@pytest.mark.parametrize(("kind", "number", "spacing"),
                         [(kind, number, spacing) for (kind, number), (_, spacings, _) in FORCE_SCANS.items()
                          for spacing in spacings],
                         ids=lambda value: f"{value * 1e3:g}mm" if isinstance(value, float) else str(value))
def test_the_force_points_never_fall_as_a_number_declared_monotone_grows(kind, number, spacing):
    window, _, count = FORCE_SCANS[(kind, number)]
    counts = force_scan(kind, number, window, spacing, count)
    assert falls(counts) == [], counts.tolist()
    if (kind, number) != ("screw", "wavenumber"):
        assert counts[-1] > counts[0], "the window spans a change of count, so the scan could see a fall"


def test_control_the_scan_finds_the_screws_radius_falling_in_the_force_points_so_its_rule_samples_it():
    (kind, number), window, spacing, count = FORCE_CONTROL
    assert falls(force_scan(kind, number, window, spacing, count)) != [], "a rule declaring it monotone fails"
    assert number not in MONOTONE_NUMBERS[kind]
    assert force_points_rule(hull_of(KINDS[kind], {number: (1e-3, 2e-3)})) is FORCE_POINTS_SAMPLED


@pytest.mark.parametrize(("kind", "number"), sorted(SPACING_CONTROLS), ids=lambda value: str(value))
def test_control_the_default_spacing_falls_as_the_body_grows_so_its_rule_samples_every_number(kind, number):
    window, count, (at, before, after) = SPACING_CONTROLS[(kind, number)]
    # the ball's fall is drag 0.4.0's rounding, kept as released (0.5.0 takes the exact step: the test below)
    spacings = spacing_scan(kind, number, window, count, uncoupled(mm(0.6)) if kind == "sphere" else None)
    found = falls(spacings)
    assert found != [], "a rule declaring it monotone fails"
    own = float(hull_of(KINDS[kind]).settings[number])
    value = own * np.linspace(window[0], window[1], count)[found[0]] * (1e3 if number == "radius" else 1.0)
    assert (round(value, 4), round(spacings[found[0]] * 1e3, 3), round(spacings[found[0] + 1] * 1e3, 3)) == \
        (at, before, after)
    assert spacing_rule(hull_of(KINDS[kind], {number: (own, 2.0 * own)})) is SPACING_SAMPLED


def test_the_default_spacing_steps_ten_units_under_a_power_of_ten_on_the_released_drags_and_one_unit_on_0_5_0():
    """The witness of the rounding: on the ball the bisection's boundary lies at 0.99953 mm at a radius of 3.9785 mm,
    the spacing to three figures, 1.00 mm, gives too few samples of the family's 338, and drag 0.4.0 steps a unit of the
    rounded value's third figure down, to 0.99 mm, where a radius of 3.976 mm (its boundary 0.99918 mm) gives 0.999 mm.
    The dip lies where the boundary is in the last half unit under 1 mm: measured, 0.999 mm at 3.977 mm and 0.99 mm
    from 3.9775 to 3.979 mm. Drag 0.5.0 steps from the boundary: 0.999 mm at both radii."""
    released: Any = uncoupled(mm(0.6))
    current: Any = mrs(mm(0.6))
    radii = (3.976e-3, 3.977e-3, 3.9785e-3)
    assert [released._family_spacing(hull_of(BALL, radius=radius)) for radius in radii] == [0.999e-3, 0.999e-3, 0.99e-3]
    assert [current._family_spacing(hull_of(BALL, radius=radius)) for radius in radii] == [0.999e-3] * 3


def test_the_balls_default_spacing_does_not_fall_as_it_grows_on_0_5_0_over_the_window_the_control_scans():
    """The first fall drag 0.4.0's scan finds (``SPACING_CONTROLS``) is the rounding's; 0.5.0's scan of the same
    window, on the same 400 values, finds none (a grid's finding, not a proof for every radius)."""
    window, count, _ = SPACING_CONTROLS[("sphere", "radius")]
    assert falls(spacing_scan("sphere", "radius", window, count)) == []


#: Two radii of FL-9 a fiftieth of its radius apart where its force points at 1.5 mm fall (106, then 91; measured).
FALLING = (2.1528e-3, 2.184e-3)


def test_the_screws_radius_falls_between_two_radii_a_fiftieth_apart_the_witness_its_rule_rests_on():
    screw = KINDS["screw"]
    below, above = (force_points(hull_of(screw, radius=radius), 1.5e-3, tuple(screw.patches)) for radius in FALLING)
    assert (below, above) == (106.0, 91.0), "the witness: a larger screw with fewer force points"


def test_the_force_points_rule_is_monotone_only_over_numbers_declared_so():
    screw = KINDS["screw"]
    assert force_points_rule(hull_of(screw, {"length": (5e-3, 9e-3), "modulation": (0.3, 0.4)})) is \
        FORCE_POINTS_MONOTONE
    assert force_points_rule(hull_of(screw, {"length": (5e-3, 9e-3), "radius": (1e-3, 2e-3)})) is FORCE_POINTS_SAMPLED
    assert force_points_rule(hull_of(BALL, {"radius": (1e-3, 2e-3)})) is FORCE_POINTS_MONOTONE
    assert force_points_rule(hull_of(PILL, {"radius": (1e-3, 2e-3), "height": (1e-3, 2e-3)})) is FORCE_POINTS_MONOTONE
