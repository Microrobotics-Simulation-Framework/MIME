"""The near-wall check (MIME-ANO-114's first step) as recorded, and the edge the generator's domain declares from it.

``references/near_wall_check.json`` is the check ``mime.surrogates.generator`` declares (``NEAR_WALL_*``): FL-9 resting
on the floor of the 1/4 in tube, its closed surface, at six spacings and nine gaps, through ``Drag.BEM`` in the tube
wall kind (the shipped table), in float64 on Modal CPU containers (``scripts/surrogate/modal_app.py``), with the commit
and the table's sha256 recorded. ``wall_excess`` takes, at each spacing but the finest, the error at each gap beyond
the error at the far gap (the body's own discretisation error, MIME-ANO-116, which the wall does not add to), the
finest spacing as the reference; ``near_wall_ratio`` is the smallest gap-to-spacing ratio above which that excess stays
within the tolerance at every spacing. Each use is judged on its quantities (``NEAR_WALL_USES``): the whole resistance
on all eight, the criterion declared before the check ran; the swim speed on itself, read from the same data after.

- The record is the declared check, every read inside the table, and its reference was never resolved better than
  at the levels it judges (the finest is at least as many spacings from the wall as any level judged).
- **The domain's edges are the record's**: ``NEAR_WALL_RATIOS`` equals ``near_wall_ratios(record)``, use by use.
- **The check discriminates**: the excess passes the tolerance at the smallest gaps, so an edge exists to be found;
  a check whose excess stayed below the tolerance everywhere would have found none, which is no evidence.
- **The control**, failing on every run: the record with the axial coupling raised by five times the tolerance just
  above the edge moves the edge, so the hold of the declared edge to the record would fail."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mime import assets
from mime.surrogates import generator as g

RECORD = json.loads((Path(__file__).parent / "references" / "near_wall_check.json").read_text())


def test_the_record_is_the_declared_check_read_inside_the_shipped_table():
    assert (RECORD["design"], RECORD["tube"]) == (g.NEAR_WALL_DESIGN, g.NEAR_WALL_TUBE)
    assert RECORD["spacings"] == list(g.NEAR_WALL_SPACINGS) and RECORD["gaps"] == list(g.NEAR_WALL_GAPS)
    assert RECORD["tolerance"] == g.NEAR_WALL_TOLERANCE
    assert RECORD["table"] == assets.reference("unit_tube_R0.997").sha256
    assert RECORD["x64"] is True and len(RECORD["samples"]) == len(g.NEAR_WALL_SPACINGS) * len(g.NEAR_WALL_GAPS)
    assert all(each["accepted"] for each in RECORD["samples"])
    for each in RECORD["samples"]:
        held = np.asarray(each["resistance"])
        assert np.all(np.isfinite(held)) and np.max(np.abs(held - held.T)) <= 1e-8 * np.max(np.abs(held))


def test_the_recorded_excess_and_edges_are_what_the_analysis_reads():
    """The excess within rounding (a swim speed is a linear solve, whose last digits differ between LAPACK builds:
    the values are ratios of order one, so 1e-9 is far above that and far below the tolerance), the edges exactly."""
    for use, quantities in g.NEAR_WALL_USES.items():
        found, recorded = g.wall_excess(RECORD, quantities), RECORD["excess"][use]
        assert found.keys() == recorded.keys()
        for spacing, row in found.items():
            assert row.keys() == recorded[spacing].keys()
            for gap, value in row.items():
                assert math.isclose(value, recorded[spacing][gap], rel_tol=1e-9, abs_tol=1e-12), (use, spacing, gap)
    assert g.near_wall_ratios(RECORD) == RECORD["ratios"]


def test_the_domains_edges_are_the_records():
    for use, excess in RECORD["excess"].items():
        print(f"\n{use}: near-wall edge {RECORD['ratios'][use]:.3f} spacings; the excess by spacing and gap:")
        for spacing, row in excess.items():
            print(f"  {float(spacing) * 1e3:.3f} mm: " + ", ".join(f"{float(gap) * 1e3:g} mm {value:.2e}"
                                                                 for gap, value in row.items()))
    assert dict(g.NEAR_WALL_RATIOS) == RECORD["ratios"] and all(np.isfinite(list(RECORD["ratios"].values())))


def test_at_every_spacing_checked_the_whole_resistances_edge_lies_past_the_largest_margin():
    """What the domain's ``outside`` states: the whole resistance is unresolved over the contact margin even at the
    finest spacing the check reached, while the swim speed's edge lies inside it."""
    finest = min(g.NEAR_WALL_SPACINGS)
    assert g.DOMAIN.edge(finest, "resistance") > g.MARGIN[1] > g.DOMAIN.edge(finest, "swim_speed")


@pytest.mark.parametrize("use", sorted(g.NEAR_WALL_USES))
def test_the_check_discriminates_the_excess_passes_the_tolerance_near_the_wall(use: str):
    smallest = repr(min(g.NEAR_WALL_GAPS))
    worst = max(row[smallest] for row in RECORD["excess"][use].values())
    assert worst > g.NEAR_WALL_TOLERANCE, "an excess below the tolerance everywhere finds no edge: no evidence"
    assert all(row[repr(max(g.NEAR_WALL_GAPS))] == 0.0 for row in RECORD["excess"][use].values())


def planted_above_the_edge(use: str) -> dict:
    """The record with the axial coupling (the force along the axis per unit spin about it, which the swim speed is
    proportional to), at a spacing but the finest whose gaps reach past the use's edge, at the first gap past it,
    raised by five times the tolerance: the excess there passes it for either use, whatever that level's own error."""
    held = json.loads(json.dumps(RECORD))
    ratio = RECORD["ratios"][use]
    far = max(g.NEAR_WALL_GAPS)
    spacing, above = min((gap / spacing, spacing, gap) for spacing in g.NEAR_WALL_SPACINGS[:-1]
                         for gap in g.NEAR_WALL_GAPS if gap != far and gap / spacing >= ratio * (1 - 1e-12))[1:]
    for each in held["samples"]:
        if each["spacing"] == spacing and each["gap"] == above:
            resistance = np.asarray(each["resistance"])
            for row, column in ((2, 5), (5, 2)):
                resistance[row, column] *= 1.0 + 5.0 * g.NEAR_WALL_TOLERANCE
            each["resistance"] = resistance.tolist()
    return held


@pytest.mark.parametrize("use", sorted(g.NEAR_WALL_USES))
def test_control_an_excess_planted_above_the_edge_moves_it(use: str):
    planted = g.near_wall_ratio(planted_above_the_edge(use), quantities=g.NEAR_WALL_USES[use])
    print(f"\n{use}: the edge moves from {RECORD['ratios'][use]:.3f} to {planted:.3f} spacings")
    assert planted > RECORD["ratios"][use]
