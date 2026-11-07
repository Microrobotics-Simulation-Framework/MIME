"""The observed order of convergence from three levels (``mime.results.observed_order``, ``StudyResult.order``): gate 5 of
S2c-1. The oracle is manufactured: ``f(h) = f0 + C h^p`` for p = 1 and p = 2 at three levels, for which the order is p and
the extrapolated value f0 up to rounding, committed in ``predictions_s2c1.json`` as within 1e-9 before the run. Controls:
the first-order plant reads 1 and not 2, on every level set tried; and each way a set of levels can be outside the
asymptotic range (ratios that disagree, a value that crosses its limit, a flat value, an order out of range, a formal order
it is not near) is refused, beside the same call on levels that are in it."""

from __future__ import annotations

import numpy as np
import pytest

from mime.build._refusals import ArgumentRefused
from mime.results import observed_order
from mime.results._order import RATIO_AGREEMENT
from mime.units import TIME, VELOCITY, Quantity

TOLERANCE = 1e-9
LEVELS = [(0.4, 0.2, 0.1), (0.4, 0.2, 0.105), (1.0, 0.5, 0.25), (0.8, 0.41, 0.2), (0.03, 0.015, 0.0075)]
F0, C = 3.7, 0.9


def plant(h, p, f0=F0, c=C):
    return Quantity.of(np.asarray(h, dtype=np.float64), TIME), Quantity.of(f0 + c * np.asarray(h) ** p, VELOCITY)


@pytest.mark.parametrize("p", [1.0, 2.0])
@pytest.mark.parametrize("h", LEVELS)
def test_the_manufactured_order_and_limit_are_recovered(p, h):
    found = observed_order(*plant(h, p))
    assert abs(found.order - p) < TOLERANCE, found.order
    assert abs(float(np.asarray(found.extrapolated.value)) - F0) < TOLERANCE * abs(F0)
    assert found.extrapolated.dimension == VELOCITY and found.levels.tolist() == sorted(h, reverse=True)


def test_levels_given_in_any_order_are_ranked_coarse_to_fine():
    h = (0.1, 0.4, 0.2)
    found = observed_order(*plant(h, 2.0))
    assert abs(found.order - 2.0) < TOLERANCE and found.values.tolist() == sorted(found.values.tolist(), reverse=True)


def test_the_first_order_plant_reads_one_not_two_on_every_level_set():
    for h in LEVELS:
        order = observed_order(*plant(h, 1.0)).order
        assert abs(order - 1.0) < TOLERANCE and abs(order - 2.0) > 0.9, (h, order)


def refused(h, f, **options):
    with pytest.raises(ArgumentRefused) as raised:
        observed_order(Quantity.of(np.asarray(h, dtype=np.float64), TIME), Quantity.of(np.asarray(f, dtype=np.float64), VELOCITY),
                       **options)
    assert raised.value.codes() == ["study.order"]
    return raised.value.refusals[0].message


def test_levels_outside_the_asymptotic_range_are_refused_each_by_what_is_wrong():
    good = plant((0.4, 0.2, 0.1), 2.0)
    assert observed_order(*good).order == pytest.approx(2.0)
    f = good[1].value
    assert "refinement ratios" in refused((0.4, 0.3, 0.1), f) and f"{RATIO_AGREEMENT:.0%}" in refused((0.4, 0.3, 0.1), f)
    assert "one side" in refused((0.4, 0.2, 0.1), [1.0, 1.3, 1.2])  # crosses its limit
    assert "one side" in refused((0.4, 0.2, 0.1), [1.0, 1.0, 1.0])  # does not move
    assert "outside" in refused((0.4, 0.2, 0.1), [2.0, 1.0, 1.0 - 1.0 / 2048])  # the differences give an order of 11
    assert "formal order" in refused((0.4, 0.2, 0.1), f, expected=1.0)
    assert observed_order(Quantity.of(np.array([0.4, 0.2, 0.1]), TIME), Quantity.of(f, VELOCITY), expected=2.0).order \
        == pytest.approx(2.0)
    assert "three distinct positive" in refused((0.4, 0.4, 0.1), f)
    assert "three distinct positive" in refused((0.4, 0.2, 0.1), [1.0, float("nan"), 2.0])
    assert "three levels" in refused((0.4, 0.2, 0.1, 0.05), [1.0, 2.0, 3.0, 4.0])
