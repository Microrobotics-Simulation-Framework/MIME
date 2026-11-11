"""The open box's books (F2b's PR 2; predictions in ``predictions_open_ports.json``): FB13, IMEX Euler's step identity
closes with open faces (``ledger_closes``), the kinetic-energy flux booked under the openings' line and not zero on
these draws; its control, an interior gift; ``N_proj`` booked at rounding; ``N_WK``'s order in dt (``term_orders``),
its control a physical loss written into it; the open box's flag declared with its measured line and rate
(``flag_problems`` holds none active without evidence), and no pending flag declared (one not built is not
declared)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from open_cases import PLANTS, books_of, draw, n_wk_sum, point_node, scan

from mime.core import ledger as books
from mime.nodes.flow.fvm import OpenFlowNode, Windkessel
from mime.nodes.flow.fvm._flags import OPEN_FLAGS
from mime.testing import flag_problems, ledger_closes, term_orders

SEEDS = (0, 3)
STEPS = 30


@pytest.mark.parametrize("plant", [None, "interior_gift"])
def test_the_books_close_with_open_faces(dtype, monkeypatch, plant):
    """FB13: the identity closes at every step to its floor with the openings' work, the kinetic-energy flux and N_proj
    booked, on drawn pulsatile runs through a narrowing (the flux is not zero: the gate can fail); an unbooked body
    force on the fluid is caught."""
    if plant:
        PLANTS[plant](monkeypatch.setattr)
    for seed in SEEDS:
        final, _ = scan(point_node(draw(seed)), STEPS)
        result = ledger_closes(final)
        assert result.passed == (plant is None), (seed, result.detail)
        assert books_of(final)["kinetic_flux"] != 0.0


def test_n_proj_is_rounding_at_two_tolerances(float64):
    """N_proj at the open faces is rounding (the half-weight faces keep the projector orthogonal) at the default
    tolerance and one a hundred times looser: it does not fall with the tolerance, so no order is declared for it."""
    point = draw(0)
    for tolerance in (1e-8, 1e-6):
        final, _ = scan(point_node(point, tolerance=tolerance), STEPS)
        index = OpenFlowNode.LEDGER_TERMS.index("projection")
        projection = abs(float(np.asarray(final[f"{books.RECORDS}terms"])[index]))
        assert projection <= 1e-13 * float(final[f"{books.RECORDS}scale"]), (tolerance, projection)


def test_n_wk_has_its_order(float64):
    """N_WK summed over a fixed time falls as dt halves at order 2 (``term_orders`` on one ensemble of three drawn
    cases at D/h 8)."""
    cases = [n_wk_sum(dataclasses.replace(draw(seed), cells=8)) for seed in (0, 1, 2)]
    report = term_orders(cases, ["windkessel"], Windkessel.orders)
    assert report.passed, report.failures()


def test_control_a_loss_written_into_n_wk_fails_its_order(float64, monkeypatch):
    PLANTS["loss_in_n_wk"](monkeypatch.setattr)
    cases = [n_wk_sum(dataclasses.replace(draw(seed), cells=8)) for seed in (0, 1, 2)]
    assert not term_orders(cases, ["windkessel"], Windkessel.orders).passed


def test_the_open_flags_are_declared_and_none_claims_coverage_unshown():
    assert not flag_problems(OPEN_FLAGS)
    assert [flag.name for flag in OPEN_FLAGS] == ["open_face_projection"]
    assert all(flag.active and not flag.pending for flag in OPEN_FLAGS)
    assert OpenFlowNode.LEDGER_FLAGS is OPEN_FLAGS


@pytest.mark.parametrize("plant", [None, "weights_uniform"])
def test_the_projection_flag_is_quiet_and_catches_its_plant(dtype, monkeypatch, plant):
    """The open_face_projection flag on a drawn run: under its line clean, past it with the planes' faces at full
    weight in the inner product (the projector not orthogonal under the books' product). Its rate over fresh draws is
    its evidence's."""
    if plant:
        PLANTS[plant](monkeypatch.setattr)
    flag = next(f for f in OPEN_FLAGS if f.name == "open_face_projection")
    node = point_node(draw(SEEDS[0]))
    final, found = scan(node, STEPS, lambda p, b, a: flag.reading(node, p, b, a))
    assert int(final["stopped_at"]) < 0
    worst = float(np.max(np.abs(np.asarray(found))))
    assert (worst <= flag.line[dtype]) == (plant is None), worst
