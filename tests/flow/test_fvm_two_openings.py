"""Two openings at one plane (F2b's PR 3; predictions in ``predictions_two_openings.json``, committed before they ran,
and in its fix round's ``predictions_two_openings_d.json``), on the rig of two parallel tubes from one inlet plane
(``twin_cases.py``), every run metered and its books judged: FB5, the two outlets' split is the 0-D network's on the
rig's own conductances; amendment 5, the shared distal resistance's split unmoved by R_d, its off-diagonals dropped or
lagged each caught; the closed valve against the walled-off vessel at the default stiff interior, its first clause
alone against a leak above its resolution at each interior, and the stiff resistance that stands in for it; FB6, the
implicit coupling stable where the test-only explicit arm is not. Each with its control on CI's draw; the controls'
rates over fresh draws are measured on Actions (the PR's evidence table)."""

from __future__ import annotations

import pytest
from open_cases import scan
from twin_cases import FLUX, LINE_LEAK, PLANTS, VALVE_INTERIORS, LaggedNode, closes, draw, fb6_passes, fb6_reading, \
    law_books, leaking, split_passes, split_reading, stand_in_reading, twin_node, valve_line_holds, valve_passes, \
    valve_reading

from mime.core import ledger as books

#: The draw CI runs (the probes run 40 or more fresh ones a control).
SEED = 0
#: Each dtype's solve tolerance on the rig (the splits' and the valve's lines are in it).
TOLERANCE = {"float64": 1e-11, "float32": 1e-5}


def planted(monkeypatch: pytest.MonkeyPatch, plant: str | None) -> None:
    if plant:
        PLANTS[plant](monkeypatch.setattr)


@pytest.mark.parametrize("plant", [None, "swapped", "patch_force"])
def test_two_outlets_split_as_the_network_on_the_rigs_conductances(dtype, monkeypatch, plant):
    """FB5: a steady inflow's split between two outlets, each a resistance, is the 0-D network's on the rig's own
    conductances (three given-pressure runs) to its line, every run's books closing; the controls: the solver's outlet
    laws swapped and not the network's, which moves the split to its complement; and the reviewer's mutant, one patch's
    force at 1.01 times its pressure, which the step identity fails."""
    planted(monkeypatch, plant if plant != "swapped" else None)
    found = split_reading(draw(SEED), TOLERANCE[dtype], swapped=plant == "swapped")
    assert split_passes(found, TOLERANCE[dtype]) == (plant is None), found
    assert found["closes"] == (plant != "patch_force"), found


@pytest.mark.parametrize("plant", [None, "dropped"])
def test_the_shared_distal_split_does_not_move_with_its_distal_resistance(dtype, monkeypatch, plant):
    """Amendment 5: the shared distal law's split is the network's on the rig's conductances, the same at distal
    resistances of 5 and 50 R_t (R_d drops out of the split: both outlets see one distal pressure), every run's books
    closing; the control, its off-diagonals dropped (each R_d folded into its own row, its books consistent with it),
    moves it past the line at each."""
    planted(monkeypatch, plant)
    found = split_reading(draw(SEED), TOLERANCE[dtype], distal=(5.0, 50.0))
    assert split_passes(found, TOLERANCE[dtype]) == (plant is None), found
    assert found["closes"], found


#: The deliveries a metered shared-distal rig takes as the build makes them.
METERED_SHARED = (
    books.LedgerDelivery("viscous", "flow.viscosity", books.DISSIPATIVE, books.HOSTED).encoded(),
    books.LedgerDelivery("penalty", "tube.wall", books.DISSIPATIVE, books.HOSTED).encoded(),
    books.LedgerDelivery("opening_0", "inflow:tube.start", books.UNMETERED, books.HOSTED, "",
                         "inflow:tube.start").encoded(),
    books.LedgerDelivery("opening_1", "shared_distal:tube.end_0+tube.end_1", books.DISSIPATIVE,
                         books.HOSTED).encoded(),
    books.LedgerDelivery("kinetic_flux", "flow.openings", books.UNMETERED, books.HOSTED).encoded(),
)


def shared_pulsatile(lagged: bool, dt: float = 0.05):
    """The rig's shared distal law under a pulsatile inflow (40 steps a period), IMEX Euler, metered."""
    resistance = 72.0
    node = twin_node(draw(SEED), ("inflow", "shared_distal"),
                     [[FLUX, 0.3 * FLUX, 1.0 / (40.0 * 0.05), 0.0], [resistance, 3.0 * resistance, 5.0 * resistance, 0.0]],
                     law_openings=((0,), (1, 2)), ids=("inflow:tube.start", "shared_distal:tube.end_0+tube.end_1"),
                     dt=dt, scheme="imex_euler", convection=False, ledger=METERED_SHARED)
    if lagged:
        node.__class__ = LaggedNode
    return node


@pytest.mark.parametrize("lagged", [False, True])
def test_the_shared_distal_laws_books_close_and_a_lagged_off_diagonal_does_not(dtype, lagged):
    """Amendment 5, control (b): in pulsatile flow the shared distal law's own books close at every step on its own
    scale; its off-diagonals lagged a step (their coupling taken at the last step's fluxes) leave a residual in them, of
    first order in dt (the probe measures its order)."""
    final, _ = scan(shared_pulsatile(lagged), 20)
    assert (law_books(final) <= 1.0) == (not lagged), law_books(final)
    assert closes(final) == (not lagged)


@pytest.mark.parametrize("plant", [None, "leaking", "opened"])
def test_a_closed_valve_is_the_walled_off_vessel(dtype, monkeypatch, plant):
    """The closed valve on the rig's second outlet against the monolithic run with that tube capped by solid, at the
    default stiff interior (0.1h): the inlet pressures differ by no more than the two runs' own solid flows at the
    closed outlet carry through the open tube's resistance (the cap's leak; the closed patch's exchange through the
    solid with its neighbour, 1.5 to 5 % of the inflow there), each measured in its run, beyond the solves' own; the
    valve's flux zero to rounding; every run's books closing. The controls: a valve leaking 2 % of the inflow, and the
    valve's row a pressure row of zero (an outlet left open)."""
    planted(monkeypatch, plant)
    found = valve_reading(draw(SEED), TOLERANCE[dtype])
    assert valve_passes(found, TOLERANCE[dtype], dtype) == (plant is None), found


@pytest.mark.parametrize("interior", VALVE_INTERIORS)
def test_the_valve_line_alone_sees_a_leak_above_its_resolution(dtype, monkeypatch, interior):
    """The valve gate's first clause on its own (the inlet pressures against the two runs' solid flows), with the
    valve's flux clause set aside: the clean valve passes the whole gate and a valve leaking ``LINE_LEAK`` of the
    inflow (above what the line resolves at that interior: a leak lowers the inlet pressure by about its own share)
    fails the line."""
    found = valve_reading(draw(SEED), TOLERANCE[dtype], interior)
    assert valve_passes(found, TOLERANCE[dtype], dtype), found
    leaking(LINE_LEAK[interior])(monkeypatch.setattr)
    found = valve_reading(draw(SEED), TOLERANCE[dtype], interior)
    assert not valve_line_holds(found, TOLERANCE[dtype]), found


def test_the_stiff_stand_in_differs_from_the_valve_only_by_its_leak(float64):
    """The stiff resistance standing in for the closed valve (impedance-only's route), as the PR measures it: its leak
    falls as one over its resistance (at 1e4 and 1e6 R_t, past the 1e2 where the leak still lowers the pressure behind
    it by a few per cent); the port system's condition number with its rows scaled is the valve's (the raw number mixes
    a flux row's units with a pressure row's, and moves with them); the CG's iterations do not move (the port system is
    solved beside the CG, never inside it); the books close."""
    found = stand_in_reading(draw(SEED), TOLERANCE["float64"], (1e4, 1e6))
    valve, softer, stiffer = found
    assert abs(valve["leak"]) < 1e-12 and 0.0 < stiffer["leak"] < softer["leak"]
    assert abs(softer["leak"] / stiffer["leak"] - 1e2) < 1.0, found
    assert all(abs(entry["scaled_condition"] / valve["scaled_condition"] - 1.0) < 0.5 for entry in found), found
    assert len({entry["iterations"] for entry in found}) == 1, found
    assert all(entry["closes"] for entry in found), found


@pytest.mark.parametrize("explicit", [False, True])
def test_the_implicit_coupling_is_stable_where_the_explicit_one_is_not(dtype, explicit):
    """FB6: at a loop gain R_p q of 5 on both outlets the implicit coupling stays within its line over two periods, its
    books closing; the test-only explicit arm (each Windkessel's pressure from the last step's flux) does not."""
    found = fb6_reading(draw(SEED), 5.0, TOLERANCE[dtype], explicit)
    assert fb6_passes(found) == (not explicit), found
    assert found["closes"] or explicit, found
