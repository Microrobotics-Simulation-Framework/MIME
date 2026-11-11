"""Two openings at one plane (F2b's PR 3; predictions in ``predictions_two_openings.json``, committed before they ran),
on the rig of two parallel tubes from one inlet plane (``twin_cases.py``): FB5, the two outlets' split is the 0-D
network's on the rig's own conductances; amendment 5, the shared distal resistance's split unmoved by R_d, its
off-diagonals dropped or lagged each caught; the closed valve against the walled-off vessel, and the stiff resistance
that stands in for it; FB6, the implicit coupling stable where the test-only explicit arm is not. Each with its
control on CI's draw; the controls' rates over fresh draws are measured on Actions (the PR's evidence table)."""

from __future__ import annotations

import pytest
from open_cases import scan
from twin_cases import FB6_LINE, FLUX, PLANTS, LaggedNode, draw, fb6_reading, law_books, split_reading, \
    stand_in_reading, twin_node, valve_passes, valve_reading

from mime.core import ledger as books

#: The draw CI runs (the probes run 40 or more fresh ones a control).
SEED = 0
#: Each dtype's solve tolerance on the rig, and the splits' and the valve's line in it: the law run's split against the
#: network on the rig's conductances, each a steady solve to that tolerance.
TOLERANCE = {"float64": 1e-11, "float32": 1e-5}
PER_TOLERANCE = 100.0


def planted(monkeypatch: pytest.MonkeyPatch, plant: str | None) -> None:
    if plant:
        PLANTS[plant](monkeypatch.setattr)


@pytest.mark.parametrize("swapped", [False, True])
def test_two_outlets_split_as_the_network_on_the_rigs_conductances(dtype, swapped):
    """FB5: a steady inflow's split between two outlets, each a resistance, is the 0-D network's on the rig's own
    conductances (three given-pressure runs) to its line; the control swaps the solver's outlet laws and not the
    network's, which moves the split to its complement."""
    found = split_reading(draw(SEED), TOLERANCE[dtype], swapped=swapped)
    assert (found["errors"][0] <= PER_TOLERANCE * TOLERANCE[dtype]) == (not swapped), found


@pytest.mark.parametrize("plant", [None, "dropped"])
def test_the_shared_distal_split_does_not_move_with_its_distal_resistance(dtype, monkeypatch, plant):
    """Amendment 5: the shared distal law's split is the network's on the rig's conductances, the same at distal
    resistances of 5 and 50 R_t (R_d drops out of the split: both outlets see one distal pressure); the control, its
    off-diagonals dropped (each R_d folded into its own row, its books consistent with it), moves it past the line at
    each."""
    planted(monkeypatch, plant)
    found = split_reading(draw(SEED), TOLERANCE[dtype], distal=(5.0, 50.0))
    assert all((error <= PER_TOLERANCE * TOLERANCE[dtype]) == (plant is None) for error in found["errors"]), found


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


@pytest.mark.parametrize("plant", [None, "leaking", "opened"])
def test_a_closed_valve_is_the_walled_off_vessel(dtype, monkeypatch, plant):
    """The closed valve on the rig's second outlet against the monolithic run with that tube capped by solid: the
    inlet pressures differ by the open tube's resistance times the cap's own leak (measured in the reference), which
    explains the difference to a tenth of it beyond the solves' own; the valve's flux zero to rounding. The controls: a
    valve leaking 2 % of the inflow, and the valve's row a pressure row of zero (an outlet left open)."""
    planted(monkeypatch, plant)
    found = valve_reading(draw(SEED), TOLERANCE[dtype])
    assert valve_passes(found, TOLERANCE[dtype], dtype) == (plant is None), found


def test_the_stiff_stand_in_leaks_as_its_resistance_and_grows_its_conditioning(float64):
    """The stiff resistance standing in for the closed valve (impedance-only's route), reported as the PR measures it:
    its leak falls as one over its resistance, the port system's condition number grows with it, the CG's iterations
    do not move (the port system is solved beside the CG, never inside it)."""
    found = stand_in_reading(draw(SEED), TOLERANCE["float64"], (1e2, 1e4))
    valve, softer, stiffer = found
    assert abs(valve["leak"]) < 1e-12 and 0.0 < stiffer["leak"] < softer["leak"]
    assert abs(softer["leak"] / stiffer["leak"] - 1e2) < 1.0, found
    assert stiffer["condition"] > 10.0 * softer["condition"] > 10.0 * valve["condition"], found
    assert len({entry["iterations"] for entry in found}) == 1, found


@pytest.mark.parametrize("explicit", [False, True])
def test_the_implicit_coupling_is_stable_where_the_explicit_one_is_not(dtype, explicit):
    """FB6: at a loop gain R_p q of 5 on both outlets the implicit coupling stays within its line over two periods; the
    test-only explicit arm (each Windkessel's pressure from the last step's flux) does not."""
    found = fb6_reading(draw(SEED), 5.0, TOLERANCE[dtype], explicit)
    assert (found["largest"] <= FB6_LINE) == (not explicit), found
