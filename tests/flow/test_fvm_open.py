"""The open box's gates (F2b's PR 2; predictions in ``predictions_open_ports.json``, committed before they ran): FB1,
the open tube is the periodic tube; FB2, the inflow's face flux is its waveform to its derived floor; FB3, the
Windkessel's own books close on its own scale; FB4, the step's superposed solution is the monolithic one at its solved
pressures. Each with its control, a planted error caught on these draws; the controls' rates over fresh draws are
measured on Actions (the PR's evidence table). The draws reach a downstream resistance of 1e4 times the vessel's, and
the gates hold there: a relative pressure's move is never formed as a difference of absolute pressures."""

from __future__ import annotations

import numpy as np
import pytest
from fvm_cases import in_dtype
from open_cases import FB1_LINE, PLANTS, books_of, draw, fb1_difference, fb1_pair, inflow_reading, point_node, scan, \
    superposed_reading

#: The draws CI runs (the probes run 40 or more fresh ones a control).
SEEDS = (0, 3)
STEPS = 30


def planted(monkeypatch: pytest.MonkeyPatch, plant: str | None) -> None:
    if plant:
        PLANTS[plant](monkeypatch.setattr)


@pytest.mark.parametrize("plant", [None, "gradient_full_cell"])
def test_the_open_tube_is_the_periodic_tube(dtype, monkeypatch, plant):
    """FB1: the open tube with its ends at given pressures ``G L`` apart, steady Stokes, equals the periodic tube
    driven by the uniform force ``G`` (on the solid too: a pressure drop is), to the solve's tolerance, off the
    cells' centre; the control, the planes' faces' gradient at a full cell, moves the open arm past the line."""
    planted(monkeypatch, plant)
    opened, periodic = fb1_pair(8, offset=(0.3, -0.2), tolerance=1e-11 if dtype == "float64" else 1e-5)
    difference = fb1_difference(opened, periodic)
    assert opened["velocity_z"].shape[-1] == periodic["velocity_z"].shape[-1] + 1  # the open arm's planes are open
    assert (difference <= FB1_LINE[dtype]) == (plant is None), difference


@pytest.mark.parametrize("plant", [None, "inflow_scaled"])
def test_the_inflows_face_flux_is_its_waveform(dtype, monkeypatch, plant):
    """FB2: at every step the start's face flux is the inflow's waveform to its floor, derived from the roundings it
    carries (the face count's summation over the superposition's scale and the waveform's own; ``inflow_reading``),
    never fitted; the control, a law that delivers 1e-3 more, is past it."""
    planted(monkeypatch, plant)
    for seed in SEEDS:
        node = point_node(draw(seed))
        final, readings = scan(node, STEPS, inflow_reading(node))
        assert int(final["stopped_at"]) < 0  # inside the CFL line: a held run is no evidence
        worst = float(np.max(np.asarray(readings)))
        assert (worst <= 1.0) == (plant is None), (seed, worst)


@pytest.mark.parametrize("plant", [None, "port_gift"])
def test_the_windkessels_books_close(dtype, monkeypatch, plant):
    """FB3: the Windkessel's own books close at every step to its floor on its own scale (the work the fluid did on it
    is its capacitor's change, its resistances' loss and N_WK; the scale its terms' magnitudes and the face flux's
    summation floor through its block), N_WK booked; the control, a negative resistance of 5 % of its row's block that
    its books do not hold (a port gift), is past the floor. (The exchange alone, the fluid's port work against the law's
    received work, pairs one face flux on both sides and closes by construction: not a gate.)"""
    planted(monkeypatch, plant)
    for seed in SEEDS:
        final, _ = scan(point_node(draw(seed)), STEPS)
        books = books_of(final)
        assert books["n_wk"] > 0.0
        assert (books["law_identity"] <= 1.0) == (plant is None), (seed, books)


@pytest.mark.parametrize("plant", [None, "unit_scaled", "unit_negated"])
def test_the_superposed_field_is_the_monolithic_solve(dtype, monkeypatch, plant):
    """FB4: the step's solution, the solve's plus each unit field times its pressure's correction, has the monolithic
    projected residual at the openings' solved pressures, within ten times the solve's own; the controls, the unit
    field scaled by 1.1 or negated (its response with it, so the rows still close), are past it in pulsatile flow."""
    planted(monkeypatch, plant)
    for seed in SEEDS:
        node = point_node(draw(seed))
        final, readings = scan(node, STEPS, superposed_reading(node))
        stopped = int(final["stopped_at"])
        assert stopped < 0 or plant is not None  # a clean draw stays inside the CFL line
        judged = np.asarray(readings)[: stopped - 1] if stopped > 1 else np.asarray(readings)
        worst = float(np.max(judged))
        assert (worst <= 1.0) == (plant is None), (seed, worst)


#: The validity text's float32 envelope: the worst of 190 drawn open runs inside the CFL line, float32's flow against
#: float64's over the largest speed after 60 steps (median 3.3e-5; draw 0 read 4.3e-6), measured on the built node
#: (https://github.com/Microrobotics-Simulation-Framework/MIME/actions/runs/42877559684, arms f32env-*).
FLOAT32_ENVELOPE = 7.2e-4


def test_float32_is_within_its_line_of_float64():
    """The ports' float32 flow against float64's on one draw, within the envelope the validity text states (an
    acceptance of the statement, measured over the draws, not an accuracy bound)."""
    point = draw(0)
    finals = {}
    for dtype in ("float64", "float32"):
        with in_dtype(dtype):
            finals[dtype], _ = scan(point_node(point, ledger=False), STEPS)
    speed = max(float(np.max(np.abs(np.asarray(finals["float64"][f"velocity_{x}"])))) for x in "xyz")
    worst = max(float(np.max(np.abs(np.asarray(finals["float64"][f"velocity_{x}"], dtype=np.float64)
                                    - np.asarray(finals["float32"][f"velocity_{x}"], dtype=np.float64))))
                for x in "xyz") / speed
    assert worst <= FLOAT32_ENVELOPE, worst


@pytest.mark.parametrize(("interior", "oracle"), [("0.1h", 0.1), ("0.01h", 0.01), ("0.01h", 0.1)])
def test_the_solids_flux_at_each_opening_is_darcys(dtype, interior, oracle):
    """FA1 = FB14 (R5-C-02): at each opening, the flux through its faces deep in the solid (beyond the band and its
    stiffening ramp) is Darcy's at the run's own pressure drop and the arm's own stiff interior, to its line (a share of
    delta_in^2 and the solve's tolerance carried to the solid's small flux: ``open_cases.fa1_line``), steady
    Stokes off the cells' centre; the control holds the oracle at the nominal 0.1h on the 0.01h arm (the record's: the
    oracle kept at the nominal depth fails). The solid's share beyond the band's half width is reported per opening."""
    from open_cases import bypass_arm, bypass_reading, fa1_line

    tolerance = 1e-11 if dtype == "float64" else 1e-5
    node, final = bypass_arm(8, interior, offset=(0.3, -0.2), tolerance=tolerance)
    found = bypass_reading(node, final, oracle)
    held = float(interior.removesuffix("h"))
    assert (found["error"] <= fa1_line(found, held, tolerance)) == (oracle == held), found
    assert all(0.0 < share < 0.1 for share in found["solid_share"]), found
