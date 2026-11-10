"""The open box's gates (F2b's PR 2; predictions in ``predictions_open_ports.json``, committed before they ran): FB1,
the open tube is the periodic tube; FB2, the inflow's face flux is its waveform; FB3, the Windkessel's own books
close; FB4, the step's superposed solution is the monolithic one at its solved pressures. Each with its control, a
planted error caught on these draws; the controls' rates over fresh draws are measured on Actions (the PR's evidence
table)."""

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
    """FB2: at every step the inlet's face flux is the inflow's waveform to 64 eps of the face's magnitude; the
    control, a law that delivers 1e-3 more, is past it."""
    planted(monkeypatch, plant)
    for seed in SEEDS:
        node = point_node(draw(seed))
        _, readings = scan(node, STEPS, inflow_reading(node))
        worst = float(np.max(np.asarray(readings)))
        assert (worst <= 1.0) == (plant is None), (seed, worst)


@pytest.mark.parametrize("plant", [None, "port_gift"])
def test_the_windkessels_books_close(dtype, monkeypatch, plant):
    """FB3: the Windkessel's own books close at every step to its floor (the work the fluid did on it is its
    capacitor's change, its resistances' loss and N_WK), N_WK booked; the control, a negative resistance in its
    stage's block that its books do not hold (a port gift), is past the floor. (The exchange alone, the fluid's port
    work against the law's received work, pairs one solved flux on both sides and closes by construction: not a gate.)"""
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
        _, readings = scan(node, STEPS, superposed_reading(node))
        worst = float(np.max(np.asarray(readings)))
        assert (worst <= 1.0) == (plant is None), (seed, worst)


def test_float32_is_within_its_line_of_float64():
    """The ports' float32 flow against float64's on one draw: the validity text's 1.6e-6 to 3.5e-6 (the spike's,
    D/h 16) read on the built node, its line ten times the larger (this draw's grid and Reynolds number differ)."""
    point = draw(0)
    finals = {}
    for dtype in ("float64", "float32"):
        with in_dtype(dtype):
            finals[dtype], _ = scan(point_node(point, ledger=False), STEPS)
    speed = max(float(np.max(np.abs(np.asarray(finals["float64"][f"velocity_{x}"])))) for x in "xyz")
    worst = max(float(np.max(np.abs(np.asarray(finals["float64"][f"velocity_{x}"], dtype=np.float64)
                                    - np.asarray(finals["float32"][f"velocity_{x}"], dtype=np.float64))))
                for x in "xyz") / speed
    assert worst <= 3.5e-5, worst
