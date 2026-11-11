"""The laws held at a flow's openings, on the law alone (F2b's PR 2): CI check (a), the block of the row each law gives
the solver passive over draws at physical scales (``mime.testing.port_law_suite``), its controls a law with its row's
block flipped and one with its proximal resistance's sign flipped in its row and its books alike, each failing every
draw; a source exempt and said so; the port system's rows (a pressure row, a flux row, two pressure rows, the gauge's
row with no pressure row) against a direct solve, a move formed without the pressure level; the openings'
well-posedness."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from mime.nodes.flow.fvm._ports import PORT_LAWS, GivenPressure, Inflow, RowKind, Windkessel, port_system, \
    waveforms_cancel, well_posed
from mime.testing import port_law_suite


@pytest.fixture(autouse=True)
def _float64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.mark.parametrize("name", sorted(PORT_LAWS))
def test_every_port_law_passes_its_suite(name):
    result = port_law_suite(PORT_LAWS[name])
    print(result)
    assert result.passed, result.detail
    assert ("exempt" in result.detail) == PORT_LAWS[name].source


class _Flipped(Windkessel):
    """The Windkessel with its row's block's sign flipped (``p = a - b q``): a sign error its own books, which book
    whatever it does, cannot see."""

    def affine(self, parameters, state, time, dt):
        a, b = super().affine(parameters, state, time, dt)
        return a, -b


class _ConsistentlyNegative(Windkessel):
    """The Windkessel with its proximal resistance's sign flipped in the row the solver solves and in its own books
    alike (non-passive; a sign its own identity cannot see, since its books agree with its row)."""

    def affine(self, parameters, state, time, dt):
        held = self._denominator(parameters, dt)
        return parameters[1] * state[0] / held, -parameters[0] + dt / held

    def books(self, parameters, state, ended, flux, pressure, dt):
        flipped = parameters.at[0].multiply(-1.0)
        return super().books(flipped, state, ended, flux, pressure, dt)


@pytest.mark.parametrize("control", [_Flipped, _ConsistentlyNegative])
def test_control_a_law_with_a_sign_error_fails_every_draw(control):
    """Each control fails every draw (structurally, at the limit dt -> 0 where the block is R_p); the second's books
    agree with its row and close, so passivity alone is what catches it."""
    result = port_law_suite(control())
    assert result.failed and "failed on 200 of 200 draws" in result.detail, result.detail


def test_the_port_system_against_a_direct_solve():
    """One pressure row and one flux row: the pressures and fluxes solve both rows with the field's affine response,
    each relative pressure the predicted one plus its move."""
    response = np.array([[2.0], [-2.0]])
    base, predicted = np.array([0.3, -0.3]), np.array([4.0, 0.0])
    a, b = np.array([-1.5, 4.0]), np.array([0.0, 3.0])
    solved = port_system([RowKind.FLUX, RowKind.PRESSURE], a, b, base, response, predicted, 1)
    pressures, fluxes = np.asarray(solved.pressures), np.asarray(solved.fluxes)
    assert math.isclose(fluxes[0], a[0], rel_tol=1e-14)
    assert math.isclose(pressures[1], a[1] + b[1] * fluxes[1], rel_tol=1e-14)
    np.testing.assert_allclose(np.asarray(solved.relative), pressures - pressures[1], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(solved.relative), predicted + np.asarray(solved.moved), rtol=1e-14)
    np.testing.assert_allclose(fluxes, base + response[:, 0] * float(solved.moved[0]), rtol=1e-14)


def test_two_pressure_rows_against_a_direct_solve():
    response = np.array([[3.0], [-3.0]])
    base, predicted = np.array([0.2, -0.25]), np.array([0.0, -1.0])
    a, b = np.array([10.0, 2.0]), np.array([0.5, 1.5])
    solved = port_system([RowKind.PRESSURE, RowKind.PRESSURE], a, b, base, response, predicted, 0)
    pressures, fluxes = np.asarray(solved.pressures), np.asarray(solved.fluxes)
    np.testing.assert_allclose(pressures, a + b * fluxes, rtol=1e-14)
    assert float(solved.relative[0]) == 0.0
    np.testing.assert_allclose(fluxes, base + response[:, 0] * float(solved.moved[1]), rtol=1e-14)


def test_a_move_is_formed_without_the_pressure_level():
    """A pressure level a hundred million times the drop: the inflow's flux and the move are those of a level of zero
    to rounding of themselves (the move is never the difference of two absolute pressures)."""
    response = np.array([[2.0], [-2.0]])
    base, predicted = np.array([0.3, -0.3]), np.array([4.0, 0.0])
    a, b = np.array([-1.5, 1.0]), np.array([0.0, 3.0])
    low = port_system([RowKind.FLUX, RowKind.PRESSURE], a, b, base, response, predicted, 1)
    high = port_system([RowKind.FLUX, RowKind.PRESSURE], a + np.array([0.0, 1e8]), b, base, response, predicted, 1)
    assert float(high.moved[0]) == float(low.moved[0]) and float(high.fluxes[0]) == float(low.fluxes[0])
    assert math.isclose(float(high.pressures[1]) - float(low.pressures[1]), 1e8, rel_tol=1e-15)


def test_with_no_pressure_row_the_gauge_sets_the_reference_and_the_fluxes_hold():
    response = np.array([[2.0], [-2.0]])
    base, predicted = np.array([0.3, -0.3]), np.array([0.0, 0.0])
    solved = port_system([RowKind.FLUX, RowKind.FLUX], np.array([-1.0, 1.0]), np.zeros(2), base, response,
                         predicted, 0)
    assert float(solved.pressures[0]) == 0.0
    np.testing.assert_allclose(np.asarray(solved.fluxes), [-1.0, 1.0], rtol=1e-14)


def test_well_posedness():
    assert well_posed([RowKind.FLUX, RowKind.PRESSURE]) == ""
    assert well_posed([RowKind.PRESSURE, RowKind.PRESSURE]) == ""
    inflow = (1.0, 0.3, 2.0, 0.5)
    assert well_posed([RowKind.FLUX, RowKind.FLUX], [inflow, (-1.0, -0.3, 2.0, 0.5)]) == ""
    assert well_posed([RowKind.FLUX, RowKind.FLUX], [inflow, (-1.0, 0.3, 2.0, 0.5 + math.pi)]) == ""
    assert "no solution" in well_posed([RowKind.FLUX, RowKind.FLUX], [inflow, (-1.0, 0.3, 2.0, 0.5)])
    assert "no solution" in well_posed([RowKind.FLUX, RowKind.FLUX], [inflow, (-1.0, -0.3, 3.0, 0.5)])
    assert not waveforms_cancel([inflow, (-1.0 + 1e-9, -0.3, 2.0, 0.5)])


def test_each_laws_row_kind_and_class():
    assert Windkessel.row is RowKind.PRESSURE and Windkessel.stores and not Windkessel.source
    assert Inflow.row is RowKind.FLUX and Inflow.source and not Inflow.stores
    assert GivenPressure.row is RowKind.PRESSURE and GivenPressure.source
