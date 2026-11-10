"""The laws held at a flow's openings, on the law alone (F2b's PR 2): CI check (a), each law's per-stage block passive
over draws at physical scales (``mime.testing.port_law_suite``), its control a law with one sign flipped failing every
draw; a source exempt and said so; the port system's rows (a pressure row, a flux row, the gauge's row with no pressure
row) against a direct solve; the openings' well-posedness."""

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
    """The Windkessel with its stage's block's sign flipped (``p = a - b q``): a sign error its own books, which book
    whatever it does, cannot see."""

    def affine(self, parameters, state, time, dt):
        a, b = super().affine(parameters, state, time, dt)
        return a, -b

    def block(self, parameters, dt):
        return -super().block(parameters, dt)


def test_control_a_law_with_its_block_flipped_fails_every_draw():
    result = port_law_suite(_Flipped())
    assert result.failed and "failed on 200 of 200 draws" in result.detail, result.detail


def test_the_port_system_against_a_direct_solve():
    """One pressure row and one flux row: the pressures and fluxes solve both rows with the field's affine response."""
    rng = np.random.default_rng(0)
    response = np.array([[2.0], [-2.0]])
    base, predicted = np.array([0.3, -0.3]), np.array([5.0, 1.0])
    a, b = np.array([-1.5, 4.0]), np.array([0.0, 3.0])
    pressures, fluxes = port_system([RowKind.FLUX, RowKind.PRESSURE], a, b, base, response, predicted, 1)
    pressures, fluxes = np.asarray(pressures), np.asarray(fluxes)
    assert math.isclose(fluxes[0], a[0], rel_tol=1e-14)
    assert math.isclose(pressures[1], a[1] + b[1] * fluxes[1], rel_tol=1e-14)
    shift = (pressures[0] - pressures[1]) - (predicted[0] - predicted[1])
    np.testing.assert_allclose(fluxes, base + response[:, 0] * shift, rtol=1e-14)
    del rng


def test_with_no_pressure_row_the_gauge_sets_the_reference_and_the_fluxes_hold():
    response = np.array([[2.0], [-2.0]])
    base, predicted = np.array([0.3, -0.3]), np.array([0.0, 0.0])
    pressures, fluxes = port_system([RowKind.FLUX, RowKind.FLUX], np.array([-1.0, 1.0]), np.zeros(2), base,
                                    response, predicted, 0)
    assert float(pressures[0]) == 0.0
    np.testing.assert_allclose(np.asarray(fluxes), [-1.0, 1.0], rtol=1e-14)


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
