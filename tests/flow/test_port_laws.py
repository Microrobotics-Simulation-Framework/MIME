"""The laws held at a flow's openings, on the law alone (F2b's PR 2): CI check (a), the block of the row each law gives
the solver passive over draws at physical scales (``mime.testing.port_law_suite``), its controls a law with its row's
block flipped and one with its proximal resistance's sign flipped in its row and its books alike, each failing every
draw; a source exempt and said so; the port system's rows (a pressure row, a flux row, two pressure rows, the gauge's
row with no pressure row) against a direct solve, a move formed without the pressure level; the openings'
well-posedness."""

from __future__ import annotations

import dataclasses
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import Phase, given, settings

from mime.core.elimination import exchange_system
from mime.nodes.flow.fvm._ports import PORT_LAWS, GivenPressure, Inflow, PortSolution, RowKind, Windkessel, \
    opening_rows, openings_solved, waveforms_cancel, well_posed
from mime.testing import PORT_LAW_DRAWS, port_law_suite
from mime.testing._port_laws import books_residual


def port_system(rows, a, b, base, response, predicted, reference, coupling=None) -> PortSolution:  # noqa: ANN001
    """The openings alone through the one elimination: their rows (``opening_rows``), solved as the one partner
    (``exchange_system``), their pressures and fluxes back (``openings_solved``)."""
    partner, held, gauge = opening_rows(rows, a, b, base, response, predicted, reference, coupling)
    deliveries = exchange_system([partner]).deliveries
    return openings_solved(len(rows), reference, held, gauge, deliveries, base, response, predicted)


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


_WINDKESSEL_ROW = (1e7, 5.269861293567362e-12, 3560477528.4278646)
_WINDKESSEL_DT = 0.0007657869934001185
#: Flake run 42971163347's draw: a state of 2.2e-313 and a flux of -4.6e-157, neither of which a vessel bed has.
_UNDERFLOWN_DRAW = (*_WINDKESSEL_ROW, _WINDKESSEL_DT, 2.2250738585e-313, -4.634435157709434e-157)
_EXTREMES = (0.0, 2.2250738585e-313, 1e-200, 1e-160, 1e-154, 1e-100, 1e100, 1e154, 1e160)


def _windkessel_extreme_books(state, flux, loss_error=1.0):
    law = Windkessel() if loss_error == 1.0 else _LossScaled(loss_error)
    row = jnp.asarray([*_WINDKESSEL_ROW, 0.0])
    return books_residual(law, row, _WINDKESSEL_DT, jnp.asarray([state]), jnp.asarray(flux))


class _LossScaled(Windkessel):
    """The Windkessel whose loss is booked `factor` times too large: the planted error the extremes test must see."""

    def __init__(self, factor):
        self.factor = factor

    def books(self, parameters, state, ended, flux, pressure, dt):
        held = super().books(parameters, state, ended, flux, pressure, dt)
        return dataclasses.replace(held, loss=held.loss * self.factor)


def test_the_windkessels_books_at_numerical_extremes_close_to_the_floor_underflow_gives_them():
    """The draws stay at what a vessel bed has (`mime.testing.PORT_LAW_DRAWS`); this is the extremes on purpose, and
    it is not a draw the suite filters out. Where a state or a flux is subnormal, or small enough that its square
    (`q**2`, the loss) falls below the smallest normal number, the loss is flushed to zero and the books are off by what
    was flushed: at most the smallest normal number times the coefficients that multiply a square (`dt R_p`, `dt / R_d`,
    `C`), judged on top of the rounding on the inputs' scale (CLAUDE.md, the property tests' second cause). Where a
    square overflows (a state or flux above about 1e154) the books are not finite: the law announces it and books
    no wrong finite number. Between, the books close to rounding. Flake run 42971163347's draw is among them."""
    tiny = np.finfo(np.float64).tiny
    row_p, compliance, row_d = _WINDKESSEL_ROW
    floor = tiny * (_WINDKESSEL_DT * row_p + _WINDKESSEL_DT / row_d + compliance)
    cases = list(itertools.product(_EXTREMES, [0.0, *_EXTREMES, *(-value for value in _EXTREMES[1:])]))
    cases.append(_UNDERFLOWN_DRAW[4:])
    for state, flux in cases:
        residual, scale, rounding = _windkessel_extreme_books(state, flux)
        if max(abs(state), abs(flux)) >= 1e154:
            assert not math.isfinite(residual) or abs(residual) <= rounding + floor, (state, flux)
        else:
            assert math.isfinite(residual) and abs(residual) <= rounding + floor, (state, flux, residual, scale)
    residual, _, rounding = _windkessel_extreme_books(*_UNDERFLOWN_DRAW[4:])
    assert 0.0 < abs(residual) <= floor and rounding < abs(residual), "the flaked draw is the flushed loss, nothing more"


@settings(max_examples=2000, deadline=None, database=None, phases=(Phase.generate,))
@given(PORT_LAW_DRAWS["windkessel"]())
def test_the_windkessels_draws_are_what_a_vessel_bed_has(drawn):
    """Every draw is in the declared ranges, a state or a flux either exactly zero or at least 1e-2 Pa or 1e-13 m^3/s
    in size: none is subnormal (the flaked draw's state is 2.2e-313), the floor the suite's tolerance (no absolute
    part) cannot judge."""
    proximal, compliance, distal, dt, state, flux = drawn
    assert 1e7 * (1 - 1e-12) <= proximal <= 1e10 * (1 + 1e-12) and 1e-12 * (1 - 1e-12) <= compliance <= 1e-8 * (1 + 1e-12)
    assert 1e8 * (1 - 1e-12) <= distal <= 1e11 * (1 + 1e-12) and 1e-5 * (1 - 1e-12) <= dt <= 1e-2 * (1 + 1e-12)
    assert state == 0.0 or 1e-2 * (1 - 1e-12) <= state <= 2e4 * (1 + 1e-12)
    assert flux == 0.0 or 1e-13 * (1 - 1e-12) <= abs(flux) <= 1e-5 * (1 + 1e-12)


@pytest.mark.parametrize("state, flux", [(1e-100, 1e-100), (1e-100, -1e-100), (1e100, 1e-154), (1e100, 1e-100)])
def test_control_the_extremes_test_sees_a_booked_error_where_the_square_survives(state, flux):
    """Where the squares are normal, the extremes test has the resolution to see a loss booked 1e-3 too large; at a
    subnormal input it has none, by design (the flushed loss is below the floor), so no control is claimed there."""
    residual, _, rounding = _windkessel_extreme_books(state, flux, loss_error=1.001)
    floor = np.finfo(np.float64).tiny * (_WINDKESSEL_DT * _WINDKESSEL_ROW[0])
    assert abs(residual) > rounding + floor


#: CI's flaked draw (run 42989936874) of the shared distal law: its two fluxes equal and opposite, so each opening's
#: pressure, ``(R_k + R_d) q_k + R_d q_j``, is a difference of terms a thousand times its size.
_CANCELLING_DRAW = (50098404.29643129, 17220226.64075733, 72004899337.38588, 0.010000000000000004, 0.0,
                    -4.139937718785167e-08, 4.139937718785167e-08)


def test_a_shared_distal_draw_whose_off_diagonals_cancel_is_judged_on_its_terms():
    """The flaked draw closes on the received work's own products in magnitude, the scale its rounding came from (on
    CI and here its residual was 6.6e-23 J, past 64 eps of the work's result, 2.3e-9 J, the earlier scale, and within
    64 eps of its products, 4.9e-6 J). The control: a loss booked 1e-6 too large is still seen there."""
    law = PORT_LAWS["shared_distal"]
    row = jnp.asarray([*_CANCELLING_DRAW[:3], 0.0])
    dt, state, flux = _CANCELLING_DRAW[3], jnp.asarray([_CANCELLING_DRAW[4]]), jnp.asarray(_CANCELLING_DRAW[5:])
    residual, _, rounding = books_residual(law, row, dt, state, flux)
    assert abs(residual) <= rounding, (residual, rounding)

    class _LossScaled(type(law)):
        def books(self, parameters, state, ended, flux, pressure, dt):
            held = super().books(parameters, state, ended, flux, pressure, dt)
            return dataclasses.replace(held, loss=held.loss * (1.0 + 1e-6))

    planted, _, planted_rounding = books_residual(_LossScaled(), row, dt, state, flux)
    assert abs(planted) > planted_rounding, (planted, planted_rounding)


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


def test_the_port_system_solves_a_block_over_two_openings():
    """The port system with one law over two openings (a block with off-diagonals) against the network solved
    directly: a field of conductance ``G`` over three openings (symmetric, its rows and columns summing to zero), a
    flux row at the first, the shared distal law's two pressure rows at the others; the fluxes and pressures are the
    network's to rounding, and the split does not move with the distal resistance."""
    from mime.nodes.flow.fvm._ports import SharedDistal, port_matrix

    g = np.array([[-2.0, 1.2, 0.8], [1.2, -1.5, 0.3], [0.8, 0.3, -1.1]])
    flux = 1.7
    law = SharedDistal()
    rows = [RowKind.FLUX, RowKind.PRESSURE, RowKind.PRESSURE]
    coupling = [[True, False, False], [False, True, True], [False, True, True]]
    splits = []
    for distal in (5.0, 50.0):
        _, b = law.affine(jnp.asarray([1.0, 3.0, distal, 0.0]), jnp.zeros(1), 0.0, 1.0)
        block = jnp.zeros((3, 3)).at[1:, 1:].set(b)
        reference = 1
        response = jnp.asarray(g[:, [0, 2]])
        solution = port_system(rows, jnp.asarray([-flux, 0.0, 0.0]), block, jnp.zeros(3), response, jnp.zeros(3),
                               reference, coupling)
        q, p = np.asarray(solution.fluxes), np.asarray(solution.pressures)
        np.testing.assert_allclose(q, g @ (p - p[1]), atol=1e-12)
        np.testing.assert_allclose(q[0], -flux, rtol=1e-12)
        np.testing.assert_allclose(p[1:], np.asarray(b) @ q[1:], rtol=1e-12)
        splits.append(q[1] / (q[1] + q[2]))
        assert np.isfinite(np.linalg.cond(np.asarray(port_matrix(rows, block, response, reference, coupling))))
    assert abs(splits[0] - splits[1]) < 1e-12, splits
