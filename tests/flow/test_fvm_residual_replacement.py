"""The CG's residual replacement in its restart form (``mime.nodes.flow.fvm._solve``'s module docstring): in float32,
by default, every ``RESTART_EVERY`` iterations the CG restarts from the true residual.

Its gates: float32's recurrence stalls with the 0.01h interior in a flow with a pressure gradient (a stenosis) are
broken, unflagged, at float64's answer within the tolerance; the derivative solves (tangent and transpose, read through
the report block) converge with the primal; float64, which does not restart by default, moves by no more than its
tolerance when asked to; a solve that ends before the first restart is bit for bit the CG without it. Its controls: a
stall the case plants (the 0.01h interior in a stenosis eight cells across, over draws of its shape, its place on the
grid and its drive), which the restart breaks on every draw and the CG without it does not; and a planted wrong restart
(the residual's sign flipped), which moves the answer and is flagged on every draw."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import Stenosis, in_dtype
from hypothesis import given, settings
from hypothesis import strategies as st
from test_fvm_solve import problem_in

import mime.nodes.flow.fvm._solve as solve_module
from mime.nodes.flow.fvm import Jacobi, MACGrid, Stopping, periodic_tube_layout, projected_solve
from mime.nodes.flow.fvm._band import DEFAULT_BAND, band_named, wall_fields
from mime.nodes.flow.fvm._solve import REPORT_FIELDS, REPORT_SIZE, default_restart_every, default_tolerance, \
    rounding_floor, stagnation_window

ITERATIONS, RESIDUAL, AT_CAP, STAGNATED = (REPORT_FIELDS.index(name) for name in
                                           ("iterations", "residual", "at_cap", "stagnated"))


class StenosisSolve:
    """Steady Stokes through a 75 % stenosis (``fvm_cases.Stenosis``) of D = 1 with the 0.01h interior, two diameters
    long, ``cells_across`` cells across the lumen (nu = 1), driven on its fluid along ``(tilt_x, tilt_y, 1)``; the
    stenosis's narrowing, its centre's offset from the box's (in cells) and the drive's tilt are the solve's
    arguments, so a draw changes no shape and the program compiles once. Its solve is ``projected_solve`` under each
    dtype's default stop, restarting every ``restart_every`` iterations (0: never; none given: the dtype's default)."""

    def __init__(self, dtype: str, cells_across: int, restart_every: int | None = None):
        self.dtype = dtype
        self.run = jnp.float64 if dtype == "float64" else jnp.float32
        self.layout = periodic_tube_layout(0.5, 2.0, cells_across=cells_across)
        self.grid = MACGrid(self.layout.cells, self.layout.spacing)
        self.stopping = Stopping(tolerance=jnp.asarray(default_tolerance(dtype), dtype=self.run),
                                 at_floor=dtype == "float32", window=stagnation_window(self.grid.cells, 20),
                                 restart_every=default_restart_every(dtype) if restart_every is None
                                 else restart_every)
        self.solved = jax.jit(self._solved)

    def pieces(self, shape):
        """The operator, its magnitude, Jacobi's diagonal and the drive, of ``shape`` (narrowing, x and y offsets in
        cells, the drive's x and y tilt)."""
        narrowing, x, y, tilt_x, tilt_y = (shape[k] for k in range(5))
        grid, h = self.grid, self.layout.spacing
        solid = Stenosis(self.layout.centre[0] + x * h, self.layout.centre[1] + y * h, 0.5,
                         grid.cells[2] * grid.spacing, narrowing)
        fields = wall_fields(grid, [solid], band_named(DEFAULT_BAND), 0.01, self.run)
        sigma = fields.inverse_permeability

        def apply(u):
            lap = grid.laplacian(u)
            return tuple(s * a - n for a, n, s in zip(u, lap, sigma, strict=True))

        def magnitude(u):
            lap = grid.laplacian_magnitude(u)
            return tuple(n + s * a for a, n, s in zip(u, lap, sigma, strict=True))

        diagonal = tuple(-grid.laplacian_diagonal() + s for s in sigma)
        along = (tilt_x, tilt_y, 1.0)
        drive = tuple((g * (1.0 - chi)).astype(self.run) for g, chi in zip(along, fields.chi, strict=True))
        return apply, magnitude, diagonal, drive

    def _solved(self, shape, tap):
        apply, magnitude, diagonal, drive = self.pieces(shape)
        return projected_solve(self.grid, apply, diagonal, drive, None, self.stopping, tap, Jacobi(), magnitude)

    def __call__(self, shape):
        shape = jnp.asarray(shape, dtype=self.run)
        u, report = self.solved(shape, jnp.zeros(REPORT_SIZE, dtype=self.run))
        return u, np.asarray(report)

    def floor(self, shape, u):
        """The rounding floor at ``u`` relative to the right-hand side, recomputed outside the solve."""
        apply, magnitude, _, drive = self.pieces(jnp.asarray(shape, dtype=self.run))
        b = self.grid.project(drive)
        return float(rounding_floor(b, magnitude, self.grid.inner)(u) / jnp.sqrt(self.grid.inner(b, b)))


@functools.cache
def stenosis(dtype: str, cells_across: int = 8, restart_every: int | None = None) -> StenosisSolve:
    with in_dtype(dtype):
        return StenosisSolve(dtype, cells_across, restart_every)


def _flagged(report) -> bool:
    return bool(report[AT_CAP] + report[STAGNATED] > 0)


def _relative(a, b, grid) -> float:
    difference = tuple(x.astype(jnp.float64) - y.astype(jnp.float64) for x, y in zip(a, b, strict=True))
    size = grid.inner(tuple(y.astype(jnp.float64) for y in b), tuple(y.astype(jnp.float64) for y in b))
    return float(jnp.sqrt(grid.inner(difference, difference) / size))


SHAPES = st.tuples(st.floats(0.4, 0.6), st.floats(-0.5, 0.5), st.floats(-0.5, 0.5), st.floats(-0.3, 0.3),
                   st.floats(-0.3, 0.3))


# -- the control the case plants: a stall the restart breaks on every draw ---------------------------------------------

@settings(max_examples=20, deadline=None)
@given(shape=SHAPES)
def test_control_a_float32_stall_the_restart_breaks_on_every_draw(shape):
    """The 0.01h interior in a stenosis eight cells across, float32: without the restart the CG's recurrence stalls
    above the solve's floor and the solve stops stagnated, flagged; with float32's default restart it converges,
    unflagged, at its tolerance or floor, and its answer is the flagged one's or closer to float64's."""
    with in_dtype("float32"):
        stalled, without = stenosis("float32", restart_every=0)(shape)
        assert without[STAGNATED] == 1.0, without
        assert without[RESIDUAL] > max(default_tolerance("float32"), stenosis("float32").floor(shape, stalled))
        solved, report = stenosis("float32")(shape)
        assert not _flagged(report), report
        assert report[RESIDUAL] <= max(default_tolerance("float32"), stenosis("float32").floor(shape, solved))
    with in_dtype("float64"):
        reference, _ = stenosis("float64")(shape)
    grid = stenosis("float32").grid
    assert _relative(solved, reference, grid) <= max(_relative(stalled, reference, grid), 1e-4)


# -- the planted wrong restart: the answer moved, and flagged ----------------------------------------------------------

@settings(max_examples=20, deadline=None)
@given(shape=SHAPES)
def test_control_a_wrong_restart_moves_the_answer_and_is_flagged_on_every_draw(shape):
    """The restart planted with the residual's sign flipped (``A x - b``) in both dtypes, at every check (20 iterations)
    so that it acts well before the solve would converge: the recurrence then carries on away from the answer, so the
    true residual stops falling; the stop, which judges the true residual alone, never passes it as converged. On every
    draw the solve is flagged and its answer is farther from the right one than ten times the tolerance."""
    for dtype in ("float32", "float64"):
        with in_dtype(dtype):
            right, _ = stenosis(dtype)(shape)
            wrong, report = flipped(dtype)(shape)
            assert _flagged(report), report
            assert _relative(wrong, right, flipped(dtype).grid) > 10 * default_tolerance(dtype)


@functools.cache
def flipped(dtype: str) -> StenosisSolve:
    """The stenosis's solve restarted at every check with its residual's sign flipped, traced once with the plant in
    place (the compiled program keeps it)."""
    original = solve_module.restarted

    def plant(true_r, precondition, inner):
        return original(tuple(-c for c in true_r), precondition, inner)

    with in_dtype(dtype), pytest.MonkeyPatch.context() as patch:
        patch.setattr(solve_module, "restarted", plant)
        solve = StenosisSolve(dtype, 8, 20)
        solve((0.5, 0.0, 0.0, 0.0, 0.0))
    return solve


# -- the gates -------------------------------------------------------------------------------------------------------

#: float32's derivative against float64's, relative: declared before it was measured
#: (``predictions_residual_replacement.json``).
DERIVATIVE_LINE = 1e-4
#: The stenosis of the interior spike's float32 stalls (sixteen cells across, the 0.01h interior, steady, driven along
#: its axis), its centre a sub-cell step off the grid's, so that no face lies on its axis (where the distance's
#: derivative is 0/0, which a zero tangent turns into NaN).
SHIFTED = (0.5, 0.3, 0.2, 0.0, 0.0)


def _axial_flux(solve: StenosisSolve, u) -> jax.Array:
    grid = solve.grid
    return jnp.sum(u[2]) * grid.cell_volume / (grid.cells[2] * grid.spacing)


def _derivatives(dtype: str, restart_every: int | None = None):
    """The axial flux's derivative in the stenosis's narrowing at ``SHIFTED``, sixteen cells across: forward (one
    tangent solve) and reverse (one transpose solve), with the primal's, the tangent's and the transpose's reports
    read through the report block."""
    with in_dtype(dtype):
        solve = stenosis(dtype, 16, restart_every)
        shape = jnp.asarray(SHIFTED, dtype=solve.run)
        tap = jnp.zeros(REPORT_SIZE, dtype=solve.run)
        along = jnp.zeros(5, dtype=solve.run).at[0].set(1.0)

        def flux(s, t):
            u, report = solve._solved(s, t)  # noqa: SLF001 - the solve with its report block's input
            return _axial_flux(solve, u), report

        (_, primal), (forward, tangent) = jax.jit(lambda s: jax.jvp(lambda v: flux(v, tap), (s,), (along,)))(shape)
        reverse, transposed = jax.jit(jax.grad(lambda s, t: flux(s, t)[0], argnums=(0, 1)))(shape, tap)
        return float(forward), float(reverse[0]), np.asarray(primal), np.asarray(tangent), np.asarray(transposed)


def test_the_derivative_solves_converge_with_the_primal_through_the_report_block():
    """The stenosis sixteen cells across with the 0.01h interior, float32, where without the restart the primal
    stalls: the axial flux's derivative in the narrowing, forward and reverse. Under float32's default restart the
    primal, the tangent and the transpose solve each converge unflagged, as their reports read through the block say,
    and both derivatives are float64's within the line below. Float64, which does not restart by default, converges
    unflagged, and asked to restart moves neither derivative by more than its tolerance."""
    forward, reverse, primal, tangent, transposed = _derivatives("float32")
    for report in (primal, tangent, transposed):
        assert not _flagged(report) and report[ITERATIONS] >= 1.0, report
    exact_forward, exact_reverse, *reports = _derivatives("float64")
    assert all(not _flagged(report) for report in reports), reports
    for derivative, exact in ((forward, exact_forward), (reverse, exact_reverse)):
        assert abs(derivative / exact - 1.0) <= DERIVATIVE_LINE, (derivative, exact)
    restarted_forward, restarted_reverse, *reports = _derivatives("float64", solve_module.RESTART_EVERY)
    assert all(not _flagged(report) for report in reports), reports
    for a, b in ((restarted_forward, exact_forward), (restarted_reverse, exact_reverse)):
        assert abs(a / b - 1.0) <= 100 * default_tolerance("float64")


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), amplitude=st.floats(1e-3, 1e3))
def test_a_solve_ending_before_its_first_restart_is_the_cg_without_it(seed, amplitude):
    """The empty tube eight cells across at a backward-Euler shift (``test_fvm_solve.Problem``), whose solves end well
    within the first ``RESTART_EVERY`` iterations: asked to restart, in either dtype, its answer and report are bit for
    bit the CG's without it."""
    for dtype in ("float32", "float64"):
        with in_dtype(dtype):
            problem = problem_in(dtype)
            b = problem.rhs(seed, amplitude)
            x, report = problem.cg(b, restart_every=solve_module.RESTART_EVERY)
            assert int(report[ITERATIONS]) < solve_module.RESTART_EVERY
            plain, plain_report = problem.cg(b, restart_every=0)
            assert np.array_equal(np.asarray(report), np.asarray(plain_report))
            assert all(np.array_equal(np.asarray(a), np.asarray(c)) for a, c in zip(x, plain, strict=True))
