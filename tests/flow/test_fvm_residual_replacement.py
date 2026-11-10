"""The CG's residual replacement as a restart at the stall (``mime.nodes.flow.fvm._solve``'s module docstring): where
the stop would find a stagnation, the CG restarts from the true residual instead, up to ``STALL_RESTARTS`` times.

Its gates: with the 0.01h interior in a flow with a pressure gradient (a stenosis eight cells across, over draws of its
shape, its place on the grid and its drive) float32's solve converges, unflagged, at float64's answer within a line, on
every draw; the derivative solves (tangent and transpose, read through the report block) converge with the primal; a
solve that never stagnates is bit for bit the CG without the restart, in either dtype. Its controls: a stall the case
plants on fixed shapes measured to stall without the restart (whether a float32 draw stalls is a rounding event that
differs by CPU, so the stall is not asserted over draws), which the restart breaks; and a planted wrong restart (the
residual's sign flipped, from the start), which moves the answer and is flagged on every draw."""

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
from mime.nodes.flow.fvm._solve import REPORT_FIELDS, REPORT_SIZE, default_tolerance, rounding_floor, \
    stagnation_window

ITERATIONS, RESIDUAL, AT_CAP, STAGNATED = (REPORT_FIELDS.index(name) for name in
                                           ("iterations", "residual", "at_cap", "stagnated"))


class StenosisSolve:
    """Steady Stokes through a 75 % stenosis (``fvm_cases.Stenosis``) of D = 1 with the 0.01h interior, two diameters
    long, ``cells_across`` cells across the lumen (nu = 1), driven on its fluid along ``(tilt_x, tilt_y, 1)``; the
    stenosis's narrowing, its centre's offset from the box's (in cells) and the drive's tilt are the solve's
    arguments, so a draw changes no shape and the program compiles once. Its solve is ``projected_solve`` under each
    dtype's default stop, restarting at a stall up to ``stall_restarts`` times (0: never; none given: the default)."""

    def __init__(self, dtype: str, cells_across: int, stall_restarts: int | None = None):
        self.dtype = dtype
        self.run = jnp.float64 if dtype == "float64" else jnp.float32
        self.layout = periodic_tube_layout(0.5, 2.0, cells_across=cells_across)
        self.grid = MACGrid(self.layout.cells, self.layout.spacing)
        self.stopping = Stopping(tolerance=jnp.asarray(default_tolerance(dtype), dtype=self.run),
                                 at_floor=dtype == "float32", window=stagnation_window(self.grid.cells, 20),
                                 stall_restarts=solve_module.STALL_RESTARTS if stall_restarts is None
                                 else stall_restarts)
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
def stenosis(dtype: str, cells_across: int = 8, stall_restarts: int | None = None) -> StenosisSolve:
    with in_dtype(dtype):
        return StenosisSolve(dtype, cells_across, stall_restarts)


def _flagged(report) -> bool:
    return bool(report[AT_CAP] + report[STAGNATED] > 0)


def _relative(a, b, grid) -> float:
    difference = tuple(x.astype(jnp.float64) - y.astype(jnp.float64) for x, y in zip(a, b, strict=True))
    size = grid.inner(tuple(y.astype(jnp.float64) for y in b), tuple(y.astype(jnp.float64) for y in b))
    return float(jnp.sqrt(grid.inner(difference, difference) / size))


SHAPES = st.tuples(st.floats(0.4, 0.6), st.floats(-0.5, 0.5), st.floats(-0.5, 0.5), st.floats(-0.3, 0.3),
                   st.floats(-0.3, 0.3))


#: float32's answer against float64's, relative, where the restart converges: float32's tolerance, declared before the
#: gate ran (``predictions_residual_replacement.json``, ``fix_round_b1``), about 19 times the largest error of the 379
#: shapes a stress run drew (5.4e-7).
ANSWER_LINE = 1e-5
#: Shapes on which the CG without the restart stalls, measured on CI's runners and here: each drawn by a different
#: stress run that passed (so it stalled on that run's runner), and stalling here on both kinds of core, its least
#: residual 4.5e-5 to 6.1e-5. Off-centre shapes stalled on 344 of 344 here; a centred one's stall turns on the CPU.
STALLING = (
    (0.4, 0.0, 0.4131182338695217, 0.0, 0.0),
    (0.4, 0.23187168742084985, 0.23187168742084985, 0.14582011025102282, 0.13811306304991489),
    (0.5748629024643002, 0.4487391197415629, 0.32589321661738846, -0.21365488601539637, 0.22488432954975907),
    (0.48454397635447477, -0.07995353997088628, -0.3569961749309025, 0.1761588039039737, -0.006010456649702567),
)


def _converged_at_float64s_answer(shape) -> None:
    with in_dtype("float32"):
        solved, report = stenosis("float32")(shape)
        assert not _flagged(report), report
        assert report[RESIDUAL] <= max(default_tolerance("float32"), stenosis("float32").floor(shape, solved))
    with in_dtype("float64"):
        reference, _ = stenosis("float64")(shape)
    assert _relative(solved, reference, stenosis("float32").grid) <= ANSWER_LINE


# -- the gate: on every draw the restarted solve converges at float64's answer ---------------------------------------

@settings(max_examples=20, deadline=None)
@given(shape=SHAPES)
def test_with_the_restart_float32_converges_at_float64s_answer_on_every_draw(shape):
    """The 0.01h interior in a stenosis eight cells across, float32, every draw (centred ones among them, which may or
    may not stall without the restart): the solve converges, unflagged, at its tolerance or floor, and its answer is
    float64's within ``ANSWER_LINE``."""
    _converged_at_float64s_answer(shape)


# -- the control the case plants: a stall, on shapes measured to stall, that the restart breaks -----------------------

@pytest.mark.parametrize("shape", STALLING)
def test_control_a_float32_stall_on_shapes_measured_to_stall_the_restart_breaks(shape):
    """Without the restart the CG's recurrence stalls above the solve's floor and the solve stops stagnated, flagged;
    restarted at the stall it converges, unflagged, at float64's answer within ``ANSWER_LINE``. The shapes are fixed:
    whether a draw stalls is a rounding event, and a centred draw that stalls on one CPU converged on another."""
    with in_dtype("float32"):
        stalled, without = stenosis("float32", stall_restarts=0)(shape)
        assert without[STAGNATED] == 1.0, without
        assert without[RESIDUAL] > max(default_tolerance("float32"), stenosis("float32").floor(shape, stalled))
    _converged_at_float64s_answer(shape)


# -- the planted wrong restart: the answer moved, and flagged ----------------------------------------------------------

@settings(max_examples=20, deadline=None)
@given(shape=SHAPES)
def test_control_a_wrong_restart_moves_the_answer_and_is_flagged_on_every_draw(shape):
    """The restart planted with the residual's sign flipped (``A x - b``) in both dtypes; the start is made through the
    same restart, so the plant acts from the first iteration: the recurrence carries on away from the answer and the
    true residual never falls; the stop, which judges the true residual alone, never passes it as converged. On every
    draw the solve is flagged and its answer is farther from the right one than ten times the tolerance. A flip at a
    retry alone, the start right, is caught by nothing here and needs no catching: the true-residual stop and the
    restart where the recursive residual falls first recover it (measured: unflagged on 21 of 21 off-centre shapes,
    its answer within 9.4e-8 of the right restart's)."""
    for dtype in ("float32", "float64"):
        with in_dtype(dtype):
            right, _ = stenosis(dtype)(shape)
            wrong, report = flipped(dtype)(shape)
            assert _flagged(report), report
            assert _relative(wrong, right, flipped(dtype).grid) > 10 * default_tolerance(dtype)


@functools.cache
def flipped(dtype: str) -> StenosisSolve:
    """The stenosis's solve with its restarts' (and so its start's) residual sign flipped, traced once with the plant
    in place (the compiled program keeps it)."""
    original = solve_module.restarted

    def plant(true_r, precondition, inner):
        return original(tuple(-c for c in true_r), precondition, inner)

    with in_dtype(dtype), pytest.MonkeyPatch.context() as patch:
        patch.setattr(solve_module, "restarted", plant)
        solve = StenosisSolve(dtype, 8)
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


def _derivatives(dtype: str, stall_restarts: int | None = None):
    """The axial flux's derivative in the stenosis's narrowing at ``SHIFTED``, sixteen cells across: forward (one
    tangent solve) and reverse (one transpose solve), with the primal's, the tangent's and the transpose's reports
    read through the report block."""
    with in_dtype(dtype):
        solve = stenosis(dtype, 16, stall_restarts)
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
    stalls: the axial flux's derivative in the narrowing, forward and reverse. Restarted at its stalls, the primal, the
    tangent and the transpose solve each converge unflagged, as their reports read through the block say, and both
    derivatives are float64's within the line below. Float64's solves converge unflagged and never stagnate, so its
    derivatives are bit for bit the CG's without the restart."""
    forward, reverse, primal, tangent, transposed = _derivatives("float32")
    for report in (primal, tangent, transposed):
        assert not _flagged(report) and report[ITERATIONS] >= 1.0, report
    exact_forward, exact_reverse, *reports = _derivatives("float64")
    assert all(not _flagged(report) for report in reports), reports
    for derivative, exact in ((forward, exact_forward), (reverse, exact_reverse)):
        assert abs(derivative / exact - 1.0) <= DERIVATIVE_LINE, (derivative, exact)
    without_forward, without_reverse, *_ = _derivatives("float64", 0)
    assert (without_forward, without_reverse) == (exact_forward, exact_reverse)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), amplitude=st.floats(1e-3, 1e3))
def test_a_solve_that_never_stagnates_is_the_cg_without_the_restart(seed, amplitude):
    """The empty tube eight cells across at a backward-Euler shift (``test_fvm_solve.Problem``), whose solves converge
    without a stagnation: in either dtype its answer and report are bit for bit the CG's without the restart. Both arms
    run this loop, the retry's carry and selects in each, so this shows the restart inert where nothing stagnates, not
    equality with the CG before it; that was measured outside the suite, the released programs and the stenosis's
    solves bit for bit against the release branch's tip."""
    for dtype in ("float32", "float64"):
        with in_dtype(dtype):
            problem = problem_in(dtype)
            b = problem.rhs(seed, amplitude)
            x, report = problem.cg(b)
            assert report[STAGNATED] == 0.0
            plain, plain_report = problem.cg(b, stall_restarts=0)
            assert np.array_equal(np.asarray(report), np.asarray(plain_report))
            assert all(np.array_equal(np.asarray(a), np.asarray(c)) for a, c in zip(x, plain, strict=True))
