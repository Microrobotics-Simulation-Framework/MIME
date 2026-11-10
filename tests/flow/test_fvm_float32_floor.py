"""Float32's default stop at the solve's rounding floor (``mime.nodes.flow.fvm._solve.rounding_floor``): where 1e-5 is
out of float32's reach the solve stops at its floor, unflagged, its values float64's within the tolerance; where the
conjugate gradient's recurrence stalls above the floor a restart from the true residual at the stall breaks it; and a
stop at the floor is accepted only where the true residual is itself small (``FLOOR_STOP_TOLERANCES``), so a singular
system, whose iterate grows and raises its own floor, is never passed as converged. The cases are the solid's interior
spike's float32 runs that the fixed 1e-5 flagged, and a periodic Laplacian with no penalty."""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import Stenosis, in_dtype, tube_node
from test_fvm_experiment import experiment

import mime.nodes.flow.fvm._solve as solve_module
from mime.dynamics import Band
from mime.nodes.flow.fvm import Jacobi, MACGrid, Stopping, guarded_cg, projected_solve
from mime.nodes.flow.fvm._band import DEFAULT_BAND, band_named, wall_fields
from mime.nodes.flow.fvm._solve import REPORT_FIELDS, REPORT_SIZE, STALL_RESTARTS, default_tolerance, \
    rounding_floor, stagnation_window
from mime.units import mm, ms

STAGNATED, RESIDUAL = REPORT_FIELDS.index("stagnated"), REPORT_FIELDS.index("residual")
FLAGS = [REPORT_FIELDS.index(name) for name in ("at_cap", "stagnated", "not_finite")]


def _tube_flux(dtype: str, **settings: object) -> tuple[float, np.ndarray, bool]:
    """The empty tube at 64 cells across with the 0.01h interior, steady Stokes, one step through the build: its lumen
    flux, its solve's report, and whether the run flagged the solve (a flagged solve warns, which is asserted)."""
    with in_dtype(dtype), warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        result = experiment(cells_across=64, length=mm(2.0 / 64.0), wall=Band(delta_in=0.01),
                            **settings).build().run(ms(20.0))
    final = result.final["water.flow"]
    flagged = any(flag.path == "water.flow.solve_flagged" for flag in result.flags)
    assert flagged == any("solve_flagged" in str(warning.message) for warning in said)
    return float(np.asarray(final["lumen_flux"])), np.asarray(final["solve_report"]), flagged


def test_float32s_default_stop_takes_the_rounding_floor_where_its_tolerance_is_out_of_reach():
    """At 64 cells across with the 0.01h interior float32 cannot reach 1e-5 (float64's answer rounded to float32 reads
    1.6e-5 to 2.1e-5): the default stop takes the floor, unflagged, and its flux is float64's within float32's
    tolerance. The control, the same tolerance given (held as given, with no floor), stagnates and is flagged."""
    flux, report, flagged = _tube_flux("float32")
    assert not flagged and report[2:5].sum() == 0.0, report
    assert report[1] > default_tolerance("float32")  # it stopped at the floor, not at the tolerance
    reference, _, _ = _tube_flux("float64")
    assert abs(flux - reference) <= default_tolerance("float32") * abs(reference)
    _, given, given_flagged = _tube_flux("float32", tolerance=1e-5)
    assert given_flagged and given[STAGNATED] == 1.0, given


def _stenosis_solve(dtype: str, tolerance: float, at_floor: bool,
                    **stop: int) -> tuple[np.ndarray, float, float, float]:
    """The 75 % stenosis at 16 cells across with the 0.01h interior, steady Stokes, driven on its fluid: the solve's
    report, the relative rounding floor at its answer, its flux through the box (m^3/s per unit length's worth), and its
    true relative residual."""
    with in_dtype(dtype):
        node = tube_node(16, g=1.0, length_cells=32, interior="0.01h")
        p = node._step_constants(None)  # noqa: SLF001 - the step's own constants
        grid = node.grid(p)
        centre = p["tube_centre"]
        solid = Stenosis(float(centre[0]), float(centre[1]), 0.5, grid.cells[2] * grid.spacing)
        run = jnp.float64 if dtype == "float64" else jnp.float32
        fields = wall_fields(grid, [solid], band_named(DEFAULT_BAND), 0.01, run)
        sigma = node.resistivity(p, fields)
        shift = jnp.asarray(0.0, dtype=run)
        apply = node.operator_of(p, grid, sigma)(shift)
        magnitude = node.magnitude_of(p, grid, sigma)(shift)
        rhs = node.drive(fields, jnp.asarray([0.0, 0.0, 1.0], dtype=run))
        stopping = Stopping(tolerance=jnp.asarray(tolerance, dtype=run), at_floor=at_floor,
                            window=stagnation_window(grid.cells, 20), **stop)

        @jax.jit
        def solved(r):
            return projected_solve(grid, apply, node.diagonal_of(p, grid, sigma, shift), r, None, stopping,
                                   jnp.zeros(REPORT_SIZE, dtype=run), Jacobi(), magnitude)

        u, report = solved(rhs)
        b = grid.project(rhs)
        size = jnp.sqrt(grid.inner(b, b))
        floor = rounding_floor(b, magnitude, grid.inner)(u) / size
        left = tuple(x - y for x, y in zip(b, grid.project(apply(grid.project(u))), strict=True))
        residual = jnp.sqrt(grid.inner(left, left)) / size
        flux = jnp.sum(u[2].astype(run)) * grid.cell_volume
        return np.asarray(report), float(floor), float(flux), float(residual)


def test_a_float32_stall_above_the_floor_is_broken_by_the_restart():
    """With the 0.01h interior in a flow with a pressure gradient float32's conjugate gradient's recurrence stalls far
    above the floor (1.0e-4 against 4.7e-6, measured): a stall, not a floor. A restart from the true residual where the
    stop would find the stagnation breaks it: the solve converges, unflagged, at its tolerance or floor, its flux
    float64's within the tolerance. The control is the CG without the restart, which stops stagnated above its floor,
    so the case tells a stall from a floor."""
    tolerance = default_tolerance("float32")
    report, floor, flux, _ = _stenosis_solve("float32", tolerance, at_floor=True)
    assert report[2:5].sum() == 0.0, report
    assert report[RESIDUAL] <= max(floor, tolerance), (report, floor)
    _, _, reference, _ = _stenosis_solve("float64", 1e-12, at_floor=False)
    assert abs(flux - reference) <= tolerance * abs(reference)
    stalled, stalled_floor, _, residual = _stenosis_solve("float32", tolerance, at_floor=True, stall_restarts=0)
    assert stalled[STAGNATED] == 1.0, stalled
    assert residual > max(stalled_floor, tolerance)


#: The singular plants' inconsistent parts, relative to the right-hand side's range part of size 1.
INCONSISTENT = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@functools.cache
def _singular(dtype: str, restarts: int, line: float | None = None):
    """The plant: the CG at the rounding-floor stop on a periodic Laplacian with no penalty, 16 cells a side (singular:
    a constant field is its null space), from a zero start; its right-hand side a smooth field of size 1 in the
    operator's range plus a constant ``c`` outside it, so no iterate solves it and the iterate grows. Returns the solve
    of ``c``: its report and its true residual relative to the right-hand side. ``line`` replaces
    ``FLOOR_STOP_TOLERANCES`` for the trace (``inf``: no line)."""
    with in_dtype(dtype), pytest.MonkeyPatch.context() as patch:
        if line is not None:
            patch.setattr(solve_module, "FLOOR_STOP_TOLERANCES", line)
        run = jnp.float64 if dtype == "float64" else jnp.float32
        n = 16
        grid = MACGrid((n, n, n), 1.0 / n)
        X = (jnp.arange(n) + 0.5) / n
        xx, yy, zz = jnp.meshgrid(X, X, X, indexing="ij")
        smooth = (jnp.sin(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * zz)).astype(run)
        stopping = Stopping(tolerance=jnp.asarray(default_tolerance(dtype), run), at_floor=True, cap=5000,
                            stall_restarts=restarts)

        def apply(u):
            return tuple(-component for component in grid.laplacian(u))

        diagonal = tuple(-grid.laplacian_diagonal() + 0.0 * a for a in grid.zeros(run))

        def solve(c):
            b = (smooth + c, 0.5 * smooth, jnp.cos(2 * jnp.pi * yy).astype(run) + c)
            x, report = guarded_cg(apply, b, Jacobi().made(diagonal), grid.zeros(run), stopping, grid.inner,
                                   grid.laplacian_magnitude)
            left = tuple(p - q for p, q in zip(b, apply(x), strict=True))
            return report, jnp.sqrt(grid.inner(left, left) / grid.inner(b, b))

        solved = jax.jit(solve)
        solved(jnp.asarray(1.0, run))  # traced here, under the patch

    def of(c: float) -> tuple[np.ndarray, float]:
        with in_dtype(dtype):
            report, residual = solved(jnp.asarray(c, run))
            return np.asarray(report), float(residual)

    return of


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_control_a_singular_system_is_flagged_with_and_without_the_restart(dtype):
    """Every plant, with the restart and without, ends flagged: the line refuses a stop at a floor the solve's own
    growing iterate raised. Without the line most were passed as converged (the next test): float32's at true residuals
    of 4.5 to 1280 times the right-hand side's, float64's (its floor on by hand) at 0.75 to 4e11 times."""
    for restarts in (0, STALL_RESTARTS):
        for c in INCONSISTENT:
            report, _ = _singular(dtype, restarts)(c)
            assert report[FLAGS].sum() > 0, (restarts, c, report)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_control_without_the_line_a_singular_system_passes_as_converged(dtype):
    """The plants discriminate: with no line (``FLOOR_STOP_TOLERANCES`` infinite, the stop before it) the solve of a
    singular system stops unflagged, at its floor, at a true residual the line refuses, in either arm."""
    line = solve_module.FLOOR_STOP_TOLERANCES * default_tolerance(dtype)
    for restarts in (0, STALL_RESTARTS):
        solved = [_singular(dtype, restarts, float("inf"))(c) for c in INCONSISTENT]
        assert any(report[FLAGS].sum() == 0 and residual > line for report, residual in solved), (restarts, solved)
