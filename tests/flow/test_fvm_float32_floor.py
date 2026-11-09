"""Float32's default stop at the solve's rounding floor (``mime.nodes.flow.fvm._solve.rounding_floor``): where 1e-5 is
out of float32's reach the solve stops at its floor, unflagged, its values float64's within the tolerance; where the
conjugate gradient's recurrence stalls above the floor float32's periodic restart from the true residual breaks the
stall. The cases are the solid's interior spike's float32 runs that the fixed 1e-5 flagged."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from fvm_cases import Stenosis, in_dtype, tube_node
from test_fvm_experiment import experiment

from mime.dynamics import Band
from mime.nodes.flow.fvm import Jacobi, Stopping, projected_solve
from mime.nodes.flow.fvm._band import DEFAULT_BAND, band_named, wall_fields
from mime.nodes.flow.fvm._solve import REPORT_FIELDS, REPORT_SIZE, default_restart_every, default_tolerance, \
    rounding_floor, stagnation_window
from mime.units import mm, ms

STAGNATED, RESIDUAL = REPORT_FIELDS.index("stagnated"), REPORT_FIELDS.index("residual")


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
                            window=stagnation_window(grid.cells, 20),
                            **{"restart_every": default_restart_every(run), **stop})

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
    above the floor (1.0e-4 against 4.7e-6, measured): a stall, not a floor. Float32's default restart from the true
    residual every 100 iterations breaks it: the solve converges, unflagged, at its tolerance or floor, its flux
    float64's within the tolerance. The control is the CG without the restart, which stops stagnated above its floor,
    so the case tells a stall from a floor."""
    tolerance = default_tolerance("float32")
    report, floor, flux, _ = _stenosis_solve("float32", tolerance, at_floor=True)
    assert report[2:5].sum() == 0.0, report
    assert report[RESIDUAL] <= max(floor, tolerance), (report, floor)
    _, _, reference, _ = _stenosis_solve("float64", 1e-12, at_floor=False)
    assert abs(flux - reference) <= tolerance * abs(reference)
    stalled, stalled_floor, _, residual = _stenosis_solve("float32", tolerance, at_floor=True, restart_every=0)
    assert stalled[STAGNATED] == 1.0, stalled
    assert residual > max(stalled_floor, tolerance)
