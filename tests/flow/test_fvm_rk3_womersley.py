"""F1b's gate 2: Womersley flow at alpha 6 through RK3's node, D/h 32, the default band and interior, six sub-cell
offsets of the axis batched (the tube's centre is a leaf), RK3 at ``nu dt / h^2 = 0.15`` (298 steps a period), the
drive's waveform read at each stage's time; started from the exact profile, six periods, the sixth's first harmonic fitted by S1's
Bessel fit over the clean lumen (SciPy, independent of the prescribed flow's series) with the viscosity, the radius and
**an amplitude** free. **Its line: S1's absolute 0.5 %** on ``|nu_fit / nu - 1|``; the 1 % viscosity plant (the viscous
term at 1.01 nu, the penalty at nu) must fail it at every offset.

**Why the amplitude is free (measured, F1b's build).** The built path drives the fluid alone (``(1 - chi) g``, round 4
item 6), which the spike did not: with S1's two-parameter fit the unplanted runs read -0.479 % to -0.113 % (S1's code,
the drive everywhere: -0.155 % to +0.207 %; the node with the drive everywhere reproduces that to the digit), so the
0.5 % line held with 0.02 % to spare at the offset (0, 0) on both sides. The band's share of the drive scales the
profile; a free amplitude takes it, and leaves the viscosity's own signature (the Stokes layer's thickness and phase):
unplanted -0.093 % to +0.308 %, the plant still +0.999 % to +1.003 % (planted from +0.910 %). The wall's position has
its own gate (F1a's gate 3), the flux its own report. Earlier (F1b's first step): float32 with the integer clock within
3.8e-7 of float64; with the drive held at the step's end instead of read at the stages, +5.4 %. At D/h 16 the fit cannot
separate a 1 % error (the tableau's band: -0.78 % to +0.49 % unplanted), hence D/h 32."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
import scipy.special
from fvm_cases import in_dtype

from mime.nodes.flow._waveform import waveform
from mime.nodes.flow.fvm import ExplicitFlowNode, _rk3, periodic_tube_layout

NU, R, ALPHA, DH = 0.005, 0.5, 6.0, 32
OMEGA = ALPHA ** 2 * NU / R ** 2
G0 = OMEGA
OFFSETS = ((0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.3, 0.1), (0.5, 0.5), (0.1, 0.4))
LINE, PERIODS = 5e-3, 6


def womersley(r, nu, radius):
    beta = np.exp(-1j * np.pi / 4) * np.sqrt(OMEGA / nu)
    return np.where(r < radius, (G0 / (1j * OMEGA)) * (1 - scipy.special.jv(0, beta * r) / scipy.special.jv(0, beta * radius)),
                    0.0)


def _nodes():
    base = periodic_tube_layout(R, 4 / DH, cells_across=DH)
    h = base.spacing
    steps = math.ceil(2 * math.pi / OMEGA / (0.15 * h * h / NU))
    dt = 2 * math.pi / OMEGA / steps
    layouts = [periodic_tube_layout(R, 4 / DH, cells_across=DH, offset_cells=o) for o in OFFSETS]
    node = ExplicitFlowNode("flow", dt, cells_x=base.cells[0], cells_y=base.cells[1], cells_z=base.cells[2],
                            spacing=h, viscosity=NU, density=1.0, tube_centre=base.centre, tube_axis=base.axis,
                            tube_radius=R)
    return node, layouts, steps, dt


@functools.cache
def _harmonics(dtype: str, plant: float):
    """Each offset's last-period first harmonic of the axial velocity (averaged along the tube), and its radii."""
    with in_dtype(dtype):
        node, layouts, steps, dt = _nodes()
        p = node._step_constants(None)  # noqa: SLF001
        grid = node.grid(p)
        points = np.asarray(grid.face_points(2, "float64"))
        radii = [np.hypot(points[..., 0, 0] - lay.centre[0], points[..., 0, 1] - lay.centre[1]) for lay in layouts]
        starts = []
        for r in radii:
            w = np.real(womersley(r, NU, R))[..., None] * np.ones((1, 1, grid.cells[2]))
            starts.append(w)
        start = dict(node.initial_state())
        dtype_ = start["velocity_z"].dtype
        coefficients = waveform((0.0, 0.0, 1.0), jnp.asarray(0.0, dtype=dtype_), G0, OMEGA / (2 * math.pi), 0.0)

        original = _rk3.step

        def planted(grid, velocity, pressure, viscosity, resistivity, force, convection, time, dt_):
            return original(grid, velocity, pressure, plant * viscosity, resistivity, force, convection, time, dt_)

        def one(centre, w0):
            q = {**p, "tube_centre": centre}
            s = {**start, "velocity_z": w0.astype(dtype_)}

            def step(state, _):
                ended = node.advance(q, state, {"waveform": coefficients}, dt)
                return ended, jnp.mean(ended["velocity_z"], axis=2)

            s, _ = jax.lax.scan(step, s, None, length=(PERIODS - 1) * steps)
            _, history = jax.lax.scan(step, s, None, length=steps)
            return history

        _rk3.step = planted
        try:
            centres = jnp.asarray([lay.centre for lay in layouts], dtype=dtype_)
            histories = jax.jit(jax.vmap(one))(centres, jnp.asarray(np.stack(starts), dtype=dtype_))
        finally:
            _rk3.step = original
    times = ((PERIODS - 1) * steps + 1 + np.arange(steps)) * dt
    weights = (2.0 / steps) * np.exp(-1j * OMEGA * times)
    amplitudes = np.tensordot(np.asarray(histories, np.float64), weights, axes=([1], [0]))
    return amplitudes, radii, grid.spacing


def fitted(dtype: str, plant: float = 1.0) -> list[float]:
    amplitudes, radii, h = _harmonics(dtype, plant)
    found = []
    for amplitude, r in zip(amplitudes, radii, strict=True):
        clean = r < R - 2.0 * h  # the band's half-width (h) and a cell
        y, x = amplitude[clean], r[clean]

        def residual(v):
            d = v[2] * womersley(x, v[0] * NU, R + v[1] * h) - y
            return np.concatenate([d.real, d.imag])

        solution = scipy.optimize.least_squares(residual, [1.0, 0.0, 1.0], method="lm", x_scale=[0.01, 0.1, 0.01],
                                                diff_step=1e-7)
        found.append(float(solution.x[0] - 1.0))
    return found


def test_womersley_at_alpha_six_fits_its_viscosity_within_half_a_percent_at_every_offset(dtype):
    assert max(abs(e) for e in fitted(dtype)) < LINE


def test_control_a_one_percent_viscosity_plant_fails_the_line_at_every_offset(dtype):
    assert min(abs(e) for e in fitted(dtype, 1.01)) >= LINE


def test_the_plant_moves_the_fit_by_one_percent_the_fits_resolution(dtype):
    """The same-dt pairing, kept as the fit's resolution (round 4, item 4), not as the gate."""
    shifts = [b - a for a, b in zip(fitted(dtype), fitted(dtype, 1.01), strict=True)]
    assert all(0.009 < s < 0.011 for s in shifts)


def test_float32_with_the_integer_clock_agrees_with_float64():
    a64, _, _ = _harmonics("float64", 1.0)
    a32, _, _ = _harmonics("float32", 1.0)
    assert np.max(np.abs(a64 - a32)) / np.max(np.abs(a64)) < 1e-5


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
