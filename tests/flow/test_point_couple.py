"""F3's gates on the point body's couple (``Drag.Stokes``, the maintainer's row #349, point 1: two-way by default through
the kernel's antisymmetric dipole), in both dtypes (``tests/flow/predictions_f3_fix.json``): the couple's coupled margin
against the flow node's own step, and a torqued body's flow against the rotlet. Each control is a plant switched by
``MIME_PLANT_<NAME>``, its rate measured by ``.github/probe/detection_rate.py``; each draw is seeded by
``DETECTION_SEED``."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from point_cases import WATER_MU, WATER_RHO, FreeBox, plant, seed

from mime.core import run_dtype
from mime.mappings import gaussian_mac
from mime.nodes.flow.fvm import ImplicitFlowNode
from mime.nodes.flow.fvm import _exchange
from mime.nodes.hydro import _point_drag
from mime.nodes.hydro._point_drag import COUPLE, FLOOR, free_space_margins

#: Each dtype's line for the couple's margin as the build computes it against the flow node's own step, absolute: the
#: node's CG stops at a relative residual of 1e-10 (float64) and 1e-5 (float32), and the margin's terms are under one.
MARGIN_LINE = {"float64": 1e-6, "float32": 1e-3}
#: The rotlet's line, relative, from 3 to 8 cells (the MAC Laplacian's error near the source, the guess committed).
ROTLET_LINE = 3e-2


def _self_rotation(width: float, r: float, scheme: str) -> float:
    """The couple's self-rotation the flow node's own step gives (rad/s per N m): a unit couple about z scattered by
    the mapping kind at the spectrum's point of a free box of the spectrum's side, one step from rest (IMEX Euler, or
    the steady limit), read back by the dipole."""
    side, h = _exchange.box_side(width), 1e-5
    dt = r * h * h * WATER_RHO / WATER_MU
    tolerance = 1e-10 if run_dtype() == jnp.float64 else 1e-5
    flow = ImplicitFlowNode("flow", dt, cells_x=side, cells_y=side, cells_z=side, spacing=h, viscosity=WATER_MU,
                            density=WATER_RHO, tube_centre=(side * h / 2, side * h / 2, 0.0), tube_axis=(0.0, 0.0, 1.0),
                            tube_radius=10.0 * side * h, scheme=scheme, convection=False, tolerance=tolerance,
                            exchange=True)
    point = jnp.asarray([[(side / 2 + 0.37) * h, (side / 2 + 0.21) * h, (side / 2 + 0.11) * h]], dtype=run_dtype())
    unit = jnp.asarray([[0.0, 0.0, 1.0]], dtype=run_dtype())

    def kernel(c: int, direction: str):  # noqa: ANN202
        return gaussian_mac(nx=side, ny=side, nz=side, spacing=h, width_cells=width, component=c, mode="curl",
                            direction=direction, count=1)

    inputs = {f"exchange_{axis}": kernel(c, "scatter").apply(unit, geom=point) for c, axis in enumerate("xyz")}
    state = flow.update(dict(flow.initial_state()), inputs, dt)
    return float(sum(kernel(c, "gather").apply(state[f"velocity_{axis}"], geom=point)[0, 2]
                     for c, axis in enumerate("xyz")))


def _diagonal_spectrum(width_cells, side=_exchange.SPECTRUM_SIDE, component=2, offset=(0.37, 0.21, 0.11)):  # noqa: ANN001, ANN201
    """The plant: the couple's spectrum with the projector's diagonal alone (each face set projected as if alone)."""
    held = _exchange.couple_spectrum(width_cells, side, component, offset)
    # Recompute with the cross terms dropped: the norm term kept, the divergence term per face set.
    dtype = _exchange._dtype_of(width_cells)  # noqa: SLF001
    frequency = jnp.fft.fftfreq(side).astype(dtype)
    sines = 4.0 * jnp.sin(jnp.pi * frequency) ** 2
    shape = ((side, 1, 1), (1, side, 1), (1, 1, side))
    eigen = held.eigenvalue
    safe = jnp.where(eigen == 0.0, 1.0, eigen)
    total = jnp.zeros_like(eigen)
    for face in range(3):
        product = None
        for q in range(3):
            coefficient = -0.5 * _exchange.levi_civita(component, q, face)
            if coefficient:
                product = coefficient
                for axis in range(3):
                    product = product * jnp.fft.fft(_exchange._line(width_cells, side, axis, face, offset, dtype,  # noqa: SLF001
                                                                    slope=axis == q)).reshape(shape[axis])
        if product is not None:
            along = sines.reshape(shape[face])
            total = total + jnp.real(product * jnp.conj(product)) * (1.0 - along / safe)
    return _exchange.KernelSpectrum(jnp.where(eigen == 0.0, 0.0, total) / side ** 3, eigen, side)


def test_the_couples_margin_is_the_flow_nodes_own(dtype, monkeypatch):
    """Gate 10: the couple's coupled one-step margin as the build computes it (its spectrum) equals the one the flow
    node's own step gives (a unit couple scattered in a free box, one IMEX Euler step and the steady limit, read by the
    dipole), at a drawn width, radius and step; and over the width rule's domain it stays above the floor (the floor's
    rotational analogue, measured). The plant drops the projector's cross terms from the couple's spectrum
    (SPECTRUM)."""
    if plant("SPECTRUM"):
        monkeypatch.setattr(_point_drag, "couple_spectrum", _diagonal_spectrum)
    rng = np.random.default_rng(10000 + seed())
    width, ratio = float(rng.choice([1.0, 1.5])), float(rng.uniform(1.0, 1.7))
    r = float(10.0 ** rng.uniform(math.log10(0.03), math.log10(30.0)))
    h = 1e-5
    radius, dt = ratio * width * h, r * h * h * WATER_RHO / WATER_MU
    spin = 1.0 / (8.0 * math.pi * WATER_MU * radius ** 3)
    one_step, steady = _self_rotation(width, r, "imex_euler"), _self_rotation(width, r, "steady")
    measured = (spin + one_step - steady) / spin
    computed = free_space_margins(radius, WATER_MU, WATER_RHO, h, width, dt)[1]
    domain = [(w, q, rr) for w in (1.0, 1.5) for q in (1.0, 1.2, 1.4, 1.6, 1.7) for rr in (0.03, 0.3, 3.0, 30.0)]
    margins, errors = [], []
    for w, q, rr in domain:
        a, step = q * w * h, rr * h * h * WATER_RHO / WATER_MU
        margin = free_space_margins(a, WATER_MU, WATER_RHO, h, w, step)[1]
        lam = a / math.sqrt(WATER_MU / WATER_RHO * step)
        exact = 8.0 * math.pi * WATER_MU * a ** 3 * (1.0 + lam ** 2 / (3.0 * (1.0 + lam)))
        model = 8.0 * math.pi * WATER_MU * a ** 3 / margin
        margins.append(margin)
        errors.append(model / exact - 1.0)
    print(f"F3 gate 10 [{dtype}]: width {width:g}, a/sigma {ratio:.3g}, r {r:.3g}: margin by the node {measured:.8g}, "
          f"by the spectrum {computed:.8g}; over the rule's domain the couple's margin {min(margins):.3g} to "
          f"{max(margins):.3g}, its one-step resistance against the exact rotating sphere's {min(errors):+.3g} to "
          f"{max(errors):+.3g}")
    assert abs(measured - computed) < MARGIN_LINE[dtype], (measured, computed)
    assert min(margins) >= FLOOR, min(margins)


def _rotlet(points: np.ndarray, centre: np.ndarray, torque: np.ndarray, sigma: float, side: float,
            reach: int = 48) -> np.ndarray:
    """The flow of a couple spread as (1/2) curl (T g), g the continuum Gaussian of width ``sigma``, in a periodic box
    of side ``side`` (m), at ``points``: its Fourier series, ``-(1/2) sum (k x T) g_hat sin(k . r) / (mu k^2 L^3)``,
    whose free-space limit is the rotlet ``T x r / (8 pi mu r^3)`` beyond the kernel; the series is taken to ``reach``
    wavenumbers a side, where the kernel's transform is under exp(-40) for a box of 32 widths."""
    n = np.arange(-reach, reach + 1)
    k = 2.0 * np.pi / side * np.stack(np.meshgrid(n, n, n, indexing="ij"), axis=-1).reshape(-1, 3)
    k = k[np.any(k != 0.0, axis=1)]
    squared = np.sum(k * k, axis=1)
    weight = np.exp(-0.5 * squared * sigma ** 2) / (WATER_MU * squared * side ** 3)
    cross = np.cross(k, torque[None, :])
    phase = np.sin((points - centre[None, :]) @ k.T)
    return -0.5 * (phase * weight[None, :]) @ cross


def test_a_torqued_bodys_flow_is_the_rotlet(dtype):
    """Gate 11: a torqued point body in a free box, two-way, at steady state: the flow it scatters along two lines
    through it across its axis (the x faces along y, the y faces along x) is the rotlet's, in the periodic box (the
    continuum Gaussian dipole's Fourier series), from 3 to 8 cells, each line's own component (so the rotation's sense
    is checked, not its size alone). The plant flips the delivered couple's sign (COUPLE_SIGN)."""
    rng = np.random.default_rng(11000 + seed())
    box = FreeBox(32, float(rng.uniform(0.8, 1.2)), 50.0, mass_ratio=1e-3, torque=1e-16)
    if plant("COUPLE_SIGN"):
        stepped = box.drag.stepped

        def flipped(p, state, inputs, dt):  # noqa: ANN001, ANN202
            ended = dict(stepped(p, state, inputs, dt))
            ended[COUPLE] = -ended[COUPLE]
            return ended

        box.drag.stepped = flipped  # type: ignore[method-assign]
    for _ in range(40):
        box.step()
    h, side = box.h, box.cells
    centre = np.asarray(box.drag_state["points"], dtype=np.float64)[0]
    i0, j0, k0 = (int(round(centre[0] / h)), int(round(centre[1] / h)), int(math.floor(centre[2] / h)))
    rows, found = [], []
    for j in range(side):
        y = (j + 0.5) * h
        if 3.0 <= abs(y - centre[1]) / h <= 8.0:
            rows.append((i0 * h, y, (k0 + 0.5) * h))
            found.append(float(box.flow_state["velocity_x"][i0, j, k0]))
    across = len(rows)
    for i in range(side):
        x = (i + 0.5) * h
        if 3.0 <= abs(x - centre[0]) / h <= 8.0:
            rows.append((x, j0 * h, (k0 + 0.5) * h))
            found.append(float(box.flow_state["velocity_y"][i, j0, k0]))
    reference = _rotlet(np.asarray(rows), centre, np.asarray([0.0, 0.0, box.torque]), h, side * h)
    expected = np.concatenate([reference[:across, 0], reference[across:, 1]])
    missed = float(np.max(np.abs(np.asarray(found) - expected) / np.abs(expected)))
    print(f"F3 gate 11 [{dtype}]: the scattered flow against the rotlet, largest relative miss {missed:.3g} over "
          f"{len(rows)} faces 3 to 8 cells out")
    assert missed < ROTLET_LINE, missed


def test_the_rotlets_series_is_the_free_rotlet_beyond_the_kernel(float64):
    """The gate's oracle against its free-space closed form: in a box of 64 kernel widths, at 4 to 6 widths, the series
    is T x r / (8 pi mu r^3) to the box's images (a field linear in r, of relative size about (4 pi/3)(r/L)^3: 2e-3
    at 5 widths)."""
    torque = np.asarray([0.0, 0.0, 1e-16])
    points = np.asarray([[0.0, 5e-5, 0.0], [6e-5, 0.0, 0.0], [0.0, -4e-5, 0.0]])
    found = _rotlet(points, np.zeros(3), torque, 1e-5, 6.4e-4, reach=80)
    distance = np.linalg.norm(points, axis=1)
    closed = np.cross(torque[None, :], points) / (8.0 * math.pi * WATER_MU * distance[:, None] ** 3)
    assert np.allclose(found, closed, rtol=5e-3, atol=1e-6 * np.max(np.abs(closed))), (found, closed)


@pytest.mark.parametrize("width", [1.0, 2.5])
def test_the_couples_spectrum_is_the_direct_solve(float64, width):
    """The couple's spectrum (its self-rotation, steady) against the scattered density solved directly through the MAC
    grid's projector by the fast transform on the same box: one number two ways."""
    side = _exchange.box_side(width)
    spectrum = float(_exchange.steady_response(_exchange.couple_spectrum(jnp.asarray(width), side), images=False))
    grid = _exchange.MACGrid((side, side, side), 1.0)
    point = jnp.asarray([[side / 2 + 0.37, side / 2 + 0.21, side / 2 + 0.11]])
    stencils = [grid.exchange_stencil(c, point, width) for c in range(3)]
    dens = [np.asarray(_exchange.scatter(grid, c, jnp.asarray([[0.0, 0.0, 1.0]]), stencils[c], "curl"))
            for c in range(3)]
    k = np.fft.fftfreq(side)
    waves = np.meshgrid(k, k, k, indexing="ij")
    symbols = [np.exp(2j * np.pi * w) - 1.0 for w in waves]
    eigen = sum(4.0 * np.sin(np.pi * w) ** 2 for w in waves)
    safe = np.where(eigen == 0.0, 1.0, eigen)
    forces = [np.fft.fftn(d) for d in dens]
    divergence = sum(s * f for s, f in zip(symbols, forces, strict=True)) / safe
    velocity = [np.real(np.fft.ifftn(np.where(eigen == 0.0, 0.0, (f - np.conj(s) * divergence) / safe)))
                for s, f in zip(symbols, forces, strict=True)]
    direct = float(sum(np.asarray(_exchange.gather(grid, c, jnp.asarray(velocity[c]), stencils[c], "curl"))[0, 2]
                       for c in range(3)))
    assert abs(direct - spectrum) <= 1e-10 * abs(spectrum), (direct, spectrum)
