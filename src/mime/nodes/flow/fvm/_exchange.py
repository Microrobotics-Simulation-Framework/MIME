"""The exchange law between a field on the grid and the points of its exchange partners (F3; the record's B8, body half
point 2, and C5): a gather ``G`` and a scatter ``S`` by one Gaussian kernel whose weights are the grid kind's
(:meth:`MACGrid.exchange_stencil`), a transpose pair in the grid's inner product, ``<S f, u> = f . G u``, to rounding.

**Two readings, two deliveries.** The kernel's average (``mode="average"``) and its Laplacian (``mode="laplacian"``,
``L w``): a reading ``G u + c G L u`` is the kernel ``(I + c L) w``'s, and the delivery ``S f + L S (c f)`` its transpose,
so that a Faxén coefficient ``c`` the partner computes (``a^2/6 - sigma^2/2`` for a sphere of radius ``a``: the
Gaussian's own second moment already reads ``(sigma^2/2) lap u``) keeps the pair workless. The coefficient is the
partner's (it may be a traced value), never the kernel's: the kernel carries only its width, a structural setting.

**The kernel's own field in a free box** (SF-3, the undisturbed-velocity correction): the steady self-response of the
kernel by the same operator on a free periodic box, with Hasimoto's image term, held as three dimensionless numbers of
the width alone (:func:`self_mobility`): ``G S``, ``G L S`` and ``G L L S``, so that ``G (I + a L)(I + b L) S`` for any
two coefficients is a sum of the three, made in the program from the partner's traced radius and the fluid's viscosity.

Room for a couple's reading and delivery (the kernel's derivative, its antisymmetric dipole) is left in the modes: a
mode names which weights a reading or a delivery uses, so a new mode is a new set of weights on the same stencil."""

from __future__ import annotations

import functools
import math
from typing import Any, NamedTuple

import numpy as np
from jax import Array

from mime.core import jnp
from mime.nodes.flow.fvm._grid import ExchangeStencil, MACGrid

#: The weights a reading or a delivery uses: the kernel's average, or its seven-point Laplacian.
MODES = ("average", "laplacian")
#: Hasimoto's constant: a periodic array of point forces at period ``L`` moves at ``F (1 - c/L ...)/(6 pi mu a)``, so
#: a free box's steady self-response gains ``c / (6 pi mu L)`` back [recalled: Hasimoto 1959, 2.837297].
HASIMOTO = 2.837297
#: The free box's side, in kernel widths, on which :func:`self_mobility` is taken: wide enough that its images beyond
#: Hasimoto's term are about 1e-5 of the response (at sigma = h: 64, 96 and 128 cells gave 0.0307188, 0.0307169 and
#: 0.0307165, falling as the cube of the side) [measured, F3's first step].
FREE_BOX_WIDTHS = 128


def _axis_weights(stencil: ExchangeStencil, mode: str) -> list[tuple[Array, Array, Array]]:
    """The separable products a mode's 3-D weights are the sum of, each as its three 1-D factors."""
    w, c = stencil.weight, stencil.curvature
    if mode == "average":
        return [(w[:, 0], w[:, 1], w[:, 2])]
    if mode == "laplacian":
        return [(c[:, 0], w[:, 1], w[:, 2]), (w[:, 0], c[:, 1], w[:, 2]), (w[:, 0], w[:, 1], c[:, 2])]
    raise ValueError(f"an exchange mode is one of {MODES}, not {mode!r}")


def gather(grid: MACGrid, component: int, field: Array, stencil: ExchangeStencil, mode: str = "average") -> Array:
    """``G u`` at each point (``(n,)``): the cell volume times the sum over the stencil's faces of the mode's weights
    times the ``component``-th face set's values ``field``."""
    index = stencil.lattice
    values = field[index[:, 0, :, None, None], index[:, 1, None, :, None], index[:, 2, None, None, :]]
    total = jnp.zeros(index.shape[0], dtype=field.dtype)
    for wx, wy, wz in _axis_weights(stencil, mode):
        total = total + jnp.einsum("nabc,na,nb,nc->n", values, wx, wy, wz)
    return grid.cell_volume * total


def scatter(grid: MACGrid, component: int, values: Array, stencil: ExchangeStencil, mode: str = "average") -> Array:
    """``S f`` on the ``component``-th face set (the grid's shape): each point's value ``values`` (``(n,)``) times the
    mode's weights, summed onto the faces; a force (N) becomes a force density (N/m^3). The transpose of
    :func:`gather` in :meth:`MACGrid.inner`."""
    index = stencil.lattice
    density = jnp.zeros(grid.cells, dtype=values.dtype)
    for wx, wy, wz in _axis_weights(stencil, mode):
        spread = values[:, None, None, None] * wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        density = density.at[index[:, 0, :, None, None], index[:, 1, None, :, None],
                             index[:, 2, None, None, :]].add(spread)
    del component
    return density


def faxen(radius: Any, width: Any, share: float) -> Any:
    """The coefficient a sphere of ``radius`` reads its field's Laplacian with beyond the kernel's own: ``share a^2 -
    sigma^2/2`` (Faxén: ``share`` 1/6 for the drag, 1/10 for the volume average), ``width`` the kernel's ``sigma``."""
    return share * radius ** 2 - 0.5 * width ** 2


@functools.lru_cache(maxsize=16)
def self_mobility(width_cells: float, side: int | None = None) -> tuple[float, float, float]:
    """The kernel's steady self-response in a free box by the grid kind's operators, as three dimensionless numbers
    ``(m00, m01, m11)``: ``G S``, ``G L S`` and ``G L L S`` along one axis for a unit force, in units of ``1/(mu h)``,
    ``1/(mu h^3)`` and ``1/(mu h^5)``, Hasimoto's image term added to the first. A function of the kernel's width in
    cells alone (a structural setting), made once in float64 on the host. Its isotropy and its dependence on where in
    the cell the point sits are the floor sweep's to measure; it is taken at the cell's centre plus a generic offset."""
    side = side or int(2 ** math.ceil(math.log2(FREE_BOX_WIDTHS * max(width_cells, 1.0))))
    grid = MACGrid((side, side, side), 1.0)
    point = np.full((1, 3), side / 2.0) + np.array([[0.37, 0.21, 0.11]])
    stencil = grid.exchange_stencil(2, jnp.asarray(point, dtype=jnp.float64), float(width_cells))
    unit = jnp.ones(1, dtype=jnp.float64)
    response = steady_free(grid, scatter(grid, 2, unit, stencil))
    laplacian_response = steady_free(grid, scatter(grid, 2, unit, stencil, "laplacian"))
    m00 = float(gather(grid, 2, response, stencil)[0]) + HASIMOTO / (6.0 * math.pi * side)
    m01 = float(gather(grid, 2, response, stencil, "laplacian")[0])
    m11 = float(gather(grid, 2, laplacian_response, stencil, "laplacian")[0])
    return m00, m01, m11


def steady_free(grid: MACGrid, density: Array, component: int = 2) -> Array:
    """The ``component``-th face set of the steady Stokes response (``mu = 1``) to a force density on that face set
    alone, on a free periodic box: ``u = P (-L)^-1 f``, its mean removed (the box's net force balanced by a uniform
    pressure gradient: Hasimoto's), by the fast transform; the projection and the Laplacian commute on the periodic
    MAC grid, so the transform of each is diagonal."""
    nx, ny, nz = grid.cells
    h = grid.spacing
    kx = np.fft.fftfreq(nx)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kz = np.fft.rfftfreq(nz)[None, None, :]
    along = (4.0 / h ** 2) * np.sin(np.pi * (kx, ky, kz)[component]) ** 2
    symbol = (4.0 / h ** 2) * (np.sin(np.pi * kx) ** 2 + np.sin(np.pi * ky) ** 2 + np.sin(np.pi * kz) ** 2)
    safe = np.where(symbol == 0.0, 1.0, symbol)
    # -L's symbol is ``symbol``; the projection keeps 1 - |d_c|^2 / symbol of a field on face set c alone (D's symbol
    # along c is d_c = (e^(2 pi i k) - 1)/h, G's -conj(d_c), D G's -symbol).
    kept = np.where(symbol == 0.0, 0.0, (1.0 - along / safe) / safe)
    transformed = np.fft.rfftn(np.asarray(density, dtype=np.float64))
    return jnp.asarray(np.fft.irfftn(transformed * kept, s=grid.cells))


#: The free box the spectral sums run over, in cells a side: the kernel's own field over the fitted range stays inside
#: it (the samples stop where the field's diffusion length reaches a quarter of the side).
SPECTRUM_SIDE = 64
#: The self-history's fixed decay modes: Laplacian eigenvalues (1/h^2) log-spaced from below the box's smallest to the
#: grid's largest, each the decay ``1/(1 + r lambda)`` a step; their amplitudes are fitted (:func:`self_history`).
HISTORY_MODES = 16
HISTORY_LAMBDAS = (1e-3, 12.0)
#: The fit's samples: the steps after a unit force at which the free box's response is matched, log-spaced; and the
#: singular values, relative to the largest, below which the least squares drops a direction, by the run's dtype.
HISTORY_SAMPLES = 48
HISTORY_RCOND = {"float64": 1e-12, "float32": 1e-6}


class KernelSpectrum(NamedTuple):
    """The kernel's free-box spectrum along one axis's face set (``h = mu = rho = 1``): each wavenumber's weight ``W =
    |w_hat|^2 P_cc / N`` and the seven-point Laplacian's eigenvalue there, made in the program from the kernel's width
    (traced), so that every self-response below is a sum over it and differentiable in the width."""

    weight: Array
    eigenvalue: Array


def kernel_spectrum(width_cells: Any, side: int = SPECTRUM_SIDE, component: int = 2,
                    offset: tuple[float, float, float] = (0.37, 0.21, 0.11)) -> KernelSpectrum:
    """The spectrum of the kernel of ``width_cells`` cells on a periodic box of ``side`` cells: the kernel is separable,
    so its transform is the product of three one-dimensional ones; the projection keeps ``1 - |d_c|^2 / lambda`` of a
    field on face set ``component``. The mean (wavenumber zero) is removed, as an unbounded fluid has none."""
    dtype = jnp.asarray(width_cells).dtype if hasattr(width_cells, "dtype") else jnp.zeros(()).dtype
    index = jnp.arange(side, dtype=dtype)
    frequency = jnp.fft.fftfreq(side).astype(dtype)
    sines = 4.0 * jnp.sin(jnp.pi * frequency) ** 2
    powers: list[Array] = []
    for axis in range(3):
        shift = 0.0 if axis == component else 0.5
        centre = side / 2.0 + offset[axis] - shift
        distance = index - centre
        line = jnp.exp(-0.5 * (distance / width_cells) ** 2)
        line = line / jnp.sum(line)
        transform = jnp.fft.fft(line)
        powers.append(jnp.real(transform * jnp.conj(transform)))
    eigenvalue = sines[:, None, None] + sines[None, :, None] + sines[None, None, :]
    along = (sines[:, None, None], sines[None, :, None], sines[None, None, :])[component]
    safe = jnp.where(eigenvalue == 0.0, 1.0, eigenvalue)
    kept = jnp.where(eigenvalue == 0.0, 0.0, 1.0 - along / safe)
    weight = powers[0][:, None, None] * powers[1][None, :, None] * powers[2][None, None, :] * kept / side ** 3
    return KernelSpectrum(weight, eigenvalue)


def paired(spectrum: KernelSpectrum, reading: Any, delivery: Any) -> Array:
    """Each wavenumber's weight for a reading with Faxén coefficient ``reading`` of a delivery with ``delivery`` (both
    in cells squared): the kernel ``(I + c L) w`` transforms to ``(1 - c lambda) w_hat``."""
    lam = spectrum.eigenvalue
    return spectrum.weight * (1.0 - reading * lam) * (1.0 - delivery * lam)


def steady_response(spectrum: KernelSpectrum, reading: Any = 0.0, delivery: Any = 0.0) -> Array:
    """The kernel's steady self-response in a free box (``1/(mu h)`` units): ``sum W / lambda`` with Hasimoto's term
    for the box's images. Equal to :func:`self_mobility`'s by the same operator (its oracle)."""
    lam = spectrum.eigenvalue
    side = lam.shape[0]
    safe = jnp.where(lam == 0.0, 1.0, lam)
    weights = paired(spectrum, reading, delivery)
    return jnp.sum(jnp.where(lam == 0.0, 0.0, weights / safe)) + HASIMOTO / (6.0 * math.pi * side)


def step_response(spectrum: KernelSpectrum, r: Any, steps: Array, reading: Any = 0.0, delivery: Any = 0.0) -> Array:
    """The kernel's own field read ``steps`` backward-Euler steps after a unit force held for one step, from rest
    (``1/(mu h)`` units; ``r = nu dt / h^2``): ``r sum W (1 + r lambda)^-(j+1)``. Step 0 is the one-step mobility
    ``M_c`` the coupled resistance is made from."""
    weights = paired(spectrum, reading, delivery)
    decay = 1.0 / (1.0 + r * spectrum.eigenvalue)
    exponent = jnp.asarray(steps).astype(decay.dtype) + 1.0
    return r * jnp.sum(weights[None] * decay[None] ** exponent[:, None, None, None], axis=(1, 2, 3))


class SelfHistory(NamedTuple):
    """The kernel's own field after a force, as fixed decay modes: the field read ``j`` steps after a unit force is
    ``sum_m amplitude_m decay_m^(j+1)`` (``1/(mu h)`` units), fitted; ``residual`` is the fit's largest miss over its
    samples, relative to the one-step mobility; ``span`` the last step the samples reach."""

    amplitude: Array
    decay: Array
    residual: Array
    span: Array


def self_history(spectrum: KernelSpectrum, r: Any, reading: Any = 0.0, delivery: Any = 0.0) -> SelfHistory:
    """**Fitted**: the free box's response of :func:`step_response` matched by :data:`HISTORY_MODES` modes whose decays
    are fixed (``1/(1 + r lambda_m)``, ``lambda_m`` log-spaced over :data:`HISTORY_LAMBDAS`) and whose amplitudes are
    linear least squares over log-spaced samples up to where the field's diffusion length reaches a quarter of the box,
    with the modes' total held to the steady response (the tail beyond the samples). Deterministic and traced: a
    derivative in the width, ``r`` or the coefficients passes through the fit. N3 reads the grid's share from it."""
    dtype = spectrum.weight.dtype
    side = spectrum.eigenvalue.shape[0]
    span = jnp.maximum((side / 4.0) ** 2 / r, 8.0)
    fractions = jnp.linspace(0.0, 1.0, HISTORY_SAMPLES, dtype=dtype)
    samples = jnp.floor(jnp.expm1(fractions * jnp.log1p(span)))
    targets = step_response(spectrum, r, samples, reading, delivery)
    lambdas = jnp.asarray(np.geomspace(*HISTORY_LAMBDAS, HISTORY_MODES), dtype=dtype)
    decay = 1.0 / (1.0 + r * lambdas)
    basis = decay[None, :] ** (samples[:, None] + 1.0)
    scale = targets[0]
    total = steady_response(spectrum, reading, delivery)
    rows = jnp.concatenate([basis / scale, (decay / (1.0 - decay))[None, :] / total * 10.0])
    right = jnp.concatenate([targets / scale, jnp.ones(1, dtype=dtype) * 10.0])
    amplitude = jnp.linalg.lstsq(rows, right, rcond=HISTORY_RCOND[np.dtype(dtype).name])[0]
    residual = jnp.max(jnp.abs(basis @ amplitude - targets)) / scale
    return SelfHistory(amplitude, decay, residual, span)


__all__ = ["FREE_BOX_WIDTHS", "HASIMOTO", "HISTORY_LAMBDAS", "HISTORY_MODES", "HISTORY_RCOND", "HISTORY_SAMPLES",
           "MODES", "SPECTRUM_SIDE", "KernelSpectrum", "SelfHistory", "faxen", "gather", "kernel_spectrum", "paired",
           "scatter", "self_history", "self_mobility", "steady_free", "step_response", "steady_response"]
