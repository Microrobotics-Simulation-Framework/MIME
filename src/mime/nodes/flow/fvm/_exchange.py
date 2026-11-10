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
from typing import Any

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
    index = stencil.index
    values = field[index[:, 0, :, None, None], index[:, 1, None, :, None], index[:, 2, None, None, :]]
    total = jnp.zeros(index.shape[0], dtype=field.dtype)
    for wx, wy, wz in _axis_weights(stencil, mode):
        total = total + jnp.einsum("nabc,na,nb,nc->n", values, wx, wy, wz)
    return grid.cell_volume * total


def scatter(grid: MACGrid, component: int, values: Array, stencil: ExchangeStencil, mode: str = "average") -> Array:
    """``S f`` on the ``component``-th face set (the grid's shape): each point's value ``values`` (``(n,)``) times the
    mode's weights, summed onto the faces; a force (N) becomes a force density (N/m^3). The transpose of
    :func:`gather` in :meth:`MACGrid.inner`."""
    index = stencil.index
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


__all__ = ["FREE_BOX_WIDTHS", "HASIMOTO", "MODES", "faxen", "gather", "scatter", "self_mobility", "steady_free"]
