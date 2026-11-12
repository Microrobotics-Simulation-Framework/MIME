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

**The couple** (``mode="curl"``): the kernel's antisymmetric dipole. Its reading is the fluid's rotation
``(1/2) G curl u``, by the kernel's derivative (``ExchangeStencil.slope``) on each face set, a vector at each point; its
delivery spreads a couple ``T`` as the force density ``(1/2) curl (T w)``, the reading's transpose, so the couple's
exchange is workless as the force's is.

**The body's own field in the free box** (the self-history's route 2, :class:`OwnField`): each partner's own flow on a
free periodic box of the grid's spacing, held by its transform and advanced by the grid's own operator (the MAC
grid's Laplacian and projector are diagonal there), so that a reading of the flow without the body subtracts it
exactly; Hasimoto's image term beside it, relaxed at the box's slowest diffusion rate. Route 1 (:func:`self_history`)
fits fixed decay modes to the same response and shares its image term."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from jax import Array

from mime.core import jnp
from mime.core.solves import SolveSite, register, solving
from mime.nodes.flow.fvm._grid import ExchangeStencil, MACGrid

#: The weights a reading or a delivery uses: the kernel's average, its seven-point Laplacian, or its antisymmetric
#: dipole (a couple's: a vector at each point).
MODES = ("average", "laplacian", "curl")
#: The modes whose value at a point is a vector of three.
VECTOR_MODES = ("curl",)
#: Hasimoto's constant: a periodic array of point forces at period ``L`` moves at ``F (1 - c/L ...)/(6 pi mu a)``, so
#: a free box's steady self-response gains ``c / (6 pi mu L)`` back [recalled: Hasimoto 1959, 2.837297].
HASIMOTO = 2.837297
#: The free box's side, in kernel widths, on which :func:`self_mobility` is taken: wide enough that its images beyond
#: Hasimoto's term are about 1e-5 of the response (at sigma = h: 64, 96 and 128 cells gave 0.0307188, 0.0307169 and
#: 0.0307165, falling as the cube of the side) [measured, F3's first step].
FREE_BOX_WIDTHS = 128


def levi_civita(i: int, j: int, k: int) -> int:
    """The permutation symbol of three axes."""
    return (i - j) * (j - k) * (k - i) // 2


def _terms(stencil: ExchangeStencil, mode: str, component: int) -> list[tuple[float, int, tuple[Array, Array, Array]]]:
    """The separable products a mode's 3-D weights on the ``component``-th face set are the sum of: each a coefficient,
    the vector component it reads into or delivers from (``-1`` for a mode of one value), and its three 1-D factors.
    The couple's: ``-(1/2) eps_{t q c}`` on the kernel's derivative along ``q``, for the rotation's (and the couple's)
    component ``t``."""
    w, c, s = stencil.weight, stencil.curvature, stencil.slope
    if mode == "average":
        return [(1.0, -1, (w[:, 0], w[:, 1], w[:, 2]))]
    if mode == "laplacian":
        return [(1.0, -1, (c[:, 0], w[:, 1], w[:, 2])), (1.0, -1, (w[:, 0], c[:, 1], w[:, 2])),
                (1.0, -1, (w[:, 0], w[:, 1], c[:, 2]))]
    if mode == "curl":
        return [(-0.5 * levi_civita(t, q, component), t,
                 tuple(s[:, axis] if axis == q else w[:, axis] for axis in range(3)))  # type: ignore[misc]
                for t in range(3) for q in range(3) if levi_civita(t, q, component)]
    raise ValueError(f"an exchange mode is one of {MODES}, not {mode!r}")


def gather(grid: MACGrid, component: int, field: Array, stencil: ExchangeStencil, mode: str = "average") -> Array:
    """``G u`` at each point (``(n,)``; ``(n, 3)`` for the couple's mode, the fluid's rotation): the cell volume times
    the sum over the stencil's faces of the mode's weights times the ``component``-th face set's values ``field``."""
    index = stencil.lattice
    values = field[index[:, 0, :, None, None], index[:, 1, None, :, None], index[:, 2, None, None, :]]
    vector = mode in VECTOR_MODES
    total = jnp.zeros((index.shape[0], 3) if vector else index.shape[0], dtype=field.dtype)
    for coefficient, column, (wx, wy, wz) in _terms(stencil, mode, component):
        read = coefficient * jnp.einsum("nabc,na,nb,nc->n", values, wx, wy, wz)
        total = total.at[:, column].add(read) if vector else total + read
    return grid.cell_volume * total


def scatter(grid: MACGrid, component: int, values: Array, stencil: ExchangeStencil, mode: str = "average") -> Array:
    """``S f`` on the ``component``-th face set (the grid's shape): each point's value ``values`` (``(n,)``; ``(n, 3)``
    for the couple's mode, a couple) times the mode's weights, summed onto the faces, each face's over its weight in
    the grid's inner product where the grid weighs its faces (an open box's planes carry half a cell, ``weights``); a
    force (N) becomes a force density (N/m^3). The transpose of :func:`gather` in :meth:`MACGrid.inner`, the planes
    included: a plane's face holds half a cell's fluid, so the density there is twice the whole cell's for the same
    momentum (without it the pair missed by 4.5e-4 at 3.2 cells from a plane and 7e-2 at 1.6, and the exchange did
    work the energy books did not pair)."""
    index = stencil.lattice
    density = jnp.zeros(grid.shape(component), dtype=values.dtype)
    for coefficient, column, (wx, wy, wz) in _terms(stencil, mode, component):
        each = coefficient * (values[:, column] if mode in VECTOR_MODES else values)
        spread = each[:, None, None, None] * wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        density = density.at[index[:, 0, :, None, None], index[:, 1, None, :, None],
                             index[:, 2, None, None, :]].add(spread)
    weighed = getattr(grid, "weights", None)
    return density if weighed is None else density / weighed(values.dtype)[component]


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


#: The free box the spectral sums and a body's own field run over, in cells a side: at least ``SPECTRUM_SIDE``, and at
#: least ``SIDE_WIDTHS`` kernel widths (to a power of two), so that the box's images beyond Hasimoto's term weigh alike
#: at every width (a 64-cell box fitted a 4-cell kernel's response to 3.7e-3 of a 128-cell box's: the point body's refuting review).
SPECTRUM_SIDE = 64
SIDE_WIDTHS = 32
#: The self-history's fixed decay modes (route 1): Laplacian eigenvalues (1/h^2) log-spaced from below the box's
#: smallest to the grid's largest, each the decay ``1/(1 + r lambda)`` a step; their amplitudes are fitted
#: (:func:`self_history`), and Hasimoto's image term is one more mode, route 2's own (:func:`image_mode`).
HISTORY_MODES = 16
HISTORY_LAMBDAS = (1e-3, 12.0)
#: The fit's samples: the steps after a unit force at which the free box's response is matched, log-spaced (its
#: residual is judged there and halfway between them); and the singular values, relative to the largest, below which
#: the least squares drops a direction, by the run's dtype.
HISTORY_SAMPLES = 48
HISTORY_RCOND = {"float64": 1e-12, "float32": 1e-6}


def box_side(width_cells: float) -> int:
    """The free box's side in cells for a kernel ``width_cells`` cells wide (:data:`SIDE_WIDTHS`)."""
    return max(SPECTRUM_SIDE, int(2 ** math.ceil(math.log2(SIDE_WIDTHS * max(float(width_cells), 1.0)))))


class KernelSpectrum(NamedTuple):
    """The kernel's free-box spectrum along one axis's face set (``h = mu = rho = 1``): each wavenumber's weight ``W =
    |w_hat|^2 P_cc / N`` and the seven-point Laplacian's eigenvalue there, so that every self-response below is a sum
    over it; on a box of ``side`` cells. Every response is a sum of the weights times a function of the eigenvalue
    alone, so a spectrum may be held aggregated by eigenvalue (:func:`aggregated`, one dimension)."""

    weight: Array
    eigenvalue: Array
    side: int


def _line(width_cells: Any, side: int, axis: int, face: int, offset: tuple[float, float, float], dtype: Any,
          slope: bool = False) -> Array:
    """The kernel's 1-D weights along ``axis`` for face set ``face`` on a box of ``side`` cells (summing to one), or
    with ``slope`` its derivative as :meth:`MACGrid.exchange_stencil` makes it (about the weights' own mean, over
    their own variance)."""
    index = jnp.arange(side, dtype=dtype)
    shift = 0.0 if axis == face else 0.5
    distance = index - (side / 2.0 + offset[axis] - shift)
    line = jnp.exp(-0.5 * (distance / width_cells) ** 2)
    line = line / jnp.sum(line)
    if not slope:
        return line
    centred = distance - jnp.sum(distance * line)
    return -centred * line / jnp.sum(centred * centred * line)


def _dtype_of(width_cells: Any) -> Any:
    return jnp.asarray(width_cells).dtype if hasattr(width_cells, "dtype") else jnp.zeros(()).dtype


def kernel_spectrum(width_cells: Any, side: int = SPECTRUM_SIDE, component: int = 2,
                    offset: tuple[float, float, float] = (0.37, 0.21, 0.11)) -> KernelSpectrum:
    """The spectrum of the kernel of ``width_cells`` cells on a periodic box of ``side`` cells: the kernel is separable,
    so its transform is the product of three one-dimensional ones; the projection keeps ``1 - |d_c|^2 / lambda`` of a
    field on face set ``component``. The mean (wavenumber zero) is removed, as an unbounded fluid has none."""
    dtype = _dtype_of(width_cells)
    frequency = jnp.fft.fftfreq(side).astype(dtype)
    sines = 4.0 * jnp.sin(jnp.pi * frequency) ** 2
    powers: list[Array] = []
    for axis in range(3):
        transform = jnp.fft.fft(_line(width_cells, side, axis, component, offset, dtype))
        powers.append(jnp.real(transform * jnp.conj(transform)))
    eigenvalue = sines[:, None, None] + sines[None, :, None] + sines[None, None, :]
    along = (sines[:, None, None], sines[None, :, None], sines[None, None, :])[component]
    safe = jnp.where(eigenvalue == 0.0, 1.0, eigenvalue)
    kept = jnp.where(eigenvalue == 0.0, 0.0, 1.0 - along / safe)
    weight = powers[0][:, None, None] * powers[1][None, :, None] * powers[2][None, None, :] * kept / side ** 3
    return KernelSpectrum(weight, eigenvalue, side)


def couple_spectrum(width_cells: Any, side: int = SPECTRUM_SIDE, component: int = 2,
                    offset: tuple[float, float, float] = (0.37, 0.21, 0.11)) -> KernelSpectrum:
    """The couple's spectrum about axis ``component`` (``h = mu = 1``): each wavenumber's weight ``F^H P F / N^3`` for
    the unit couple's force density ``(1/2) curl (e_c w)`` on the face sets across the axis, ``P`` the MAC grid's
    projector (its divergence's symbol ``e^(2 pi i k_c / N) - 1`` on face set ``c``), and the Laplacian's eigenvalue
    there: the kernel's self-rotation is a sum over it, as its self-mobility is over :func:`kernel_spectrum`'s."""
    dtype = _dtype_of(width_cells)
    frequency = jnp.fft.fftfreq(side).astype(dtype)
    sines = 4.0 * jnp.sin(jnp.pi * frequency) ** 2
    shape = ((side, 1, 1), (1, side, 1), (1, 1, side))
    forces: list[Any] = []
    for face in range(3):
        total: Any = 0.0
        for q in range(3):
            coefficient = -0.5 * levi_civita(component, q, face)
            if coefficient:
                product: Any = coefficient
                for axis in range(3):
                    product = product * jnp.fft.fft(_line(width_cells, side, axis, face, offset, dtype,
                                                          slope=axis == q)).reshape(shape[axis])
                total = total + product
        forces.append(total)
    divergence: Any = 0.0
    for face in range(3):
        divergence = divergence + (jnp.exp(2j * jnp.pi * frequency) - 1.0).reshape(shape[face]) * forces[face]
    eigenvalue = sines[:, None, None] + sines[None, :, None] + sines[None, None, :]
    safe = jnp.where(eigenvalue == 0.0, 1.0, eigenvalue)
    norm = sum(jnp.real(force * jnp.conj(force)) for force in forces if not isinstance(force, float))
    kept = jnp.where(eigenvalue == 0.0, 0.0, norm - jnp.real(divergence * jnp.conj(divergence)) / safe)
    return KernelSpectrum((kept / side ** 3).astype(dtype), eigenvalue, side)


def aggregated(spectrum: KernelSpectrum) -> KernelSpectrum:
    """``spectrum`` with its weights summed over the wavenumbers that share an eigenvalue (on the host, in its dtype):
    every response is a sum of the weights times a function of the eigenvalue alone, so this holds the same sums with
    some 6,500 terms on a 64-cell box in place of 262,144. A function of the kernel's width and the box alone, a
    structural setting, made once where the node is built."""
    eigenvalue = np.asarray(spectrum.eigenvalue).ravel()
    weight = np.asarray(spectrum.weight).ravel()
    keys, inverse = np.unique(np.round(eigenvalue.astype(np.float64), 10), return_inverse=True)
    summed = np.bincount(inverse.ravel(), weights=weight.astype(np.float64), minlength=keys.shape[0])
    dtype = weight.dtype
    return KernelSpectrum(jnp.asarray(summed, dtype=dtype), jnp.asarray(keys, dtype=dtype), spectrum.side)


@functools.lru_cache(maxsize=64)
def held_spectrum(kind: str, width_cells: float, side: int, dtype: str, component: int = 2,
                  offset: tuple[float, float, float] = (0.37, 0.21, 0.11)) -> KernelSpectrum:
    """A kernel's free-box spectrum aggregated by eigenvalue (:func:`aggregated`), ``kind`` ``"force"``
    (:func:`kernel_spectrum`) or ``"couple"`` (:func:`couple_spectrum`), made once a process for each width, box,
    dtype and place: a structural quantity the build's width ladder, its refusals' searches and every node read."""
    if np.dtype(dtype) != np.dtype(jnp.zeros(()).dtype):
        raise ValueError(f"a spectrum is made in the run's dtype, {jnp.zeros(()).dtype}, not {dtype}")
    made = kernel_spectrum if kind == "force" else couple_spectrum
    return aggregated(made(float(width_cells), side, component, offset))


def paired(spectrum: KernelSpectrum, reading: Any, delivery: Any) -> Array:
    """Each wavenumber's weight for a reading with Faxén coefficient ``reading`` of a delivery with ``delivery`` (both
    in cells squared): the kernel ``(I + c L) w`` transforms to ``(1 - c lambda) w_hat``."""
    lam = spectrum.eigenvalue
    return spectrum.weight * (1.0 - reading * lam) * (1.0 - delivery * lam)


def image_term(side: int) -> float:
    """Hasimoto's term for a periodic box of ``side`` cells (``1/(mu h)`` units): what its images take from a steady
    self-response, the unbounded fluid's less the box's."""
    return HASIMOTO / (6.0 * math.pi * side)


def steady_response(spectrum: KernelSpectrum, reading: Any = 0.0, delivery: Any = 0.0, images: bool = True) -> Array:
    """The kernel's steady self-response in a free box (``1/(mu h)`` units): ``sum W / lambda``, with Hasimoto's term
    for the box's images (``images``). Equal to :func:`self_mobility`'s by the same operator (its oracle)."""
    lam = spectrum.eigenvalue
    safe = jnp.where(lam == 0.0, 1.0, lam)
    weights = paired(spectrum, reading, delivery)
    return jnp.sum(jnp.where(lam == 0.0, 0.0, weights / safe)) + (image_term(spectrum.side) if images else 0.0)


def step_response(spectrum: KernelSpectrum, r: Any, steps: Array, reading: Any = 0.0, delivery: Any = 0.0) -> Array:
    """The kernel's own field read ``steps`` backward-Euler steps after a unit force held for one step, from rest
    (``1/(mu h)`` units; ``r = nu dt / h^2``): ``r sum W (1 + r lambda)^-(j+1)``. Step 0 is the one-step mobility
    ``M_c`` the coupled resistance is made from."""
    weights = jnp.reshape(paired(spectrum, reading, delivery), (-1,))
    decay = jnp.reshape(1.0 / (1.0 + r * spectrum.eigenvalue), (-1,))
    exponent = jnp.asarray(steps).astype(decay.dtype) + 1.0
    return r * jnp.sum(weights[None, :] * decay[None, :] ** exponent[:, None], axis=1)


def image_mode(r: Any, side: int) -> tuple[Any, Any]:
    """Hasimoto's image term as a decay mode a step, ``(amplitude, decay)``: read ``j`` steps after a unit force as
    ``amplitude decay^(j+1)``, its sum over every step the term itself (``1/(mu h)`` units), relaxed at the box's
    slowest diffusion rate (its smallest Laplacian eigenvalue, ``4 sin^2(pi / side)``). Route 2 holds it beside its own
    field; route 1 as one more mode, the same."""
    smallest = 4.0 * math.sin(math.pi / side) ** 2
    decay = 1.0 / (1.0 + r * smallest)
    return image_term(side) * r * smallest, decay


class SelfHistory(NamedTuple):
    """The kernel's own field after a force, as fixed decay modes: the field read ``j`` steps after a unit force is
    ``sum_m amplitude_m decay_m^(j+1)`` (``1/(mu h)`` units), fitted but the last (Hasimoto's, :func:`image_mode`);
    ``residual`` the fit's largest miss at its samples and halfway between them, relative to the one-step mobility;
    ``span`` the last step the samples reach."""

    amplitude: Array
    decay: Array
    residual: Array
    span: Array


#: The places in the cell (in cells, from the face's lattice point) besides the spectrum's own at which a fit's residual
#: is judged, on every face set: route 1's one set of modes serves every place and component, and at a kernel one cell
#: wide the response moves by up to 3e-4 of the one-step response between them (1e-9 at 1.5 cells) [measured, F3's fix
#: round], far above the fit's own miss there.
JUDGED_PLACES = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))


#: The self-history's fit: the modes' amplitudes (the decays are fixed) by linear least squares over the sampled
#: responses, made once a chunk in ``run_constants`` and held, so it is a fit of a model of the free box and no solve of the
#: flow; ``HISTORY_MODES`` unknowns (the image term is one more).
HISTORY_SITE = register(SolveSite(
    "point_self_history", "lstsq", "mime.nodes.flow.fvm._exchange.self_history", nodes=("PointDragNode",),
    exempt="a fit of the self-history's decay amplitudes to sampled responses of the free box, by the SVD's least squares, "
           "made once a chunk in run_constants and held: the arithmetic of a model's coefficients, not of the flow",
    bound=HISTORY_MODES + 1))


def self_history(spectrum: KernelSpectrum, r: Any, reading: Any = 0.0, delivery: Any = 0.0,
                 places: Sequence[KernelSpectrum] = ()) -> SelfHistory:
    """**Fitted** (route 1): the free box's response of :func:`step_response` matched by :data:`HISTORY_MODES` modes
    whose decays are fixed (``1/(1 + r lambda_m)``, ``lambda_m`` log-spaced over :data:`HISTORY_LAMBDAS`) and whose
    amplitudes are linear least squares over log-spaced samples up to where the field's diffusion length reaches a
    quarter of the box, with the modes' total held to the box's steady response (the tail beyond the samples), and
    Hasimoto's image term one more mode, route 2's (:func:`image_mode`). Its residual is judged at the samples and
    halfway between them, and against the responses at the other ``places`` (the kernel elsewhere in the cell or on
    another face set), whose spread the one fit carries. Deterministic and traced: a derivative in the width, ``r`` or
    the coefficients passes through the fit. N3 reads the grid's share from it."""
    dtype = spectrum.weight.dtype
    side = spectrum.side
    span = jnp.maximum((side / 4.0) ** 2 / r, 8.0)
    fractions = jnp.linspace(0.0, 1.0, HISTORY_SAMPLES, dtype=dtype)
    samples = jnp.floor(jnp.expm1(fractions * jnp.log1p(span)))
    between = jnp.floor(jnp.expm1(0.5 * (fractions[1:] + fractions[:-1]) * jnp.log1p(span)))
    targets = step_response(spectrum, r, samples, reading, delivery)
    lambdas = jnp.asarray(np.geomspace(*HISTORY_LAMBDAS, HISTORY_MODES), dtype=dtype)
    decay = 1.0 / (1.0 + r * lambdas)
    basis = decay[None, :] ** (samples[:, None] + 1.0)
    scale = targets[0]
    total = steady_response(spectrum, reading, delivery, images=False)
    rows = jnp.concatenate([basis / scale, (decay / (1.0 - decay))[None, :] / total * 10.0])
    right = jnp.concatenate([targets / scale, jnp.ones(1, dtype=dtype) * 10.0])
    with solving(HISTORY_SITE):
        amplitude = jnp.linalg.lstsq(rows, right, rcond=HISTORY_RCOND[np.dtype(dtype).name])[0]
    judged = decay[None, :] ** (between[:, None] + 1.0) @ amplitude - step_response(spectrum, r, between, reading,
                                                                                      delivery)
    residual = jnp.maximum(jnp.max(jnp.abs(basis @ amplitude - targets)), jnp.max(jnp.abs(judged))) / scale
    for place in places:
        residual = jnp.maximum(residual, jnp.max(jnp.abs(basis @ amplitude - step_response(place, r, samples, reading,
                                                                                           delivery))) / scale)
    image, image_decay = image_mode(r, side)
    return SelfHistory(jnp.concatenate([amplitude, jnp.reshape(image / image_decay, (1,)).astype(dtype)]),
                       jnp.concatenate([decay, jnp.reshape(image_decay, (1,)).astype(dtype)]), residual, span)


class OwnBox(NamedTuple):
    """Route 2's free box for one body, as its transform's half spectrum (``rfft``'s layout, ``(n, n, n//2 + 1)``):
    the seven-point Laplacian's eigenvalue (1/m^2), each face set's divergence symbol ``(e^(2 pi i k_c / n) - 1)/h``
    and the multiplicity of each wavenumber in the half spectrum; made once from the box's side and the grid's
    spacing (both structural)."""

    side: int
    spacing: float
    eigenvalue: Array
    symbols: tuple[Array, Array, Array]
    multiplicity: Array


def own_box(side: int, spacing: float, dtype: Any) -> OwnBox:
    """Route 2's free box of ``side`` cells of ``spacing`` (m) in ``dtype``, made in the program from its wavenumbers
    (no array of the box's size is held as a constant)."""
    complex_type = jnp.complex128 if np.dtype(dtype) == np.float64 else jnp.complex64
    full = jnp.fft.fftfreq(side).astype(dtype)
    half = jnp.fft.rfftfreq(side).astype(dtype)
    frequencies = (full[:, None, None], full[None, :, None], half[None, None, :])
    eigenvalue = sum(4.0 * jnp.sin(jnp.pi * k) ** 2 for k in frequencies) / spacing ** 2
    symbols = tuple(((jnp.exp(2j * jnp.pi * k) - 1.0) / spacing).astype(complex_type) for k in frequencies)
    multiplicity = jnp.where((half == 0.0) | (half == 0.5), 1.0, 2.0)[None, None, :].astype(dtype)
    return OwnBox(side, spacing, jnp.broadcast_to(eigenvalue, (side, side, side // 2 + 1)).astype(dtype),
                  (symbols[0], symbols[1], symbols[2]), multiplicity)


def own_kernels(box: OwnBox, point: Array, width_cells: float) -> list[Array]:
    """Each face set's kernel at ``point`` (m, the grid's coordinates) on route 2's box, by its transform (1/m^3): the
    stencil's separable weights (:meth:`MACGrid.exchange_stencil`, the run's own), each axis's 1-D sum of weights
    times phases at its lattice indices."""
    side = box.side
    grid = MACGrid((side, side, side), box.spacing)
    complex_type = jnp.complex128 if point.dtype == jnp.float64 else jnp.complex64
    wavenumbers = (np.arange(side), np.arange(side), np.arange(side // 2 + 1))
    shapes = ((side, 1, 1), (1, side, 1), (1, 1, side // 2 + 1))
    kernels: list[Array] = []
    for face in range(3):
        stencil = grid.exchange_stencil(face, point[None, :], width_cells)
        product: Any = 1.0
        for axis in range(3):
            angle = (2.0 * np.pi / side) * jnp.asarray(wavenumbers[axis], dtype=point.dtype)[:, None] \
                * stencil.lattice[0, axis][None, :].astype(point.dtype)
            weights = stencil.weight[0, axis]
            line = (jnp.cos(angle) @ weights) - 1j * (jnp.sin(angle) @ weights)
            product = product * line.astype(complex_type).reshape(shapes[axis])
        kernels.append(product)
    return kernels


def own_read(box: OwnBox, field: Array, kernels: Sequence[Array], coefficient: Any) -> Array:
    """The own field read by the kernel ``(I + c L) w`` at the body (m/s, ``(3,)``): ``h^3 sum_x w u`` by Parseval on
    the half spectrum; ``field`` holds each face set's transform's real and imaginary parts ``(2, 3, n, n, m)``."""
    factor = 1.0 - coefficient * box.eigenvalue
    scale = box.spacing ** 3 / box.side ** 3
    read = [scale * jnp.sum(box.multiplicity * factor * (jnp.real(kernel) * field[0, face]
                                                           + jnp.imag(kernel) * field[1, face]))
            for face, kernel in enumerate(kernels)]
    return jnp.stack(read)


def own_advance(box: OwnBox, field: Array, kernels: Sequence[Array], force: Array, coefficient: Any, decay: Array,
                gain: Array) -> Array:
    """The own field a step on: ``decay`` times it plus ``gain`` times the projected force density of ``force`` (N,
    ``(3,)``) spread by the kernel ``(I + c L) w`` (the grid's IMEX Euler step on the free box, Stokes: ``decay = 1/(1 +
    nu dt lambda)``, ``gain = dt/rho`` times it, both zero at wavenumber zero, whose mean the box removes)."""
    factor = 1.0 - coefficient * box.eigenvalue
    spread = [force[face] * factor * kernel for face, kernel in enumerate(kernels)]
    safe = jnp.where(box.eigenvalue == 0.0, 1.0, box.eigenvalue)
    divergence = sum(symbol * each for symbol, each in zip(box.symbols, spread, strict=True)) / safe
    projected = [each - jnp.conj(symbol) * divergence for symbol, each in zip(box.symbols, spread, strict=True)]
    held = field[0] + 1j * field[1]
    moved = jnp.stack([decay * held[face] + gain * projected[face] for face in range(3)])
    return jnp.stack([jnp.real(moved), jnp.imag(moved)]).astype(field.dtype)


__all__ = ["FREE_BOX_WIDTHS", "HASIMOTO", "HISTORY_LAMBDAS", "HISTORY_MODES", "HISTORY_RCOND", "HISTORY_SAMPLES",
           "JUDGED_PLACES", "MODES", "SIDE_WIDTHS", "SPECTRUM_SIDE", "VECTOR_MODES", "KernelSpectrum", "OwnBox", "SelfHistory",
           "aggregated", "box_side", "couple_spectrum", "held_spectrum", "faxen", "gather", "image_mode", "image_term", "kernel_spectrum",
           "levi_civita", "own_advance", "own_box", "own_kernels", "own_read", "paired", "scatter", "self_history",
           "self_mobility", "steady_free", "step_response", "steady_response"]
