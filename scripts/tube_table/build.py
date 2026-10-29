"""The unit tube's image table, built fast: the Fourier-Bessel series of ``tests/tube_fourier_bessel.py`` (Liron &
Shahar 1978, without the v0.2 builder's defect) restructured for a tensor-product grid.

The unit tube (radius 1, viscosity 1); the image of any tube is ``G(x / R) / (R mu)``. A source sits at
``(rho_s, 0, 0)``; a target at ``(rho_t, dphi, dz)`` in cylindrical coordinates. A block's rows are the velocity's
cylindrical components at the target (radial, azimuthal, axial), its columns the force's at the source, as the table
``mime.nodes.hydro.bem._tube`` reads holds them.

**The restructuring.** The wall's Fourier coefficients depend only on the source's radius, so one source radius is one
job. For each axial wavenumber ``k`` and angular mode ``n`` the wall condition fixes three coefficients per force
column; the target's side separates: modified Bessel functions per target radius, then the sum over modes against
``cos/sin(n dphi)`` and the sum over wavenumbers against ``cos/sin(k dz)``, both matrix products over the grid's angles
and offsets.

**Ratios, not values.** A target's mode enters only as ``I_n(k rho_t) / I_n(k R)``, computed from ``log I_n`` by the
ratios ``I_m / I_{m-1}`` (backward recurrence from an exact start), so no order or argument overflows or underflows;
the series' coefficients are solved in the system divided by ``I_n(k R)``. The series of the tests skips a mode whose
scaled wall Bessel function underflows (large ``n``, small ``k``); this one keeps it.

**On the axis** (``rho_t = 0``) the modes take their limits (``n I_n(k rho) / rho -> k / 2`` and ``I_n' -> 1 / 2``
for ``n = 1``, the others 0), the table's side of the axis.

**The plane image** (``plane``): Blake's (1971) image of the Stokeslet in a plane wall, closed form. With ``plane`` the
wall condition is that of the remainder, ``-(G_free + B)``, so the series computes the image minus ``B`` directly.
``plane`` is ``(angle, sign)``: the plane tangent to the wall at that angle, its image times ``sign``; ``(0, 1)`` is
the nearest wall's, the other two are controls."""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import fft, special

PREFACTOR = 1.0 / (8.0 * math.pi)
#: The angle parity of each block entry (rows the target's component, columns the force's): True where the entry goes
#: as ``sin(n dphi)``; and the axial parity, True where it goes as ``sin(k dz)``.
SIN_IN_ANGLE = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
SIN_ALONG = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=bool)
#: Each force column's system: the parity ``p`` in ``k`` and the sign ``sn`` its modes take ``n`` with.
COLUMNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0))
NEAREST: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class Truncation:
    """``modes`` (n = 0 .. modes), ``wavenumbers`` Gauss-Legendre nodes in ``t`` over ``[0, sqrt(k_max)]`` with
    ``k = t^2``, ``angles`` points round the wall for its Fourier coefficients."""

    modes: int
    wavenumbers: int
    k_max: float
    angles: int


def series_truncation(modes: int, wavenumbers: int, angles: int, largest_source_radius: float) -> Truncation:
    """The truncation ``tests/tube_fourier_bessel.image`` uses for these counts and its sources (its ``gap`` rule)."""
    gap = max(1.0 - largest_source_radius, 0.1)
    return Truncation(modes, wavenumbers, 30.0 / gap, angles)


def scaled_truncation(source_radius: float, readable_radius: float, *, modes: float, wavenumbers: float,
                      reach: float) -> Truncation:
    """A truncation on the scale of the pair's combined distance from the wall, ``H = (1 - rho_s) + (1 - rho_max)``:
    ``modes / H`` modes, ``wavenumbers / H`` nodes, ``k_max = reach / H``, and enough wall points that the
    trapezoid's aliasing is below ``exp(-25)``."""
    h = (1.0 - source_radius) + (1.0 - readable_radius)
    count = int(math.ceil(modes / h))
    angles = 2 * int(math.ceil((count + 25.0 / (1.0 - source_radius)) / 2.0))
    return Truncation(count, int(math.ceil(wavenumbers / h)), reach / h, max(angles, 2 * count + 2))


def wavenumber_nodes(truncation: Truncation) -> tuple[np.ndarray, np.ndarray]:
    t, weight = np.polynomial.legendre.leggauss(truncation.wavenumbers)
    t_max = math.sqrt(truncation.k_max)
    t, weight = 0.5 * t_max * (t + 1.0), 0.5 * t_max * weight
    return t ** 2, weight * 2.0 * t


# --- modified Bessel functions by ratios ---------------------------------------------------------------------------


def ratio_start(order: int, x: np.ndarray) -> np.ndarray:
    """``I_order(x) / I_{order-1}(x)``, exactly where the scaled functions are representable, else the lower bound
    ``x / (order + sqrt(order^2 + x^2))`` (there ``order >> x`` and the backward recurrence damps any error at once)."""
    x = np.asarray(x, dtype=np.float64)
    below = special.ive(order - 1, x)
    exact = special.ive(order, x) / np.where(below > 1e-280, below, 1.0)
    bound = x / (order + np.sqrt(order * order + x * x))
    return np.where(below > 1e-280, exact, bound)


def ladder_numpy(x: np.ndarray, start: np.ndarray, top: int) -> tuple[np.ndarray, np.ndarray]:
    """``(log I_n(x), I_{n+1}(x) / I_n(x))`` for ``n = 0 .. top``, each ``(top + 1, *x.shape)``, from ``start =
    I_{top+1}(x) / I_top(x)`` by the backward recurrence ``r_m = x / (2 m + x r_{m+1})`` (stable for ``I``)."""
    ratios = np.empty((top + 2, *x.shape))
    ratios[top + 1] = start
    for m in range(top, 0, -1):
        ratios[m] = x / (2.0 * m + x * ratios[m + 1])
    with np.errstate(divide="ignore"):
        logs = np.log(ratios[1:top + 1])
    log_i = np.empty((top + 1, *x.shape))
    log_i[0] = np.log(special.i0e(x)) + x
    log_i[1:] = log_i[0] + np.cumsum(logs, axis=0)
    return log_i, ratios[1:]


# --- the free Stokeslet and Blake's image, as sums of z^c |R|^-m ---------------------------------------------------


class Kernel:
    """A sum of ``coefficient * z**c * |R|**-m`` over terms ``(c, m)``, ``R = (Rx, Ry, z)``: the closed-form fields
    below, evaluated either in space or, along the axis, by their Fourier transform."""

    def __init__(self, terms: dict[tuple[int, int], Any] | None = None) -> None:
        self.terms = dict(terms or {})

    def __add__(self, other: Kernel) -> Kernel:
        out = dict(self.terms)
        for key, value in other.terms.items():
            out[key] = out[key] + value if key in out else value
        return Kernel(out)

    def scaled(self, factor: Any) -> Kernel:
        return Kernel({key: value * factor for key, value in self.terms.items()})

    def __sub__(self, other: Kernel) -> Kernel:
        return self + other.scaled(-1.0)

    def times_component(self, j: int, in_plane: Sequence[np.ndarray]) -> Kernel:
        if j == 2:
            return Kernel({(c + 1, m): value for (c, m), value in self.terms.items()})
        return self.scaled(in_plane[j])

    def in_space(self, z: np.ndarray, distance: np.ndarray) -> Any:
        return sum(value * z ** c * distance ** (-m) for (c, m), value in self.terms.items())

    def transformed(self, k: np.ndarray, d: np.ndarray) -> Any:
        """``int cos(k z) f dz`` for the terms even in z, ``int sin(k z) f dz`` for the odd ones, over the whole
        axis, at in-plane distance ``d``: ``C_m = 2 sqrt(pi) / Gamma(m / 2) (k / 2d)^nu K_nu(k d)``, ``nu = (m-1)/2``;
        a ``z`` factor gives ``k C_{m-2} / (m - 2)``, ``z^2`` gives ``C_{m-2} - d^2 C_m``."""
        cache: dict[int, np.ndarray] = {}

        def c_of(m: int) -> np.ndarray:
            if m not in cache:
                nu = (m - 1) // 2
                kd = k * d
                cache[m] = 2.0 * math.sqrt(math.pi) / math.gamma(m / 2.0) * (k / (2.0 * d)) ** nu * special.kv(nu, kd)
            return cache[m]

        total: Any = 0.0
        for (c, m), value in self.terms.items():
            if c == 0:
                total = total + value * c_of(m)
            elif c == 1:
                total = total + value * k * c_of(m - 2) / (m - 2)
            elif c == 2:
                total = total + value * (c_of(m - 2) - d * d * c_of(m))
            else:
                raise ValueError(f"no transform for z^{c}")
        return total


def _inverse(m: int) -> Kernel:
    return Kernel({(0, m): 1.0})


def stokeslet_kernels(in_plane: Sequence[np.ndarray]) -> list[list[Kernel]]:
    """``S_jk = delta_jk / |R| + R_j R_k / |R|^3`` (without ``1 / 8 pi``)."""
    out = []
    for j in range(3):
        row = []
        for k in range(3):
            term = _inverse(3).times_component(j, in_plane).times_component(k, in_plane)
            row.append(term + _inverse(1) if j == k else term)
        out.append(row)
    return out


def blake_kernels(in_plane: Sequence[np.ndarray], h: float, normal: np.ndarray) -> list[list[Kernel]]:
    """Blake's image of a Stokeslet a distance ``h`` from a plane wall, ``R`` from its mirror point, ``normal`` the
    wall's unit normal out of the fluid (in the xy plane): ``B_jk = -S_jk + 2 h M_kl d/dR_l [h R_j / |R|^3 - S_jm
    e_m]``, ``e = -normal`` (into the fluid), ``M = I - 2 normal normal^T`` (Blake 1971, eq. 15)."""
    e = -np.asarray(normal, dtype=np.float64)
    along_e = in_plane[0] * e[0] + in_plane[1] * e[1]
    mirror = np.eye(3) - 2.0 * np.outer(normal, normal)
    stokeslet = stokeslet_kernels(in_plane)

    def component(j: int) -> Kernel:
        return Kernel({(0, 0): 1.0}).times_component(j, in_plane)

    q = [[Kernel() for _ in range(3)] for _ in range(3)]
    for j in range(3):
        for el in range(3):
            # d/dR_l (h R_j / |R|^3) = h (delta_jl / |R|^3 - 3 R_j R_l / |R|^5)
            term = _inverse(5).times_component(j, in_plane).times_component(el, in_plane).scaled(-3.0 * h)
            if j == el:
                term = term + _inverse(3).scaled(h)
            # d/dR_l (e_j / |R| + R_j (R.e) / |R|^3)
            #   = -e_j R_l / |R|^3 + (delta_jl (R.e) + R_j e_l) / |R|^3 - 3 R_j (R.e) R_l / |R|^5
            derivative = _inverse(3).times_component(el, in_plane).scaled(-e[j])
            if j == el:
                derivative = derivative + _inverse(3).scaled(along_e)
            derivative = derivative + _inverse(3).times_component(j, in_plane).scaled(e[el])
            derivative = derivative + _inverse(5).times_component(j, in_plane).times_component(
                el, in_plane).scaled(-3.0 * along_e)
            q[j][el] = term - derivative
    out = []
    for j in range(3):
        row = []
        for k in range(3):
            entry = stokeslet[j][k].scaled(-1.0)
            for el in range(3):
                if mirror[k, el] != 0.0:
                    entry = entry + q[j][el].scaled(2.0 * h * mirror[k, el])
            row.append(entry)
        out.append(row)
    return out


def plane_geometry(source_radius: float, plane: tuple[float, float]) -> tuple[float, np.ndarray, np.ndarray]:
    """The source's distance from the plane tangent to the wall at ``plane``'s angle, the plane's outward normal and
    the source's mirror point in it."""
    angle = plane[0]
    normal = np.array([math.cos(angle), math.sin(angle), 0.0])
    h = 1.0 - source_radius * normal[0]
    mirror = np.array([source_radius, 0.0, 0.0]) + 2.0 * h * normal
    return h, normal, mirror


def blake_cylindrical(source_radius: float, rho: np.ndarray, phi: np.ndarray, z: np.ndarray,
                      plane: tuple[float, float]) -> np.ndarray:
    """``(T, F, Z, 3, 3)``: ``sign`` times Blake's image for ``plane``, at the grid's targets, rows cylindrical at the
    target, columns the source's (Cartesian at angle 0)."""
    h, normal, mirror = plane_geometry(source_radius, plane)
    x = rho[:, None, None] * np.cos(phi)[None, :, None] - mirror[0]
    y = rho[:, None, None] * np.sin(phi)[None, :, None] - mirror[1]
    zz = np.broadcast_to(z[None, None, :], (len(rho), len(phi), len(z)))
    x, y = np.broadcast_to(x, zz.shape), np.broadcast_to(y, zz.shape)
    distance = np.sqrt(x * x + y * y + zz * zz)
    kernels = blake_kernels((x, y), h, normal)
    cart = np.empty((*zz.shape, 3, 3))
    for j in range(3):
        for k in range(3):
            cart[..., j, k] = kernels[j][k].in_space(zz, distance)
    return plane[1] * PREFACTOR * _to_target_cylindrical(cart, phi)


def free_cylindrical(source_radius: float, rho: np.ndarray, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``(T, F, Z, 3, 3)``: the free Stokeslet on the grid, in the blocks' components."""
    x = rho[:, None, None] * np.cos(phi)[None, :, None] - source_radius
    y = rho[:, None, None] * np.sin(phi)[None, :, None]
    zz = np.broadcast_to(z[None, None, :], (len(rho), len(phi), len(z)))
    x, y = np.broadcast_to(x, zz.shape), np.broadcast_to(y, zz.shape)
    r = np.stack([x, y, zz], axis=-1)
    distance = np.linalg.norm(r, axis=-1)
    cart = (np.eye(3) / distance[..., None, None] + r[..., :, None] * r[..., None, :] / distance[..., None, None] ** 3)
    return PREFACTOR * _to_target_cylindrical(cart, phi)


def _to_target_cylindrical(cart: np.ndarray, phi: np.ndarray) -> np.ndarray:
    c, s = np.cos(phi)[None, :, None, None], np.sin(phi)[None, :, None, None]
    out = np.empty_like(cart)
    out[..., 0, :] = c * cart[..., 0, :] + s * cart[..., 1, :]
    out[..., 1, :] = -s * cart[..., 0, :] + c * cart[..., 1, :]
    out[..., 2, :] = cart[..., 2, :]
    return out


# --- the wall's side: one source radius --------------------------------------------------------------------------


def _wall_fields(source_radius: float, k: np.ndarray, angles: int,
                 plane: tuple[float, float] | None) -> list[list[np.ndarray]]:
    """``[column][row]``, each ``(K, angles)``: the axial Fourier transform of the field the image must cancel on the
    wall, the free Stokeslet's (plus the plane image's), its rows cylindrical at the wall point."""
    theta = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    cos_q, sin_q = np.cos(theta), np.sin(theta)
    wall = np.sqrt(1.0 + source_radius ** 2 - 2.0 * source_radius * cos_q)[None, :]
    dx, dy = (cos_q - source_radius)[None, :], sin_q[None, :]
    kk = k[:, None]
    k0, k1 = special.k0(kk * wall), special.k1(kk * wall)
    over = kk / wall
    g = np.empty((3, 3, len(k), angles))
    g[0, 0] = 2 * k0 + 2 * over * dx ** 2 * k1
    g[0, 1] = g[1, 0] = 2 * over * dx * dy * k1
    g[1, 1] = 2 * k0 + 2 * over * dy ** 2 * k1
    g[0, 2] = g[2, 0] = 2 * kk * dx * k0
    g[1, 2] = g[2, 1] = 2 * kk * dy * k0
    g[2, 2] = 4 * k0 - 2 * kk * wall * k1
    if plane is not None:
        h, normal, mirror = plane_geometry(source_radius, plane)
        mx, my = (cos_q - mirror[0])[None, :], (sin_q - mirror[1])[None, :]
        d = np.sqrt(mx * mx + my * my)
        kernels = blake_kernels((mx, my), h, normal)
        for j in range(3):
            for col in range(3):
                g[j, col] = g[j, col] + plane[1] * kernels[j][col].transformed(kk, d)
    g *= PREFACTOR
    return [[g[0, col] * cos_q + g[1, col] * sin_q, -g[0, col] * sin_q + g[1, col] * cos_q, g[2, col]]
            for col in range(3)]


def solve3(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Batched ``matrix^-1 rhs`` for ``(..., 3, 3)`` and ``(..., 3)``, by cofactors after scaling each column to unit
    largest entry."""
    scale = np.max(np.abs(matrix), axis=-2, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    a = matrix / scale
    c00 = a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1]
    c01 = a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2]
    c02 = a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0]
    det = a[..., 0, 0] * c00 + a[..., 0, 1] * c01 + a[..., 0, 2] * c02
    c10 = a[..., 0, 2] * a[..., 2, 1] - a[..., 0, 1] * a[..., 2, 2]
    c11 = a[..., 0, 0] * a[..., 2, 2] - a[..., 0, 2] * a[..., 2, 0]
    c12 = a[..., 0, 1] * a[..., 2, 0] - a[..., 0, 0] * a[..., 2, 1]
    c20 = a[..., 0, 1] * a[..., 1, 2] - a[..., 0, 2] * a[..., 1, 1]
    c21 = a[..., 0, 2] * a[..., 1, 0] - a[..., 0, 0] * a[..., 1, 2]
    c22 = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    b0, b1, b2 = rhs[..., 0], rhs[..., 1], rhs[..., 2]
    # a system singular to rounding (the axisymmetric mode at the smallest wavenumbers) is skipped, as the series of
    # the tests skips it (``LinAlgError``)
    usable = det != 0.0
    y = np.stack([c00 * b0 + c10 * b1 + c20 * b2, c01 * b0 + c11 * b1 + c21 * b2,
                  c02 * b0 + c12 * b1 + c22 * b2], axis=-1) / np.where(usable, det, 1.0)[..., None]
    return np.where(usable[..., None], y / scale[..., 0, :], 0.0)


def wall_system(kk: np.ndarray, nn: np.ndarray, r1: np.ndarray, r2: np.ndarray, p: float, sn: float) -> np.ndarray:
    """``(K, N, 3, 3)``: :func:`target_terms`' basis on the wall (``rho = 1``), divided by ``I_n(k)``."""
    system = np.zeros((*r1.shape, 3, 3))
    system[..., 0, 0] = sn * nn
    system[..., 0, 1] = p * r1
    system[..., 0, 2] = p * r1 * (nn + kk * r2)
    system[..., 1, 0] = -(nn + kk * r1)
    system[..., 1, 1] = p * sn * r1
    system[..., 1, 2] = -(nn + 2.0) * p * sn * r1
    system[..., 2, 1] = -1.0
    system[..., 2, 2] = -kk * r1
    return system


@dataclass
class WallSide:
    """One chunk of wavenumbers for one source radius: the nodes ``k`` and weights ``weight / pi``, the coefficients
    ``coefficients[k, n, a, column]`` of the system divided by ``I_n(k)``, and ``log I_n(k)`` as ``log_wall[n, k]``."""

    k: np.ndarray
    weight: np.ndarray
    coefficients: np.ndarray
    log_wall: np.ndarray


def wall_side(source_radius: float, k: np.ndarray, weight: np.ndarray, truncation: Truncation,
              plane: tuple[float, float] | None = None, solver: Callable[..., np.ndarray] = solve3) -> WallSide:
    top = truncation.modes
    if truncation.angles < 2 * top + 2:
        raise ValueError("the wall needs at least 2 modes + 2 points")
    fields = _wall_fields(source_radius, k, truncation.angles, plane)
    stacked = np.stack([fields[col][row] for col in range(3) for row in range(3)])
    spectrum = fft.rfft(stacked, axis=-1, workers=1)[..., :top + 1]
    w0 = np.full(top + 1, 2.0 / truncation.angles)
    w0[0] = 1.0 / truncation.angles
    by_cos, by_sin = spectrum.real * w0, -spectrum.imag * w0
    rhs = np.empty((len(k), top + 1, 3, 3))
    for col in range(3):
        for row in range(3):
            uses_sin = (row == 1) != (col == 1)
            rhs[:, :, row, col] = (by_sin if uses_sin else by_cos)[3 * col + row]
    log_wall, ratios = ladder_numpy(k, ratio_start(top + 2, k), top + 1)
    r1, r2 = ratios[:top + 1].T, ratios[1:top + 2].T  # (K, N): I_{n+1}/I_n and I_{n+2}/I_{n+1} at k
    kk, nn = k[:, None], np.arange(top + 1, dtype=np.float64)[None, :]
    coefficients = np.empty((len(k), top + 1, 3, 3))
    for col, (p, sn) in enumerate(COLUMNS):
        coefficients[..., col] = -solver(wall_system(kk, nn, r1, r2, p, sn), rhs[..., col])
    return WallSide(k, weight / np.pi, coefficients, log_wall[:top + 1])


# --- the target's side: Bessel functions per target radius, then two matrix products ------------------------------


def target_terms(xp: Any, rho: Any, side_k: Any, coefficients: Any, log_wall: Any, log_target: Any,
                 ratio1: Any, ratio2: Any, top: int) -> list[Any]:
    """The nine block entries ``u[row][col]``, each ``(N, T, K)``, of one chunk: the target's basis divided by
    ``I_n(k)``, times the wall's coefficients; on the axis the modes' limits.

    **The basis.** The series' three modes per ``(k, n)`` become parallel as ``k -> 0`` (its second is ``p k`` times
    its first to leading order), so the system solved in them loses about ``log10(n^3 / k^2)`` digits. Here they are
    recombined by a change of basis independent of the radius (so the image is the same): the first mode, then
    ``(a1 - p sn k a0) / k^2 = I_n (p r1, p sn r1, -1)`` and ``(a2 - (p sn n / k) a0) - (n + 2)`` times that,
    ``= I_n (p r1 (n + x r2), -(n + 2) p sn r1, -x r1)``, ``r1 = I_{n+1}/I_n``, ``r2 = I_{n+2}/I_{n+1}`` at
    ``x = k rho``, by the recurrences, with no cancellation; the system then loses at most about ``log10 n``."""
    k = side_k[None, None, :]
    n = xp.arange(top + 1, dtype=xp.float64)[:, None, None]
    rho3 = rho[None, :, None]
    on_axis = rho3 <= 0.0
    safe_rho = xp.where(on_axis, 1.0, rho3)
    x = k * rho3
    lam = xp.exp(log_target - log_wall[:, None, :])
    l1 = ratio1 * lam
    inverse_i1 = xp.exp(-log_wall[1])[None, None, :]
    a = xp.where(on_axis, xp.where(n == 1.0, 0.5 * k * inverse_i1, 0.0), n * lam / safe_rho)
    l2 = l1 * (n + x * ratio2)
    xl1 = x * l1
    out: list[list[Any]] = [[None] * 3 for _ in range(3)]
    for col, (p, sn) in enumerate(COLUMNS):
        c = [xp.transpose(coefficients[:, :, b, col])[:, None, :] for b in range(3)]  # (N, 1, K)
        out[0][col] = sn * a * c[0] + p * l1 * c[1] + p * l2 * c[2]
        out[1][col] = -(a + k * l1) * c[0] + p * sn * l1 * c[1] - (n + 2.0) * p * sn * l1 * c[2]
        out[2][col] = -lam * c[1] - xl1 * c[2]
    return [out[row][col] for row in range(3) for col in range(3)]


def contract(xp: Any, terms: list[Any], weight: Any, side_k: Any, phi: Any, z: Any, segments: Sequence[int]) -> Any:
    """``(S, T, F, Z, 9)``: the chunk's sums over modes (cumulative at each of ``segments``' ends) and wavenumbers."""
    n = xp.arange(segments[-1], dtype=xp.float64)
    angle = n[:, None] * phi[None, :]
    by_angle = (xp.cos(angle), xp.sin(angle))
    axial = side_k[:, None] * z[None, :]
    by_z = (xp.cos(axial) * weight[:, None], xp.sin(axial) * weight[:, None])
    results = []
    for e, u in enumerate(terms):
        row, col = divmod(e, 3)
        trig_a, trig_z = by_angle[int(SIN_IN_ANGLE[row, col])], by_z[int(SIN_ALONG[row, col])]
        parts, start = [], 0
        running = None
        for end in segments:
            v = xp.tensordot(u[start:end], trig_a[start:end], axes=([0], [0]))  # (T, K, F)
            g = xp.tensordot(v, trig_z, axes=([1], [0]))  # (T, F, Z)
            running = g if running is None else running + g
            parts.append(running)
            start = end
        results.append(xp.stack(parts))
    return xp.stack(results, axis=-1)


def _target_numpy(rho: np.ndarray, side: WallSide, top: int, phi: np.ndarray, z: np.ndarray,
                  segments: Sequence[int]) -> np.ndarray:
    x = rho[:, None] * side.k[None, :]
    log_target, ratios = ladder_numpy(x, ratio_start(top + 2, x), top + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = target_terms(np, rho, side.k, side.coefficients, side.log_wall, log_target[:top + 1],
                             ratios[:top + 1], ratios[1:top + 2], top)
    return contract(np, terms, side.weight, side.k, phi, z, segments)


def jax_target() -> Callable[..., Any]:
    """The target's side on JAX's default device (float64: ``JAX_ENABLE_X64=1``), for :func:`image_grid`'s
    ``device``: the start of each ratio ladder is computed here (scipy), the rest there."""
    import jax
    import jax.numpy as jnp

    def ladder(x: Any, start: Any, top: int) -> tuple[Any, Any]:
        def step(following: Any, m: Any) -> tuple[Any, Any]:
            ratio = x / (2.0 * m + x * following)
            return ratio, ratio

        _, below = jax.lax.scan(step, start, jnp.arange(top, 0, -1, dtype=jnp.float64))
        ratios = jnp.concatenate([below[::-1], start[None]], axis=0)
        log0 = jnp.log(jax.scipy.special.i0e(x)) + x
        log_i = jnp.concatenate([log0[None], log0[None] + jnp.cumsum(jnp.log(ratios[:top]), axis=0)], axis=0)
        return log_i, ratios

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def run(top: int, segments: tuple[int, ...], rho: Any, k: Any, weight: Any, coefficients: Any, log_wall: Any,
            start: Any, phi: Any, z: Any) -> Any:
        log_target, ratios = ladder(rho[:, None] * k[None, :], start, top + 1)
        terms = target_terms(jnp, rho, k, coefficients, log_wall, log_target[:top + 1], ratios[:top + 1],
                             ratios[1:top + 2], top)
        return contract(jnp, terms, weight, k, phi, z, segments)

    def target(rho: np.ndarray, side: WallSide, top: int, phi: np.ndarray, z: np.ndarray,
               segments: Sequence[int]) -> np.ndarray:
        start = ratio_start(top + 2, rho[:, None] * side.k[None, :])
        return np.asarray(run(top, tuple(segments), rho, side.k, side.weight, side.coefficients, side.log_wall,
                              start, phi, z))

    return target


def _chunks(count: int, rows: int, top: int, angles: int, budget: float) -> list[slice]:
    """Wavenumber chunks: ``budget`` elements per ``(mode, target, wavenumber)`` array, ``2e6`` per wall array."""
    size = max(1, min(int(budget // max(1, rows * (top + 1))), int(2e6 // angles)))
    return [slice(i, min(i + size, count)) for i in range(0, count, size)]


def image_grid(source_radius: float, rho: Sequence[float], phi: Sequence[float], z: Sequence[float],
               truncation: Truncation, *, plane: tuple[float, float] | None = None,
               mode_checkpoints: Sequence[int] | None = None, threads: int | None = None,
               device: Callable[..., Any] | None = None, budget: float = 4e6) -> np.ndarray:
    """``(T, F, Z, 3, 3)`` (or ``(S, T, F, Z, 3, 3)`` with ``mode_checkpoints``: the partial sums to each count of
    modes): the image (with ``plane``, the image minus that plane image) on the grid ``rho x phi x z`` for a source at
    ``(source_radius, 0, 0)``. ``device(rho, side, top, phi, z, segments)`` replaces the numpy target side (the GPU's);
    the wall's side always runs here, ``threads`` chunks at a time."""
    rho_a, phi_a, z_a = (np.asarray(v, dtype=np.float64) for v in (rho, phi, z))
    top = truncation.modes
    checkpoints = sorted({min(int(m), top) for m in (mode_checkpoints or [top])})
    segments = [m + 1 for m in checkpoints]
    k_all, w_all = wavenumber_nodes(truncation)
    pieces = _chunks(len(k_all), len(rho_a), top, truncation.angles, budget)
    target = device or _target_numpy

    def wall(piece: slice) -> WallSide:
        return wall_side(source_radius, k_all[piece], w_all[piece], truncation, plane)

    total = np.zeros((len(segments), len(rho_a), len(phi_a), len(z_a), 9))
    workers = threads or int(os.environ.get("TUBE_TABLE_THREADS", "0")) or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if device is None:
            for part in pool.map(lambda piece: target(rho_a, wall(piece), top, phi_a, z_a, segments), pieces):
                total += part
        else:
            for side in pool.map(wall, pieces):
                total += np.asarray(target(rho_a, side, top, phi_a, z_a, segments))
    total = total.reshape(*total.shape[:-1], 3, 3)
    return total if mode_checkpoints is not None else total[0]


def image_points(targets: np.ndarray, sources: np.ndarray, truncation: Truncation, **options: Any) -> np.ndarray:
    """``(Nt, Ns, 3, 3)`` Cartesian, as ``tests/tube_fourier_bessel.image`` returns it, from :func:`image_grid` pair
    by pair (for checks; slow for many points)."""
    out = np.empty((len(targets), len(sources), 3, 3))
    for j, source in enumerate(sources):
        rho_s, phi_s = math.hypot(source[0], source[1]), math.atan2(source[1], source[0])
        for i, target in enumerate(targets):
            rho_t, phi_t = math.hypot(target[0], target[1]), math.atan2(target[1], target[0])
            block = image_grid(rho_s, [rho_t], [phi_t - phi_s], [target[2] - source[2]], truncation,
                               threads=1, **options)[0, 0, 0]
            out[i, j] = _turn(phi_t).T @ block @ _turn(phi_s)
    return out


def _turn(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


# --- the table's read: mime.core.tables' cubic, in numpy -------------------------------------------------------------


def _slopes(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """The derivative at each knot of the parabola through it and its neighbours (at an end, the next two)."""
    count = len(x)
    if count == 2:
        secant = (f[..., 1] - f[..., 0]) / (x[1] - x[0])
        return np.stack([secant, secant], axis=-1)
    out = np.empty_like(f)
    for j in range(count):
        trio = (j - 1, j, j + 1) if 0 < j < count - 1 else (0, 1, 2) if j == 0 else (count - 3, count - 2, count - 1)
        total = 0.0
        for a in trio:
            b, c = (other for other in trio if other != a)
            total = total + f[..., a] * (2.0 * x[j] - x[b] - x[c]) / ((x[a] - x[b]) * (x[a] - x[c]))
        out[..., j] = total
    return out


def hermite_read(x: np.ndarray, f: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """The piecewise cubic Hermite read of ``f`` (knots along its last axis) at ``queries`` inside ``[x0, x_end]``."""
    slope = _slopes(x, f)
    cell = np.clip(np.searchsorted(x, queries, side="right") - 1, 0, len(x) - 2)
    width = x[cell + 1] - x[cell]
    t = (queries - x[cell]) / width
    t2, t3 = t * t, t * t * t
    return ((1 - 3 * t2 + 2 * t3) * f[..., cell] + (t - 2 * t2 + t3) * width * slope[..., cell]
            + (3 * t2 - 2 * t3) * f[..., cell + 1] + (t3 - t2) * width * slope[..., cell + 1])


def mirrored(x: np.ndarray, f: np.ndarray, parity: float, *, wraps: bool) -> tuple[np.ndarray, np.ndarray]:
    """The axis as the tube's table holds a symmetric one: two knots mirrored past 0 (and past pi when ``wraps``),
    their values times ``parity``, an odd component zero at the symmetric ends."""
    f = f.copy()
    if parity < 0:
        f[..., 0] = 0.0
        if wraps:
            f[..., -1] = 0.0
    xs = [-x[2:0:-1], x]
    fs = [parity * f[..., 2:0:-1], f]
    if wraps:
        xs.append(2.0 * np.pi - x[-2:-4:-1])
        fs.append(parity * f[..., -2:-4:-1])
    return np.concatenate(xs), np.concatenate(fs, axis=-1)


def graded_knots(length: float, near: float, step: float) -> np.ndarray:
    """Knots on ``[0, length]`` spaced about ``step * (x + near)``: geometric in ``x + near``, ending at ``length``."""
    count = max(2, int(math.ceil(math.log((length + near) / near) / math.log1p(step))))
    return near * (((length + near) / near) ** (np.arange(count + 1) / count) - 1.0)


__all__ = ["NEAREST", "Truncation", "blake_cylindrical", "free_cylindrical", "graded_knots", "hermite_read",
           "image_grid", "image_points", "mirrored", "scaled_truncation", "series_truncation", "solve3",
           "target_terms", "contract", "wall_side", "wavenumber_nodes"]
