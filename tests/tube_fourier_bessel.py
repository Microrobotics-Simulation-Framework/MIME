"""The image system of a Stokeslet inside an infinite circular tube with a no-slip wall, by its Fourier-Bessel series
(Liron & Shahar 1978): the oracle the tube's wall tables are checked against.

The v0.2 line's direct code (``cylinder_greens_function_v2._assemble_image_only``), ported by contract: the same
truncation (``modes`` Fourier modes in the angle, ``wavenumbers`` Gauss-Legendre nodes in the axial wavenumber after
the substitution ``k = t^2 / R``, ``angles`` points round the wall for the wall's Fourier coefficients) and the same
arithmetic, so that it reproduces the 2026-05-21 tables at their knots to rounding where they are what it computes.
What it is checked against is not itself: the image plus the free Stokeslet vanishes on the wall (the condition that
defines it); the image is divergence-free inside the tube and reciprocal, ``G(x, y) = G(y, x)^T``, as a Stokes flow's
Green's function is; and on the axis it gives the classical first wall correction of a sphere on the axis of a tube
(its coefficient 2.10444, Bohlin's, in Happel & Brenner 1965), all in ``tests/oracles/test_tube_wall.py``.

**The v0.2 builder's defect.** ``as_built=True`` keeps it, so that the 2026-05-21 tables can be held to what built
them; the oracle is the series without it (``image``'s docstring says what it is). Phase 7h found it: the no-slip
condition cannot see it, and the image it gives fails divergence and reciprocity.

Lengths are in any unit, the viscosity in the matching one; the image of a unit force is a velocity per force,
``1 / (8 pi mu)`` times a length to the minus one."""

from __future__ import annotations

import numpy as np
from scipy import special


def cylindrical(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each point's distance from the axis, its angle and its height, the tube's axis along z through the origin."""
    return np.hypot(points[:, 0], points[:, 1]), np.arctan2(points[:, 1], points[:, 0]), points[:, 2]


def to_cylindrical(angle: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)``: the rotation from Cartesian components to cylindrical ones (radial, azimuthal, axial) at each
    angle."""
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.zeros((len(angle), 3, 3))
    rotation[:, 0, 0], rotation[:, 0, 1] = c, s
    rotation[:, 1, 0], rotation[:, 1, 1] = -s, c
    rotation[:, 2, 2] = 1.0
    return rotation


def _scaled_bessel(n: int, s: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """``I_n`` and its derivative, each times ``exp(-s)``."""
    value = special.ive(n, s)
    if n == 0:
        return value, special.ive(1, s)
    inverse = np.where(np.asarray(s) > 1e-30, n / np.maximum(np.asarray(s), 1e-300), 0.0)
    return value, special.ive(n - 1, s) - inverse * value


def free_stokeslet(targets: np.ndarray, sources: np.ndarray, viscosity: float) -> np.ndarray:
    """``(Nt, Ns, 3, 3)``: the unbounded singular Stokeslet, the velocity at each target per unit force at each
    source, Cartesian."""
    r = targets[:, None, :] - sources[None, :, :]
    distance = np.sqrt(np.sum(r * r, axis=-1))
    inverse = 1.0 / np.maximum(distance, 1e-300)
    return (np.eye(3) * inverse[..., None, None] + r[..., :, None] * r[..., None, :] * inverse[..., None, None] ** 3) \
        / (8.0 * np.pi * viscosity)


def image(targets: np.ndarray, sources: np.ndarray, radius: float, viscosity: float, *, modes: int = 15,
          wavenumbers: int = 80, angles: int = 64, as_built: bool = False) -> np.ndarray:
    """``(Nt, Ns, 3, 3)``: the image velocity at each target per unit force at each source, Cartesian, for a tube of
    ``radius`` about the z axis. The tables were built at ``modes=15, wavenumbers=80, angles=64``.

    ``as_built`` computes it as the v0.2 builder did, which the 2026-05-21 tables hold: for a force along the source's
    azimuthal direction (the second column) its velocity's radial and axial components go as ``sin(n dphi)`` and its
    azimuthal one as ``cos(n dphi)``, the opposite parity in the angle to a radial force's, and the builder took the
    radial force's modes for it too (the same ``n`` where its parity makes it ``-n``). The modes then meet the wall's
    condition, which fixes their coefficients whatever the modes are, but are not Stokes flows inside it: that image
    is not divergence-free and not reciprocal. Without it, each column takes its own parity's modes."""
    rho_t, phi_t, z_t = cylindrical(targets)
    rho_s, phi_s, z_s = cylindrical(sources)
    bins, inverse = np.unique(np.round(rho_s / (1e-6 * radius)) * (1e-6 * radius), return_inverse=True)
    gap = max(radius - float(np.max(rho_s)), 0.1 * radius)
    t, weight = np.polynomial.legendre.leggauss(wavenumbers)
    t_max = np.sqrt(30.0 * radius / gap)
    t, weight = 0.5 * t_max * (t + 1.0), 0.5 * t_max * weight
    k_nodes, k_weights = t ** 2 / radius, weight * 2.0 * t / radius
    phi_q = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    cos_q, sin_q = np.cos(phi_q), np.sin(phi_q)
    turn_t, turn_s = to_cylindrical(phi_t), to_cylindrical(phi_s)
    dphi = phi_t[:, None] - phi_s[None, :]
    dz = z_t[:, None] - z_s[None, :]
    wall = np.maximum(np.sqrt(radius ** 2 + bins[:, None] ** 2 - 2.0 * radius * bins[:, None] * cos_q[None, :]), 1e-30)
    dx, dy = radius * cos_q[None, :] - bins[:, None], radius * sin_q[None, :]
    prefactor = 1.0 / (8.0 * np.pi * viscosity)
    total = np.zeros((len(targets), len(sources), 3, 3))
    for k, wk in zip(k_nodes, k_weights, strict=True):
        if k < 1e-15:
            continue
        k0, k1 = special.kv(0, k * wall), special.kv(1, k * wall)
        over = np.where(wall > 1e-20, k / wall, 0.0)
        gxx = prefactor * (2 * k0 + 2 * over * dx ** 2 * k1)
        gxy = prefactor * (2 * over * dx * dy * k1)
        gyy = prefactor * (2 * k0 + 2 * over * dy ** 2 * k1)
        gxz, gyz = prefactor * (2 * k * dx * k0), prefactor * (2 * k * dy * k0)
        gzz = prefactor * (4 * k0 - 2 * k * wall * k1)
        conditions = (
            (gxx * cos_q + gxy * sin_q, -gxx * sin_q + gxy * cos_q, gxz),
            (gxy * cos_q + gyy * sin_q, -gxy * sin_q + gyy * cos_q, gyz),
            (gxz * cos_q + gyz * sin_q, -gxz * sin_q + gyz * cos_q, gzz),
        )
        ckz, skz = np.cos(k * dz), np.sin(k * dz)
        for n in range(modes + 1):
            s_wall = k * radius
            i_wall, ip_wall = _scaled_bessel(n, s_wall)
            q_wall = (k ** 2 * radius ** 2 + n ** 2) / (k * radius) if s_wall > 1e-30 else 0.0
            if max(abs(float(i_wall)), abs(float(ip_wall))) < 1e-250:
                continue

            def system(p: float, n: int = n, i_wall: float = float(i_wall), ip_wall: float = float(ip_wall),
                       q_wall: float = q_wall, k: float = float(k)) -> np.ndarray:
                return np.array([[n * i_wall / radius, p * k ** 2 * ip_wall, p * q_wall * i_wall],
                                 [-k * ip_wall, -p * n * k * i_wall / radius, -p * n * ip_wall],
                                 [0.0, -k ** 2 * i_wall, -(2 * i_wall + k * radius * ip_wall)]])

            turned = n if as_built else -n
            try:
                transverse, axial = np.linalg.inv(system(1.0)), np.linalg.inv(system(-1.0))
                azimuthal_force = np.linalg.inv(system(1.0, n=turned))
            except np.linalg.LinAlgError:
                continue
            s_t = k * rho_t
            i_t, ip_t = _scaled_bessel(n, s_t)
            q_t = np.where(s_t > 1e-30, (k ** 2 * rho_t ** 2 + n ** 2) / np.maximum(k * rho_t, 1e-300), 0.0)
            n_over = np.where(rho_t > 1e-30, n * i_t / np.maximum(rho_t, 1e-300), 0.0)
            nk_over = np.where(rho_t > 1e-30, n * k * i_t / np.maximum(rho_t, 1e-300), 0.0)
            decay = np.exp(k * (rho_t - radius))

            def basis(p: float, i_t: np.ndarray = i_t, ip_t: np.ndarray = ip_t, q_t: np.ndarray = q_t,
                      n_over: np.ndarray = n_over, nk_over: np.ndarray = nk_over, n: int = n,
                      k: float = float(k)) -> np.ndarray:
                b = np.zeros((len(rho_t), 3, 3))
                b[:, 0, 0], b[:, 0, 1], b[:, 0, 2] = n_over, p * k ** 2 * ip_t, p * q_t * i_t
                b[:, 1, 0], b[:, 1, 1], b[:, 1, 2] = -k * ip_t, -p * nk_over, -p * n * ip_t
                b[:, 2, 1], b[:, 2, 2] = -k ** 2 * i_t, -(2 * i_t + k * rho_t * ip_t)
                return b

            cos_n, sin_n = np.cos(n * phi_q), np.sin(n * phi_q)
            w0 = 1.0 / angles if n == 0 else 2.0 / angles
            rhs = np.zeros((len(bins), 3, 3))
            for column, (radial, azimuthal, along) in enumerate(conditions):
                rhs[:, 0, column] = w0 * (radial @ (cos_n if column != 1 else sin_n))
                rhs[:, 1, column] = w0 * (azimuthal @ (sin_n if column != 1 else cos_n))
                rhs[:, 2, column] = w0 * (along @ (cos_n if column != 1 else sin_n))
            coefficients = np.zeros((len(bins), 3, 3))
            coefficients[:, :, 0] = -np.einsum("ab,sb->sa", transverse, rhs[:, :, 0])
            coefficients[:, :, 1] = -np.einsum("ab,sb->sa", azimuthal_force, rhs[:, :, 1])
            coefficients[:, :, 2] = -np.einsum("ab,sb->sa", axial, rhs[:, :, 2])
            per_source = coefficients[inverse]
            sign = 1.0 if as_built else -1.0
            u_t = decay[:, None, None, None] * np.concatenate([
                np.matmul(basis(1.0)[:, None, :, :], per_source[None, :, :, 0:1]),
                np.matmul(basis(1.0, n_over=sign * n_over, nk_over=sign * nk_over, n=turned)[:, None, :, :],
                          per_source[None, :, :, 1:2])], axis=-1)
            u_a = decay[:, None, None] * np.matmul(basis(-1.0)[:, None, :, :],
                                                   per_source[None, :, :, 2:3])[..., 0]
            cn, sn = np.cos(n * dphi), np.sin(n * dphi)
            block = np.empty((len(targets), len(sources), 3, 3))
            block[..., 0, 0], block[..., 1, 0], block[..., 2, 0] = (u_t[..., 0, 0] * cn * ckz, u_t[..., 1, 0] * sn * ckz,
                                                                    u_t[..., 2, 0] * cn * skz)
            block[..., 0, 1], block[..., 1, 1], block[..., 2, 1] = (u_t[..., 0, 1] * sn * ckz, u_t[..., 1, 1] * cn * ckz,
                                                                    u_t[..., 2, 1] * sn * skz)
            block[..., 0, 2], block[..., 1, 2], block[..., 2, 2] = (u_a[..., 0] * cn * skz, u_a[..., 1] * sn * skz,
                                                                    u_a[..., 2] * cn * ckz)
            total += (wk / np.pi) * block
    rotated = np.matmul(total, turn_s[None, :, :, :])
    return np.matmul(turn_t.transpose(0, 2, 1)[:, None, :, :], rotated)


__all__ = ["cylindrical", "free_stokeslet", "image", "to_cylindrical"]
