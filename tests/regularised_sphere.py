"""A sphere's drag and rotational drag with the regularised Stokeslet at a blob ``eps`` and exact quadrature: the
continuous regularised problem, which a discretisation of the MRS rung converges to at that ``eps``, so that the
difference is the discretisation's error alone (the regularisation's is the difference from Stokes's law).

By the sphere's symmetry the first-kind equation for a translation maps the two poloidal fields of degree one,
``e_z`` and ``n_z n``, onto themselves (a 2 x 2 system whose entries are each density's velocity at the pole and at
the equator), and the rotation's toroidal field ``e_z x n`` onto itself (one number, its velocity at the equator).
Each velocity is a smooth surface integral, taken here by Gauss-Legendre panels in the polar angle about the target,
graded at the scale of ``eps``, and the trapezoid rule round it. Independent of MIME's code (numpy alone); its own
control is the limit ``eps -> 0``, Stokes's law (``test_mrs_separated.py``). The translation's construction is the
sim2real error budget's (2026-10-31), whose route (b) met it within 5.4e-6 at every blob it tried."""

import math

import numpy as np


def _panels(eps: float, per_panel: int) -> tuple[np.ndarray, np.ndarray]:
    edges = [0.0]
    step = eps / 8.0
    while edges[-1] + step < math.pi:
        edges.append(edges[-1] + step)
        step *= 1.6
    edges.append(math.pi)
    x, w = np.polynomial.legendre.leggauss(per_panel)
    nodes = [0.5 * (hi - lo) * x + 0.5 * (hi + lo) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    weights = [0.5 * (hi - lo) * w for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return np.concatenate(nodes), np.concatenate(weights)


def _velocity(target: np.ndarray, density, eps: float, per_panel: int, around: int) -> np.ndarray:
    """``8 pi mu u`` at ``target`` on the unit sphere from the traction ``density(y)``, at ``mu = 1``."""
    theta, w_theta = _panels(eps, per_panel)
    psi = 2.0 * math.pi * np.arange(around) / around
    axis = target / np.linalg.norm(target)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    y = (st[..., None] * (np.cos(psi)[None, :, None] * e1 + np.sin(psi)[None, :, None] * e2)
         + ct[..., None] * axis).reshape(-1, 3)
    weight = (w_theta[:, None] * np.sin(theta)[:, None] * np.full((1, around), 2.0 * math.pi / around)).reshape(-1)
    r = target - y
    r2 = np.sum(r * r, axis=1)
    t = density(y)
    u = ((r2 + 2 * eps * eps)[:, None] * t + r * np.sum(r * t, axis=1)[:, None]) / ((r2 + eps * eps) ** 1.5)[:, None]
    return np.sum(u * weight[:, None], axis=0)


def regularised_drag(eps_over_a: float, per_panel: int = 48, around: int = 32) -> float:
    """The translating sphere's drag over Stokes's ``6 pi mu a U``, at ``eps / a``."""
    pole, equator = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    densities = (lambda y: np.broadcast_to(np.array([0.0, 0.0, 1.0]), y.shape), lambda y: y[:, 2:3] * y)
    matrix = np.empty((2, 2))
    for column, density in enumerate(densities):
        at_pole = _velocity(pole, density, eps_over_a, per_panel, around)[2]
        at_equator = _velocity(equator, density, eps_over_a, per_panel, around)[2]
        matrix[0, column], matrix[1, column] = at_equator, at_pole - at_equator
    alpha, beta = np.linalg.solve(matrix, np.array([8.0 * math.pi, 0.0]))
    return float((alpha * 4.0 * math.pi + beta * 4.0 * math.pi / 3.0) / (6.0 * math.pi))


def regularised_rotational_drag(eps_over_a: float, per_panel: int = 48, around: int = 32) -> float:
    """The rotating sphere's torque over Stokes's ``8 pi mu a^3 Omega``, at ``eps / a``: the traction ``alpha e_z x
    n`` gives ``e_z x n`` at the equator for ``alpha = 8 pi / c``, ``c`` its velocity there at ``alpha = 1``, and its
    torque is ``alpha 8 pi / 3``."""
    equator = np.array([1.0, 0.0, 0.0])
    c = _velocity(equator, lambda y: np.cross(np.array([0.0, 0.0, 1.0]), y), eps_over_a, per_panel, around)[1]
    return float(8.0 * math.pi / c / 3.0)
