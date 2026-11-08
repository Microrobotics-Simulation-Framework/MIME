"""Oracles for the plane wall kind that share no code with MIME's image (``mime.nodes.hydro.mrs._plane``): Blake's (1971)
singular image of a Stokeslet above the plane ``z = 0``, written out again from its formula, with its own check (its
sum with the singular Stokeslet vanishes on the wall); the regularised free Stokeslet, written again; Brenner's (1961)
exact series for a sphere moving along the plane's normal; and the draws the property tests take (a plane, a source
above it, a target on it or in the medium) at a body's physical scales."""

import math

import numpy as np


def free_stokeslet(targets: np.ndarray, sources: np.ndarray, eps_squared: float) -> np.ndarray:
    """The regularised Stokeslet at unit viscosity, ``(T, S, 3, 3)``: ``((r^2 + 2 eps^2) I + r r^T) / (8 pi (r^2 +
    eps^2)^(3/2))`` (Cortez 2005), written again; ``eps_squared`` 0 is the singular Stokeslet."""
    r = targets[:, None, :] - sources[None, :, :]
    r2 = np.sum(r * r, axis=-1)
    scale = 1.0 / (8.0 * math.pi * (r2 + eps_squared) ** 1.5)
    return scale[..., None, None] * (r[..., :, None] * r[..., None, :]
                                     + (r2 + 2.0 * eps_squared)[..., None, None] * np.eye(3))


def blake(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Blake's singular image above the wall ``z = 0`` at unit viscosity, ``(T, S, 3, 3)``: ``-S0(R) + 2 h M_jk
    d/dR_k [h R_i / |R|^3 - S0_i3(R)]``, ``R`` from the source's mirror image, ``M = diag(1, 1, -1)``, ``S0`` the
    singular Stokeslet times ``8 pi``; the derivative written out term by term."""
    h = sources[None, :, 2]
    R = targets[:, None, :] - sources[None, :, :] * np.array([1.0, 1.0, -1.0])
    d = np.sqrt(np.sum(R * R, axis=-1))
    eye = np.eye(3)
    m = np.array([1.0, 1.0, -1.0])
    image = -(eye / d[..., None, None] + R[..., :, None] * R[..., None, :] / (d ** 3)[..., None, None])
    # d/dR_k [h R_i / d^3]: h (delta_ik / d^3 - 3 R_i R_k / d^5)
    first = h[..., None, None] * (eye / (d ** 3)[..., None, None]
                                  - 3.0 * R[..., :, None] * R[..., None, :] / (d ** 5)[..., None, None])
    # d/dR_k S0_i3 = d/dR_k [delta_i3 / d + R_i R_3 / d^3]
    r3 = R[..., 2]
    second = (-eye[2][None, None, :, None] * R[..., None, :] / (d ** 3)[..., None, None]
              + (eye * r3[..., None, None] + R[..., :, None] * eye[2][None, None, None, :]) / (d ** 3)[..., None, None]
              - 3.0 * R[..., :, None] * (r3 / d ** 5)[..., None, None] * R[..., None, :])
    return (image + 2.0 * h[..., None, None] * (first - second) * m) / (8.0 * math.pi)


def brenner(gap: float) -> float:
    """Brenner (1961): the force on a sphere moving along a plane wall's normal over Stokes's ``6 pi mu a U``, its
    surface ``gap`` radii from the wall; the series summed until its terms vanish in float64."""
    alpha = math.acosh(1.0 + gap)
    total = 0.0
    for n in range(1, int(60.0 / alpha) + 200):
        x = (2 * n + 1) * alpha
        if x >= 700.0:
            break
        sech = 2.0 * math.exp(-x) / (1.0 + math.exp(-2.0 * x))
        numerator = 2.0 * math.tanh(x) + (2 * n + 1) * math.sinh(2.0 * alpha) * sech
        denominator = 2.0 - 2.0 * sech - (2 * n + 1) ** 2 * math.sinh(alpha) ** 2 * sech
        total += n * (n + 1) / ((2 * n - 1) * (2 * n + 3)) * (numerator / denominator - 1.0)
    return 4.0 / 3.0 * math.sinh(alpha) * total


def frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The unit normal and two unit directions along the plane, orthogonal."""
    n = normal / np.linalg.norm(normal)
    helper = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return n, u, np.cross(n, u)
