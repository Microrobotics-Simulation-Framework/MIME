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


def _rings(gap: float, beta: float, most: float, least_around: int = 8) -> list[tuple[float, float, int]]:
    """Polar rings from the sphere's pole nearest the wall: (polar angle, its step, points round it), the step
    ``beta sqrt(gap + 1 - cos theta)`` and at most ``most``: the lubrication's scale near the contact, even away from
    it."""
    out = [(0.0, beta * math.sqrt(gap), 1)]
    theta = 0.0
    while True:
        step = min(most, beta * math.sqrt(gap + 1.0 - math.cos(theta)))
        theta += step
        if theta >= math.pi - 0.5 * step:
            break
        around = max(least_around, int(math.ceil(2.0 * math.pi * math.sin(theta) / step / 4.0)) * 4)
        out.append((theta, step, around))
    out.append((math.pi, min(most, beta * math.sqrt(gap + 2.0)), 1))
    return out


def _ring_points(rings, centre: np.ndarray, radius_of, stagger: bool = False) -> np.ndarray:
    points = []
    for k, (theta, step, around) in enumerate(rings):
        offset = (0.5 if (stagger and k % 2) else 0.0) * 2.0 * math.pi / around
        phi = offset + 2.0 * math.pi * np.arange(around) / around
        n = np.stack([math.sin(theta) * np.cos(phi), math.sin(theta) * np.sin(phi), -math.cos(theta) * np.ones(around)],
                     1)
        points.append(centre + radius_of(step) * n)
    return np.concatenate(points)


def _motions(x: np.ndarray, centre: np.ndarray) -> np.ndarray:
    arm = x - centre
    eye = np.eye(3)
    columns = [np.broadcast_to(eye[i], arm.shape) for i in range(3)] + [np.cross(eye[i], arm) for i in range(3)]
    return np.stack(columns, -1).reshape(-1, 6)


#: The MFS's settings: ``beta`` and ``most`` set its rings' steps (``_rings``), ``depth`` the sources' depth in local
#: steps; the collocation rings 1.5 times denser.
MFS_SETTINGS = {"coarse": {"beta": 0.35, "most": 0.12, "depth": 1.5}, "fine": {"beta": 0.25, "most": 0.085, "depth": 1.5}}


def mfs(gap: float, *, beta: float, most: float, depth: float, oversample: float = 1.5) -> tuple[np.ndarray, float, int]:
    """The method of fundamental solutions for the unit sphere ``gap`` radii above the wall ``z = 0`` at unit viscosity:
    singular Stokeslets with Blake's image (``free_stokeslet`` and ``blake``), ``depth`` local steps inside the
    surface on rings graded towards the contact, collocated in least squares on staggered rings ``oversample`` times
    denser. Its 6 x 6 resistance about the centre (the force and torque on the fluid per unit motion), its collocation
    residual's largest entry, and its count of sources. It shares no code with MIME."""
    centre = np.array([0.0, 0.0, 1.0 + gap])
    sources = _ring_points(_rings(gap, beta, most), centre, lambda step: 1.0 - depth * step)
    collocation = _ring_points(_rings(gap, beta / oversample, most / oversample), centre, lambda step: 1.0, stagger=True)
    kernel = (free_stokeslet(collocation, sources, 0.0) + blake(collocation, sources)).transpose(0, 2, 1, 3).reshape(
        3 * len(collocation), 3 * len(sources))
    rhs = _motions(collocation, centre)
    forces, *_ = np.linalg.lstsq(kernel, rhs, rcond=None)
    residual = float(np.abs(kernel @ forces - rhs).max())
    return _motions(sources, centre).T @ forces, residual, len(sources)
