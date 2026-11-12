"""An exact oracle for a ball's silhouette through a pinhole without distortion, written apart from ``mime.sensing``:

- **the conic**: the silhouette is the ellipse the ball's tangent cone cuts from the image plane, its area and centre in
  closed form;
- **each pixel's covered area**, by integrating along the pixel's width the length of each vertical line inside both the
  ellipse and the pixel (a composite midpoint rule of ``SAMPLES`` points a pixel, its error under 1e-6 of a pixel);
- **each pixel's derivative**, the boundary integral of the box-filtered image's derivative (spike 1's exact truth): the
  silhouette's normal speed times its length, summed into the pixel each piece lies in, over ``POINTS`` points of the
  tangent circle; the speed by central differences at a fixed point of the circle (a smooth function of the
  parameters there, so with no sampling noise).

The parameters are the ball's centre (world x, z across the view, y its depth) and the focal length (both axes), the
camera that of ``camera_cases.across``: at ``-distance`` on the world's y, looking along +y, the image's x the world's x
and the image's y the world's -z."""

from __future__ import annotations

import numpy as np

SAMPLES = 4000
POINTS = 400_000


def _local(centre: np.ndarray, distance: float) -> np.ndarray:
    """The ball's centre in the optical frame (x right, y down, z forward)."""
    return np.array([centre[0], -centre[2], centre[1] + distance])


def ellipse(theta: np.ndarray, radius: float, distance: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray,
                                                                                               float]:
    """The silhouette's quadratic form ``P`` and centre in pixels, and its area: inside where
    ``(p - centre)^T P (p - centre) < 1``."""
    c = _local(theta[:3], distance)
    focal = theta[3]
    s2 = c @ c - radius * radius
    m = np.outer(c[:2], c[:2]) - s2 * np.eye(2)
    n = c[2] * c[:2]
    n0 = c[2] * c[2] - s2
    x0 = -np.linalg.solve(m, n)
    k = n0 + n @ x0
    p = -m / k / (focal * focal)
    centre = focal * x0 + np.array([cx, cy])
    return p, centre, float(np.pi / np.sqrt(np.linalg.det(p)))


def areas(theta: np.ndarray, radius: float, distance: float, cx: float, cy: float, shape: tuple[int, int]) -> np.ndarray:
    """The area of each pixel (centres at integer coordinates) inside the silhouette."""
    p, centre, _ = ellipse(theta, radius, distance, cx, cy)
    height, width = shape
    out = np.zeros(shape)
    half_u = np.sqrt(p[1, 1] / np.linalg.det(p))
    columns = range(max(0, int(np.floor(centre[0] - half_u)) - 1), min(width, int(np.ceil(centre[0] + half_u)) + 2))
    offsets = (np.arange(SAMPLES) + 0.5) / SAMPLES - 0.5
    for i in columns:
        u = i + offsets - centre[0]
        # (u, w) P (u, w)^T = 1 in w = v - centre_v: a w^2 + 2 b u w + c u^2 - 1 = 0.
        a, b, c = p[1, 1], p[0, 1], p[0, 0]
        disc = (b * u) ** 2 - a * (c * u * u - 1)
        root = np.sqrt(np.maximum(disc, 0.0))
        low, high = (-b * u - root) / a + centre[1], (-b * u + root) / a + centre[1]
        inside = disc > 0
        rows = range(max(0, int(np.floor(np.min(np.where(inside, low, np.inf)))) - 1),
                     min(height, int(np.ceil(np.max(np.where(inside, high, -np.inf)))) + 2)) if inside.any() else ()
        for j in rows:
            length = np.clip(np.minimum(high, j + 0.5) - np.maximum(low, j - 0.5), 0.0, None)
            out[j, i] = np.mean(np.where(inside, length, 0.0))
    return out


def _circle(theta: np.ndarray, radius: float, distance: float, cx: float, cy: float, phi: np.ndarray) -> np.ndarray:
    """Points of the silhouette in pixels at the tangent circle's angles ``phi`` (a parametrisation smooth in theta)."""
    c = _local(theta[:3], distance)
    focal = theta[3]
    d = np.linalg.norm(c)
    axis = c / d
    first = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    sine = radius / d
    cosine = np.sqrt(1 - sine * sine)
    rays = cosine * axis[:, None] + sine * (np.cos(phi) * first[:, None] + np.sin(phi) * second[:, None])
    return np.stack([focal * rays[0] / rays[2] + cx, focal * rays[1] / rays[2] + cy], axis=-1)


def derivatives(theta: np.ndarray, radius: float, distance: float, cx: float, cy: float,
                shape: tuple[int, int]) -> np.ndarray:
    """``(4, height, width)``: each pixel's covered area's derivative in each parameter, by the boundary integral."""
    phi = (np.arange(POINTS) + 0.5) * 2 * np.pi / POINTS
    points = _circle(theta, radius, distance, cx, cy, phi)
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    length = np.linalg.norm(tangent, axis=-1) / 2
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=-1) / (2 * length[:, None])
    _, centre, _ = ellipse(theta, radius, distance, cx, cy)
    outward = np.sign(np.sum((points - centre) * normal, axis=-1))[:, None]
    normal = normal * outward
    i = np.rint(points[:, 0]).astype(int)
    j = np.rint(points[:, 1]).astype(int)
    keep = (i >= 0) & (i < shape[1]) & (j >= 0) & (j < shape[0])
    out = np.zeros((4, *shape))
    for k in range(4):
        step = np.zeros(4)
        step[k] = 1e-6 * (abs(theta[k]) + (1e-3 if k < 3 else 1.0))
        speed = (_circle(theta + step, radius, distance, cx, cy, phi)
                 - _circle(theta - step, radius, distance, cx, cy, phi)) / (2 * step[k])
        flux = np.sum(speed * normal, axis=-1) * length
        np.add.at(out[k], (j[keep], i[keep]), flux[keep])
    return out


def motion(image: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    """The derivative of the silhouette's equivalent radius ``sqrt(A / pi)`` and of its centroid, in pixels, from its
    coverage image and that image's derivative in one parameter."""
    v, u = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    area, d_area = image.sum(), derivative.sum()
    centroid = np.array([np.sum(u * image), np.sum(v * image)]) / area
    d_centroid = (np.array([np.sum(u * derivative), np.sum(v * derivative)]) - centroid * d_area) / area
    return np.concatenate([[d_area / (2 * np.sqrt(np.pi * area))], d_centroid])
