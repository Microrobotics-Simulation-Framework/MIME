"""The helix: a helical filament, the geometry of slender-body theory and of many microswimmers, as a geometry kind."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax
import numpy as np

from mime.core import jnp
from mime.gradients import GradientStatus
from mime.registry import PHYSICAL
from mime.registry import (
    Centreline,
    CentrelineFrame,
    CentrelineSegment,
    Density,
    Drawing,
    GradientRecord,
    NoPrimitive,
    RegionPrimitive,
    Samples,
    Sampling,
    Solid,
    Tensor,
    register_geometry,
    uniform,
)
from mime.scene._kinds import NO_PRIMITIVE_ADVICE, banded_faces, closed_drawing
from mime.scene._sampling import quasi_uniform
from mime.units import DIMENSIONLESS, Length, Magnitude, Quantity

#: Gauss-Legendre nodes a quarter turn the lateral surface's second moments are integrated with, at build: the
#: integrand is trigonometric in the turn's angle of degree at most three times polynomials of degree two, which this
#: integrates to rounding (``tests/scene/test_helix.py`` holds it to the divergence theorem over the samples).
MOMENT_NODES = 16
#: Points round the filament and through its section the lateral surface's moments are taken at: the section's
#: integrands are trigonometric of degree three and polynomial of degree five, which these integrate exactly.
MOMENT_AROUND, MOMENT_THROUGH = 8, 4
#: Points a turn the reach of the helix's centreline about the centroid's axis is found from (a build-time number).
REACH_POINTS = 4096
#: The turns a helix whose turn count JAX traces is taken to make at most, where the hull query counts the roots the
#: farthest point can be at.
TRACED_TURNS = 64


def _gauss_legendre(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [-1, 1]: exact for polynomials below degree ``2 count``."""
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return nodes, weights


def _around(count: int) -> np.ndarray:
    """``count`` angles evenly round a turn, each in the middle of its interval: the trapezoid rule on a periodic
    integrand, exact for trigonometric polynomials below degree ``count``."""
    return 2.0 * math.pi * (np.arange(count) + 0.5) / count


def _counted(length: float, spacing: float, least: int) -> int:
    return max(least, math.ceil(length / spacing))


def _held(settings: Mapping[str, Magnitude], name: str) -> float:
    value = settings[name]
    if isinstance(value, tuple):
        raise TypeError(f"setting {name!r} is one number, not {value!r}")
    return float(value)


class HelixTurn:
    """The helix's centreline and its Frenet frame in its axis frame (the axis along x through the origin, the axis's
    middle at the origin), at angles ``theta`` round it: ``b`` the pitch over two pi, ``w`` the arc length per radian,
    ``kappa`` and ``tau`` the curvature and the torsion. JAX functions of the parameters where any is a JAX value (a
    node's step, a sample's derivative), and NumPy in float64 where all are numbers (the build's mass properties and
    reach), so that what the build derives does not depend on the precision a process happens to run in."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        values = [parameters[name] for name in ("radius", "pitch", "turns", "filament_radius")]
        self.xp: Any = np if all(isinstance(v, (int, float, np.floating)) for v in values) else jnp
        xp = self.xp
        self.radius, pitch, self.turns, self.filament = (xp.asarray(v) for v in values)
        self.b = pitch / (2.0 * math.pi)
        self.w = xp.sqrt(self.radius * self.radius + self.b * self.b)
        self.kappa = self.radius / (self.w * self.w)
        self.tau = self.b / (self.w * self.w)
        self.sweep = 2.0 * math.pi * self.turns
        self.length = self.sweep * self.w

    def curve(self, theta: Any) -> tuple[Any, Any, Any, Any]:
        """The point, the unit tangent, the principal normal (towards the axis) and the binormal, ``(n, 3)`` each."""
        xp = self.xp
        theta = xp.asarray(theta)
        cos, sin = xp.cos(theta), xp.sin(theta)
        zero = xp.zeros_like(theta)
        point = xp.stack([self.b * (theta - 0.5 * self.sweep), self.radius * cos, self.radius * sin], axis=-1)
        tangent = xp.stack([xp.broadcast_to(self.b, theta.shape), -self.radius * sin, self.radius * cos],
                           axis=-1) / self.w
        normal = xp.stack([zero, -cos, -sin], axis=-1)
        binormal = xp.stack([xp.broadcast_to(self.radius, theta.shape), self.b * sin, -self.b * cos],
                            axis=-1) / self.w
        return point, tangent, normal, binormal

    def centroid(self) -> Any:
        """The solid's centroid in the axis frame: the filament's first moment, ``pi r^2`` times the centreline's, less
        the curvature's share ``pi r^4 kappa / 4`` of the principal normal's integral, and each cap's (a hemisphere's
        volume at ``3 r / 8`` out along its end's tangent), over the volume."""
        xp = self.xp
        r, sweep = self.filament, self.sweep
        across = xp.stack([xp.zeros_like(sweep), xp.sin(sweep), 1.0 - xp.cos(sweep)])
        lateral = self.w * across * (math.pi * r * r * self.radius + math.pi * r ** 4 * self.kappa / 4.0)
        ends = self.curve(xp.stack([xp.zeros_like(sweep), sweep]))
        caps = (2.0 / 3.0) * math.pi * r ** 3 * (ends[0][0] + ends[0][1] - 3.0 * r / 8.0 * (ends[1][0] - ends[1][1]))
        volume = math.pi * r * r * self.length + 4.0 / 3.0 * math.pi * r ** 3
        return (lateral + caps) / volume


def _length(parameters: Mapping[str, Any]) -> Any:
    return HelixTurn(parameters).length


def _frame(parameters: Mapping[str, Any], s: Any) -> CentrelineFrame:
    """The helix by arc length, its normal the rotation-minimising frame's: the principal normal turned about the
    tangent by ``-tau s``, so that it turns only as the tangent does."""
    turn = HelixTurn(parameters)
    at = jnp.asarray(s)
    point, tangent, normal, binormal = turn.curve(at / turn.w)
    twist = -turn.tau * at
    transported = jnp.cos(twist)[:, None] * normal + jnp.sin(twist)[:, None] * binormal
    return CentrelineFrame(point - turn.centroid(), tangent, transported)


def _radius(parameters: Mapping[str, Any], s: Any) -> Any:
    return jnp.broadcast_to(jnp.asarray(parameters["filament_radius"]), jnp.asarray(s).shape)


def _filament(parameters: Mapping[str, Any]) -> Any:
    return jnp.asarray(parameters["filament_radius"])


def _wavelength(parameters: Mapping[str, Any]) -> Any:
    return 2.0 * math.pi * HelixTurn(parameters).w


#: The helix's centreline: one segment, its functions module-level, so the same for every helix (its values are its
#: parameters'), and a consumer compiles them once.
HELIX_CENTRELINE = Centreline((CentrelineSegment(length=_length, frame=_frame, radius=_radius,
                                                 filament_radius=_filament, wavelength=_wavelength),))


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Helix(Solid):
    """A helical filament: a filament of ``filament_radius`` round a helix of ``radius`` (its centreline's distance from
    its axis), ``pitch`` (the axial advance a turn, signed: above zero right-handed) and ``turns``, its axis along its
    frame's x as the screw's, closed by hemispherical caps (the points within ``filament_radius`` of the centreline's
    arc). Its frame's origin is its centroid, which lies off the axis where the turns are not whole (and by the caps
    always): the centreline runs from ``x = -pitch turns / 2`` to ``+pitch turns / 2`` along the axis, less the
    centroid. Its patches are the ``lateral`` tube about the centreline and the two hemispherical ``caps``, both the
    boundary of its region ``solid``, which no registered shape evaluates (the conformance harness holds a level set of
    it).

    **Its centreline** (``mime.registry.Centreline``) is one segment, the helix by arc length, its frame's normal
    the rotation-minimising one (the principal normal turned about the tangent by minus the torsion's integral, ``-tau
    s``), its radius the filament's, its wavelength the arc length of a turn; every value a JAX function of the four
    settings, so a study's derivative in them flows through it and through the samples.

    Its volume is the tube's, ``pi r^2 L`` (``L`` the centreline's length, Pappus's theorem for a tube about a curve),
    and the caps', ``4 pi r^3 / 3``; its inertia about the centroid the lateral tube's second moments integrated along
    the centreline at build (``MOMENT_NODES`` a quarter turn, to rounding) and the hemispheres' closed forms.

    Its scope: a circular section, constant radius and pitch, the filament neither folding on its inner side
    (``filament_radius`` below the centreline's radius of curvature) nor meeting its neighbouring turn (the turns'
    separation across the filament, ``|pitch| radius / w`` with ``w`` the arc length per radian, above twice the
    filament's radius), both refused when a value is declared. A screw-shaped body (a thread round a core) is
    ``Screw``."""

    name = "helix"
    version = "0.1.0"
    setting_roles = dict.fromkeys(("radius", "pitch", "turns", "filament_radius"), PHYSICAL)
    differentiability = GradientStatus.EXACT
    gradient = GradientRecord()
    references = ("Bishop1975",)
    builds = {"": ()}
    regions = ("solid",)
    patches = {"lateral": "solid", "caps": "solid"}
    radius: Annotated[Quantity, Length]
    pitch: Annotated[Quantity, Length]
    turns: Annotated[Quantity, DIMENSIONLESS]
    filament_radius: Annotated[Quantity, Length]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        for name, what in (("radius", "a length"), ("filament_radius", "a length"), ("turns", "a number")):
            value = settings[name]
            if not (isinstance(value, float) and math.isfinite(value) and value > 0.0):
                found.append(f"{name} is {value!r}, not {what} above zero")
        pitch = settings["pitch"]
        if not (isinstance(pitch, float) and math.isfinite(pitch) and pitch != 0.0):
            found.append(f"pitch is {pitch!r} m, not a length other than zero (its sign is the handedness)")
        if found:
            return found
        radius, pitch, filament = _held(settings, "radius"), _held(settings, "pitch"), _held(settings,
                                                                                               "filament_radius")
        b = pitch / (2.0 * math.pi)
        w = math.hypot(radius, b)
        if filament * radius / (w * w) >= 1.0:
            found.append(f"filament_radius {filament!r} m reaches past the centreline's radius of curvature "
                         f"{w * w / radius!r} m, so the filament folds on its inner side")
        if abs(pitch) * radius / w <= 2.0 * filament:
            found.append(f"the turns lie {abs(pitch) * radius / w!r} m apart across the filament, and the filament "
                         f"is {2.0 * filament!r} m thick, so neighbouring turns meet: a larger pitch or a thinner "
                         f"filament")
        return found

    def _turn(self, settings: Mapping[str, Magnitude]) -> HelixTurn:
        return HelixTurn({name: _held(settings, name) for name in ("radius", "pitch", "turns", "filament_radius")})

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        filament = _held(settings, "filament_radius")
        return math.pi * filament * filament * float(self._turn(settings).length) + 4.0 / 3.0 * math.pi * filament ** 3

    def _moments(self, settings: Mapping[str, Magnitude]) -> np.ndarray:
        """The second moments about the centroid, ``int x x^T dV`` (m^5), in float64 at build."""
        turn = self._turn(settings)
        r, sweep, w, kappa = (float(v) for v in (turn.filament, turn.sweep, turn.w, turn.kappa))
        centroid = np.asarray(turn.centroid(), dtype=np.float64)
        panels = max(1, math.ceil(4.0 * sweep / (2.0 * math.pi)))
        nodes, weights = _gauss_legendre(MOMENT_NODES)
        edges = np.linspace(0.0, sweep, panels + 1)
        theta = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * (edges[1:, None] - edges[:-1, None])
                 * nodes[None, :]).ravel()
        dtheta = (0.5 * (edges[1:, None] - edges[:-1, None]) * weights[None, :]).ravel()
        point, _, normal, binormal = (np.asarray(v, dtype=np.float64) for v in turn.curve(theta))
        rho_nodes, rho_weights = _gauss_legendre(MOMENT_THROUGH)
        rho, drho = r * (rho_nodes + 1.0) / 2.0, r * rho_weights / 2.0
        phi = _around(MOMENT_AROUND)
        dphi = 2.0 * math.pi / MOMENT_AROUND
        moments = np.zeros((3, 3))
        for k in range(len(theta)):
            across = np.cos(phi)[:, None] * normal[k] + np.sin(phi)[:, None] * binormal[k]  # (phi, 3)
            x = point[k][None, None, :] + rho[:, None, None] * across[None, :, :] - centroid  # (rho, phi, 3)
            element = (1.0 - kappa * rho[:, None] * np.cos(phi)[None, :]) * rho[:, None] * drho[:, None] * dphi
            moments += np.einsum("ap,api,apj->ij", element, x, x) * w * dtheta[k]
        ends = turn.curve(np.asarray([0.0, sweep]))
        hemisphere = 2.0 / 3.0 * math.pi * r ** 3
        for end, sign in ((0, -1.0), (1, 1.0)):
            e = np.asarray(ends[0][end], dtype=np.float64) - centroid
            out = sign * np.asarray(ends[1][end], dtype=np.float64)
            first = hemisphere * 3.0 * r / 8.0 * out
            moments += (hemisphere * np.outer(e, e) + np.outer(e, first) + np.outer(first, e)
                        + 2.0 * math.pi * r ** 5 / 15.0 * np.eye(3))
        return moments

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        moments = self._moments(settings) / self.volume(settings)
        inertia = np.trace(moments) * np.eye(3) - moments
        rows = [tuple(float(v) for v in row) for row in 0.5 * (inertia + inertia.T)]
        return (rows[0][0], rows[0][1], rows[0][2]), (rows[1][0], rows[1][1], rows[1][2]), \
            (rows[2][0], rows[2][1], rows[2][2])

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "solid":
            raise KeyError(region)
        return {name: _held(settings, name) for name in ("radius", "pitch", "turns", "filament_radius")}

    @override
    def sample_counts(self, settings: Mapping[str, Magnitude], spacing: float, patch: str) -> tuple[int, ...]:
        filament = _held(settings, "filament_radius")
        if patch == "lateral":
            return _counted(2.0 * math.pi * filament, spacing, 3), _counted(float(self._turn(settings).length),
                                                                            spacing, 1)
        if patch == "caps":
            return _counted(math.pi * filament / 2.0, spacing, 2), _counted(2.0 * math.pi * filament, spacing, 3)
        raise KeyError(patch)

    @override
    def samples(self, parameters: Mapping[str, Any], patch: str, counts: tuple[int, ...]) -> Samples:
        """The lateral tube by the trapezoid rule round the filament and Gauss-Legendre nodes along its centreline,
        weighted by its surface element ``r (1 - kappa r cos phi)``; each cap by Gauss-Legendre nodes in the cosine of
        the angle from its pole and the trapezoid rule round it; all less the centroid, from the parameters as JAX
        values."""
        turn = HelixTurn(parameters)
        centroid = turn.centroid()
        r = turn.filament
        if patch == "lateral":
            around, along = counts
            phi = jnp.asarray(_around(around))
            node, weight = _gauss_legendre(along)
            theta = turn.sweep * (jnp.asarray(node) + 1.0) / 2.0
            point, _, normal, binormal = turn.curve(theta)
            across = (jnp.cos(phi)[:, None, None] * normal[None, :, :]
                      + jnp.sin(phi)[:, None, None] * binormal[None, :, :]).reshape(-1, 3)
            points = (jnp.broadcast_to(point[None, :, :], (around, along, 3)).reshape(-1, 3) + r * across
                      - centroid)
            element = (r * (1.0 - turn.kappa * r * jnp.cos(phi))[:, None] * (2.0 * math.pi / around)
                       * (turn.length / 2.0) * jnp.asarray(weight)[None, :])
            return Samples(points, across, element.reshape(-1))
        if patch == "caps":
            polar, around = counts
            node, weight = _gauss_legendre(polar)
            cosine, weight = (node + 1.0) / 2.0, weight / 2.0
            sine = np.sqrt(1.0 - cosine ** 2)
            angle = _around(around)
            ends = turn.curve(jnp.stack([jnp.zeros_like(turn.sweep), turn.sweep]))
            found: list[tuple[Any, Any, Any]] = []
            for end, sign in ((0, -1.0), (1, 1.0)):
                out = sign * ends[1][end]
                normals = (jnp.asarray(np.repeat(cosine, around))[:, None] * out
                           + jnp.asarray(np.outer(sine, np.cos(angle)).ravel())[:, None] * ends[2][end]
                           + jnp.asarray(np.outer(sine, np.sin(angle)).ravel())[:, None] * ends[3][end])
                areas = jnp.asarray(np.outer(weight, np.full(around, 2.0 * math.pi / around)).ravel())
                found.append((ends[0][end] + r * normals - centroid, normals, r * r * areas))
            return Samples(*(jnp.concatenate([f[i] for f in found]) for i in range(3)))
        raise KeyError(patch)

    @override
    def area(self, settings: Mapping[str, Magnitude], patch: str) -> float:
        filament = _held(settings, "filament_radius")
        if patch == "lateral":
            return 2.0 * math.pi * filament * float(self._turn(settings).length)
        if patch == "caps":
            return 4.0 * math.pi * filament * filament
        raise KeyError(patch)

    @override
    def family_parameters(self, settings: Mapping[str, Magnitude]) -> dict[str, float | int]:
        return {name: _held(settings, name) for name in ("radius", "pitch", "turns", "filament_radius")}

    @override
    def sampling(self, settings: Mapping[str, Magnitude], spacing: float, patches: Sequence[str],
                 density: Density = uniform) -> Sampling:
        return quasi_uniform(self, settings, spacing, patches, density)

    def _centreline_points(self, settings: Mapping[str, Magnitude]) -> np.ndarray:
        turn = self._turn(settings)
        count = max(2, math.ceil(REACH_POINTS * float(turn.turns)))
        theta = np.linspace(0.0, float(turn.sweep), count)
        return np.asarray(turn.curve(theta)[0], dtype=np.float64) - np.asarray(turn.centroid(), dtype=np.float64)

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        """The centreline's largest distance from the axis through the centroid, found on ``REACH_POINTS`` a turn,
        and the filament's radius."""
        points = self._centreline_points(settings)
        return float(np.max(np.hypot(points[:, 1], points[:, 2]))) + _held(settings, "filament_radius")

    @override
    def reach(self, parameters: Mapping[str, Any], direction: Any) -> Any:
        """The hull query, exact: the body is the union of balls of the filament's radius centred on the centreline, so
        its reach is the centreline's farthest point along ``direction`` plus the radius times the direction's length.
        The farthest point along ``d`` of ``(b (t - sweep/2), R cos t, R sin t)`` less the centroid is at a stationary
        point of ``a t + A cos(t - phi)``, ``a = b d_x`` and ``A = R |d_yz|``, at ``sin(t - phi) = a / A`` (two roots a
        turn, found for every turn the helix makes), or at an end; the gradient is that point. A turn count traced by
        JAX is taken to be below ``TRACED_TURNS``."""
        names = ("radius", "pitch", "turns", "filament_radius")
        turn, d = HelixTurn({name: jnp.asarray(parameters[name]) for name in names}), jnp.asarray(direction)
        try:
            turns = int(math.ceil(float(turn.turns)))
        except Exception:  # noqa: BLE001 - a traced turn count has no value to read
            turns = TRACED_TURNS
        phi = jnp.arctan2(d[2], d[1])
        along, across = turn.b * d[0], turn.radius * jnp.hypot(d[1], d[2])
        base = jnp.arcsin(jnp.clip(along / jnp.where(across > 0.0, across, 1.0), -1.0, 1.0))
        laps = 2.0 * math.pi * jnp.arange(-1, turns + 2)
        roots = jnp.concatenate([phi + base + laps, phi + math.pi - base + laps])
        # The roots are held out of the derivative: at a stationary point the value does not move with them, and the
        # derivative is the point's; an end carries its own.
        inside = jax.lax.stop_gradient(jnp.clip(roots, 0.0, jax.lax.stop_gradient(turn.sweep)))
        candidates = jnp.concatenate([inside, jnp.stack([jnp.zeros_like(turn.sweep), turn.sweep])])
        points = turn.curve(candidates)[0] - turn.centroid()
        return jnp.max(points @ d) + turn.filament * jnp.sqrt(jnp.sum(d * d))

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        points = self._centreline_points(settings)
        return float(np.max(np.linalg.norm(points, axis=1))) + _held(settings, "filament_radius")

    @override
    def centreline(self, settings: Mapping[str, Magnitude]) -> Centreline:
        return HELIX_CENTRELINE

    @override
    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        """Sections along the centreline, ``segments`` points round each, as many as keep each a ``segments``-th of a
        turn from the next, closed at each end by rings over its hemisphere and a fan from its pole: every vertex on
        the surface."""
        turn = self._turn(settings)
        r, sweep = float(turn.filament), float(turn.sweep)
        centroid = np.asarray(turn.centroid(), dtype=np.float64)
        spans = max(1, math.ceil(segments * sweep / (2.0 * math.pi)))
        polar = max(1, segments // 4)
        phi = 2.0 * math.pi * np.arange(segments) / segments
        theta = np.linspace(0.0, sweep, spans + 1)
        point, tangent, normal, binormal = (np.asarray(v, dtype=np.float64) for v in turn.curve(theta))
        rings: list[np.ndarray] = []
        start_cap = [(np.cos(a), np.sin(a)) for a in (np.pi / 2.0 * (np.arange(1, polar) / polar))]
        for cos_a, sin_a in start_cap:  # from near the start's pole outward
            across = np.cos(phi)[:, None] * normal[0] + np.sin(phi)[:, None] * binormal[0]
            rings.append(point[0] + r * (-cos_a * tangent[0] + sin_a * across))
        for k in range(spans + 1):
            across = np.cos(phi)[:, None] * normal[k] + np.sin(phi)[:, None] * binormal[k]
            rings.append(point[k] + r * across)
        for cos_a, sin_a in reversed(start_cap):  # towards the end's pole
            across = np.cos(phi)[:, None] * normal[-1] + np.sin(phi)[:, None] * binormal[-1]
            rings.append(point[-1] + r * (cos_a * tangent[-1] + sin_a * across))
        sections = np.concatenate(rings)
        poles = np.stack([point[0] - r * tangent[0], point[-1] + r * tangent[-1]])
        points = np.concatenate([sections, poles]) - centroid
        low, high = len(sections), len(sections) + 1
        faces = [face[::-1] for face in banded_faces(len(rings), segments, low, high, start=0)]
        return closed_drawing(points, faces, f"the filament as {len(rings)} rings of {segments} points along its "
                                             f"centreline and over its two hemispherical caps, closed by a fan at "
                                             f"each pole")

    @override
    def region_primitive(self, region: str, settings: Mapping[str, Magnitude]) -> RegionPrimitive | NoPrimitive:
        if region != "solid":
            raise KeyError(region)
        return NoPrimitive(f"a helical filament is no primitive. {NO_PRIMITIVE_ADVICE}")


__all__ = ["HELIX_CENTRELINE", "MOMENT_NODES", "Helix", "HelixTurn"]
