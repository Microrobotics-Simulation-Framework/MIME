"""The shape conformance suite: each registered shape's derivatives equal the finite difference on the point's side of
every switch it declares, at points off, near and on each declared set, under a random rotation and placement.

**Cases.** Each shape has a :class:`ShapeCase` (``SHAPE_CASES``, by its registered name): a strategy for its
parameters in its own frame at physical scales, its size, how it is turned and moved, the parameters a study may move
(its scalars, and its centre, axis or orientation, which a pose or a placement moves), and for each switch it declares
how a point is placed a signed distance from the set (the indicator's value there) and the band's width. A shape MIME registers with no case fails the walk (:func:`uncased_shapes`; a lab's
shapes are its own to case).

**A probe** (:func:`shape_probes`) picks a set and a class: ``off`` (a twentieth of the shape's size or more from the
set), ``near`` (4 to 10^4 of the set's bands for the dtype drawn, never more than a tenth of the size) or ``on`` (on the
set, inside its band); a direction for the point and a parameter to move with it, together one curve (a centre moves
along a direction, an axis turns across itself and an orientation about a direction). Autodiff's
derivative along the curve (a JVP) is taken in the dtype drawn; the reference is the central difference along it,
extrapolated (Richardson), always in float64. Its step is at most a sixteenth of the distance to the nearest switch at
the point, which every owner's indicator gives through a collector (half the gap for the screw's seam, whose
indicator is the gap between its two nearest roots; for its focus, ``R (delta / R)^(3/2)``, how far across the normal
the nearest point stays analytic a distance ``delta`` from the caustic's cusp), so no difference straddles a switch. On a set no difference is
taken: both one-sided slopes are reported, not compared, and autodiff must be finite there; so too where a point
placed off one set lies on another part of a set (the screw's seam runs out along every symmetry line of its
cross-section, not only along its axis).

**Dtypes** [#168]. A float32 probe is placed in float32's bands, so it lies outside both dtypes' bands, and its
autodiff (computed in float32 throughout) is compared with the float64 difference: the only float32 against float64
comparison. Each dtype has its own tolerance and planted tangent, each set from that dtype's measured agreement. Near a
set the float32 comparison resolves its floor, not its tolerance: autodiff's rounding floor, ``k eps32 S (|p| + size)
/ d``, is about 1e-3 of the scale a tenth of the size from a set and 1e-1 a thousandth from it, so a float32 probe
near a set checks finiteness and errors of that size, and the float32 control is placed where its floor lies below
the planted tangent.

**The criterion**: ``|AD - FD| <= SHAPE_TOLERANCE[dtype] S + floor``, ``S`` the derivative's scale from its inputs:
the largest entry of the capability's Jacobian in the point (the derivative along any direction), or the difference's
size, or the capability's own (1 for the distance and the nearest point, 1 / size for the normal and the tangent),
whichever is largest. A derivative along one direction can be a small remainder of a large Jacobian (near a cone,
along its normal), and is judged on the Jacobian's scale, not its own. The floor is rounding only: the difference's
own, the larger of a model, ``k eps64 (S (|p| + size) + |f|) / h`` (``k = DIFFERENCE_FLOOR_ULPS``), and its noise
read directly, four times the spread of the same difference at base points a few units in the last place apart; and
autodiff's own, in the dtype it ran in, ``k eps S (|p| + size) / d``, ``d`` the distance to the nearest switch (a
point's rounding turns a direction across a nearby set by about ``eps |p| / d``). No floor comes from two
differences' difference: ``2 |coarse - fine|`` accepts any tangent across a switch.

**The control**: :func:`check_shape` with ``planted`` scales autodiff's tangent by ``1 + planted`` through a
``custom_jvp``; :data:`PLANTED_TANGENT`, by dtype, lies above that dtype's tolerance, which lies above its measured
agreement, each by the margin its test records."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Any, cast

import jax
import numpy as np
from hypothesis import strategies as st

from mime.core import jnp, nearest_point, quaternion, shape_named, shape_names, switch
from mime.core._geometry import AXIS_ULPS

type Parameters = dict[str, Any]
type Placement = Callable[[Parameters, float, np.ndarray], np.ndarray]

#: The relative agreement a probe must reach, by dtype, beyond the rounding floor: set from the agreement the suite
#: measured over its stress runs (tests/saved/test_shape_suite.py records them), with the margin recorded there.
SHAPE_TOLERANCE: Mapping[str, float] = MappingProxyType({"float64": 1e-6, "float32": 1e-4})
#: The planted wrong tangent's relative error, by dtype: above that dtype's tolerance, so that a probe placed where the
#: floor lies below it must reject it.
PLANTED_TANGENT: Mapping[str, float] = MappingProxyType({"float64": 1e-4, "float32": 1e-2})
#: The rounding floor's modelled part, in units in the last place.
DIFFERENCE_FLOOR_ULPS = 128.0
#: The floor's measured part: the spread of the difference over base points this many units in the last place of the
#: point's scale apart (far below the step, so that the derivative does not move between them), times
#: ``NOISE_FACTOR``: the difference's own rounding noise, read rather than modelled.
ROUNDING_SHIFT_ULPS = 64.0
NOISE_FACTOR = 4.0
PROBE_CLASSES = ("off", "near", "on")
_jit: Any = cast(Any, jax).jit
_jvp: Any = cast(Any, jax).jvp
_custom_jvp: Any = cast(Any, jax).custom_jvp
_tree_map: Any = cast(Any, jax).tree_util.tree_map
_jacfwd: Any = cast(Any, jax).jacfwd


@dataclass(frozen=True)
class SwitchSet:
    """A declared switch's set: ``place(parameters, offset, draw)`` gives a point in the shape's own frame whose
    indicator is ``offset`` (its sign the side), from three uniform numbers in ``draw``; ``band(parameters, dtype)``
    is the set's width for that dtype."""

    owner: str
    place: Placement
    band: Callable[[Parameters, Any], float]


@dataclass(frozen=True)
class ShapeCase:
    """A registered shape's case: its ``parameters`` in its own frame, its ``size``, ``turned`` (its parameters
    turned by a unit quaternion and moved by a translation), the parameters a study may move (``moved``, with the
    dimensionless scalars in ``plain``: a probe's curve moves a length by up to its own distance along it, a
    dimensionless parameter by up to that over the size, a ``centre`` by up to its distance along a direction, and an
    ``axis`` or an ``orientation`` turns by up to that over the size), its sets, and its capabilities. ``unplaced``
    names each declared owner no point can be placed at, with why."""

    shape: str
    parameters: st.SearchStrategy[Parameters]
    size: Callable[[Parameters], float]
    turned: Callable[[Parameters, np.ndarray, np.ndarray], Parameters]
    sets: tuple[SwitchSet, ...]
    moved: tuple[str, ...]
    plain: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ("distance", "normal", "nearest")
    unplaced: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


SHAPE_CASES: dict[str, ShapeCase] = {}


def register_shape_case(case: ShapeCase) -> ShapeCase:
    SHAPE_CASES[case.shape] = case
    return case


def uncased_shapes(package: str = "mime") -> list[str]:
    """Each shape registered from ``package`` with no case, and each owner such a shape declares that its case neither
    places nor says why it cannot. A shape of another package (a lab's, a test's) is its package's to case."""
    from mime.core import shape_switches

    found: list[str] = []
    for name in shape_names():
        if type(shape_named(name)).__module__.split(".")[0] != package:
            continue
        case = SHAPE_CASES.get(name)
        if case is None:
            found.append(f"shape {name}: no case in the shape conformance suite")
            continue
        placed = {s.owner for s in case.sets} | set(case.unplaced)
        found += [f"shape {name}: its switch {d.owner} has no set in its case" for d in shape_switches(name)
                  if d.owner not in placed]
    return found


@dataclass(frozen=True)
class ShapeProbe:
    """One drawn point of a shape's case: its world parameters, the point, the curve's direction in the point and in
    the parameters, the set probed, its class, the signed offset placed, the dtype, and the size."""

    shape: str
    parameters: Parameters
    point: np.ndarray
    direction: np.ndarray
    moved: Mapping[str, Any]
    owner: str
    kind: str
    offset: float
    dtype: str
    size: float


def _turn_from(draw: np.ndarray) -> np.ndarray:
    """A unit quaternion from three uniform numbers, evenly over rotations (Shoemake's method): a rotation every draw
    gives, never a zero vector normalised."""
    first, second = math.sqrt(1.0 - draw[0]), math.sqrt(draw[0])
    a, b = 2.0 * math.pi * draw[1], 2.0 * math.pi * draw[2]
    return np.array([second * math.cos(b), first * math.sin(a), first * math.cos(a), second * math.sin(b)])


def _rotation(q: np.ndarray) -> np.ndarray:
    return np.asarray(quaternion.rotation_matrix(jnp.asarray(q, dtype=jnp.float64)))


@st.composite
def shape_probes(draw: Any, case: ShapeCase, kinds: Sequence[str] = PROBE_CLASSES,
                 dtypes: Sequence[str] = ("float64", "float32")) -> ShapeProbe:
    """A probe of ``case``: its parameters, a random rotation and placement, a set, a class, a direction and a
    parameter moved with the point."""
    own = draw(case.parameters)
    size = case.size(own)
    turn = _turn_from(np.array(draw(st.tuples(*(st.floats(0.0, 1.0),) * 3))))
    shift = np.array(draw(st.tuples(*(st.floats(-3.0, 3.0),) * 3))) * size
    chosen = draw(st.sampled_from(case.sets))
    kind = draw(st.sampled_from(tuple(kinds)))
    dtype = draw(st.sampled_from(tuple(dtypes)))
    side = draw(st.sampled_from((-1.0, 1.0)))
    band = chosen.band(own, jnp.dtype(dtype))
    if kind == "on":
        offset = 0.0
    elif kind == "near":
        offset = side * min(10.0 ** draw(st.floats(math.log10(4.0), 4.0)) * band, 0.1 * size)
    else:
        offset = side * draw(st.floats(0.05, 1.0)) * size
    where = np.array(draw(st.tuples(*(st.floats(0.0, 1.0),) * 3)))
    rotation = _rotation(turn)
    point = rotation @ chosen.place(own, offset, where) + shift
    direction = _direction_from(np.array(draw(st.tuples(*(st.floats(0.0, 1.0),) * 2))))
    name = draw(st.sampled_from(case.moved))
    rate = draw(st.floats(-1.0, 1.0))
    turning = np.array(draw(st.tuples(*(st.floats(0.0, 1.0),) * 2)))
    world = case.turned(own, turn, shift)
    moved = {name: moved_rate(case, world, name, rate, turning, size)}
    return ShapeProbe(case.shape, world, point, direction, moved, chosen.owner, kind, offset, dtype, size)


def moved_rate(case: ShapeCase, parameters: Parameters, name: str, rate: float, draw: np.ndarray,
               size: float) -> Any:
    """How fast ``name`` moves along a probe's curve, per metre the point moves: a length at ``rate``, a dimensionless
    scalar at ``rate / size``, a ``centre`` at ``rate`` along a direction drawn from two uniform numbers, an ``axis``
    turned across itself, and an ``orientation`` turned about a drawn direction, each at ``rate / size`` radians."""
    if name == "centre":
        return rate * _direction_from(draw)
    if name == "axis":
        axis = np.asarray(parameters["axis"], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        first = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
        first /= np.linalg.norm(first)
        second = np.cross(axis, first)
        angle = 2.0 * math.pi * draw[0]
        return rate / size * (math.cos(angle) * first + math.sin(angle) * second)
    if name == "orientation":
        q = np.asarray(parameters["orientation"], dtype=np.float64)
        omega = rate / size * _direction_from(draw)
        return 0.5 * np.concatenate([[-np.dot(q[1:], omega)], q[0] * omega + np.cross(q[1:], omega)])
    return rate / size if name in case.plain else rate


def capability(shape: Any, name: str) -> Callable[[Parameters, Any], Any]:
    """The run-time capability ``name`` of ``shape`` as a function of its parameters and a point."""
    if name == "distance":
        return lambda parameters, point: shape.signed_distance(parameters, point)
    if name == "normal":
        return lambda parameters, point: shape.outward_normal(parameters, point)
    if name == "nearest":
        return lambda parameters, point: nearest_point(shape, parameters, point)
    if name == "tangent":
        return lambda parameters, point: shape.tangent(parameters, point)
    raise KeyError(name)


def _split(parameters: Parameters, dtype: Any) -> tuple[dict[str, Any], tuple[tuple[str, int], ...]]:
    """The parameters as arrays of ``dtype`` and the integers, which are structural, as a hashable tuple."""
    numeric = {key: jnp.asarray(value, dtype=dtype) for key, value in parameters.items()
               if not (isinstance(value, int) and not isinstance(value, bool))}
    whole = tuple(sorted((key, value) for key, value in parameters.items()
                         if isinstance(value, int) and not isinstance(value, bool)))
    return numeric, whole


def _planted(f: Callable[..., Any], scale: float) -> Callable[..., Any]:
    """``f`` with its tangent scaled by ``1 + scale``: a wrong derivative, for the control."""
    wrapped = _custom_jvp(f)

    def rule(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, Any]:
        value, tangent = _jvp(f, primals, tangents)

        def scaled(held: Any) -> Any:
            return (1.0 + scale) * held

        return value, _tree_map(scaled, tangent)

    wrapped.defjvp(rule)
    return wrapped


@partial(_jit, static_argnames=("shape", "name", "whole", "planted"))
def _derivative(shape: str, name: str, whole: tuple[tuple[str, int], ...], planted: float, numeric: dict[str, Any],
                point: Any, numeric_tangent: dict[str, Any], point_tangent: Any) -> tuple[Any, Any]:
    f = capability(shape_named(shape), name)

    def along(values: dict[str, Any], at: Any) -> Any:
        return f({**values, **dict(whole)}, at)

    held = _planted(along, planted) if planted else along
    return _jvp(held, (numeric, point), (numeric_tangent, point_tangent))


@partial(_jit, static_argnames=("shape", "name", "whole"))
def _jacobian(shape: str, name: str, whole: tuple[tuple[str, int], ...], numeric: dict[str, Any], point: Any) -> Any:
    """The capability's Jacobian in the point: the derivative's scale along any direction, which the inputs give."""
    f = capability(shape_named(shape), name)

    def at(held: Any) -> Any:
        return f({**numeric, **dict(whole)}, held)

    return _jacfwd(at)(point)


@partial(_jit, static_argnames=("shape", "name", "whole"))
def _value(shape: str, name: str, whole: tuple[tuple[str, int], ...], numeric: dict[str, Any], point: Any) -> Any:
    return capability(shape_named(shape), name)({**numeric, **dict(whole)}, point)


#: The owners each traced evaluation of ``_indicators`` collected, in order, by its static arguments.
_OWNERS: dict[tuple[str, tuple[tuple[str, int], ...]], list[str]] = {}


@partial(_jit, static_argnames=("shape", "whole"))
def _indicators(shape: str, whole: tuple[tuple[str, int], ...], numeric: dict[str, Any], point: Any) -> list[Any]:
    """Each switch's indicator nearest zero, as the shape's distance and normal evaluate them at ``point``: collected
    while the evaluation is traced, so its owners are recorded once per trace."""
    held, evaluated = {**numeric, **dict(whole)}, shape_named(shape)
    with switch.collecting() as rows:
        evaluated.signed_distance(held, point)
        evaluated.outward_normal(held, point)
    _OWNERS[(shape, whole)] = [row.owner for row in rows]
    return [jnp.min(jnp.abs(jnp.ravel(row.indicator))) for row in rows]


def switch_distances(probe: ShapeProbe) -> dict[str, float]:
    """How far the probe's point is from each switch the shape evaluates there, by owner: half the magnitude of its
    indicator (each MIME shape's indicator is a distance, or the screw seam's gap, at most twice its distance, or the
    screw focus's ``R (delta / R)^(3/2)``, the reach of analyticity about the caustic's cusp)."""
    numeric, whole = _split(probe.parameters, jnp.float64)
    indicators = _indicators(probe.shape, whole, numeric, jnp.asarray(probe.point, dtype=jnp.float64))
    found: dict[str, float] = {}
    for owner, indicator in zip(_OWNERS[(probe.shape, whole)], indicators, strict=True):
        found[owner] = min(found.get(owner, math.inf), float(indicator) / 2.0)
    return found


@dataclass(frozen=True)
class ShapeCheck:
    """One probe of one capability: autodiff's derivative, the difference's (``None`` on a set), the scale, the
    floor, the step, the distance to the nearest switch, the one-sided slopes on a set, and the problems found."""

    probe: ShapeProbe
    name: str
    autodiff: np.ndarray
    difference: np.ndarray | None
    scale: float
    floor: float
    step: float
    nearest_switch: float
    one_sided: tuple[np.ndarray, np.ndarray] | None
    problems: tuple[str, ...]

    @property
    def agreement(self) -> float:
        """The relative disagreement, ``|AD - FD| / S``: what the tolerance is measured on."""
        if self.difference is None:
            return 0.0
        return float(np.max(np.abs(self.autodiff - self.difference))) / self.scale

    def line(self) -> str:
        p = self.probe
        return (f"{p.shape} {self.name} {p.dtype} {p.kind} {p.owner} offset {p.offset:.3g} m: agreement "
                f"{self.agreement:.3g}, floor {self.floor / self.scale:.3g}, step {self.step:.3g} m, nearest switch "
                f"{self.nearest_switch:.3g} m")


def _natural(name: str, size: float) -> float:
    return 1.0 if name in ("distance", "nearest") else 1.0 / size


def check_shape(probe: ShapeProbe, name: str, *, planted: float = 0.0) -> ShapeCheck:
    """``probe`` of capability ``name``: autodiff in the probe's dtype against the float64 difference on the point's
    side, or, on a set, autodiff's finiteness and both one-sided slopes."""
    dtype = jnp.dtype(probe.dtype)
    problems: list[str] = []
    numeric, whole = _split(probe.parameters, dtype)
    tangent = {key: jnp.zeros_like(value) for key, value in numeric.items()}
    tangent.update({key: jnp.asarray(rate, dtype=dtype) for key, rate in probe.moved.items()})
    value, derivative = _derivative(probe.shape, name, whole, planted, numeric, jnp.asarray(probe.point, dtype=dtype),
                                    tangent, jnp.asarray(probe.direction, dtype=dtype))
    autodiff = np.asarray(derivative, dtype=np.float64).ravel()
    if np.asarray(value).dtype != dtype or np.asarray(derivative).dtype != dtype:
        problems.append(f"computed in {np.asarray(value).dtype}, not the {dtype} it was given")
    if not np.all(np.isfinite(autodiff)):
        problems.append(f"autodiff's derivative is not finite: {autodiff}")
    distances = switch_distances(probe)
    nearest = min(distances.values(), default=math.inf)
    numeric64, _ = _split(probe.parameters, jnp.float64)
    reach = float(np.linalg.norm(probe.point)) + probe.size

    def at(t: float) -> np.ndarray:
        moved = {key: value + t * probe.moved.get(key, 0.0) for key, value in numeric64.items()}
        return np.asarray(_value(probe.shape, name, whole, moved, jnp.asarray(probe.point + t * probe.direction,
                                                                                    dtype=jnp.float64)),
                          dtype=np.float64).ravel()

    on_a_set = nearest / 16.0 <= ROUNDING_SHIFT_ULPS * float(np.finfo(np.float64).eps) * reach
    if probe.kind == "on" or on_a_set:
        step = max(1e3 * max(distances.get(probe.owner, 0.0), 1e-300), 1e-6 * probe.size)
        centre = at(0.0)
        sides = ((at(step) - centre) / step, (centre - at(-step)) / step)
        return ShapeCheck(probe, name, autodiff, None, _natural(name, probe.size), 0.0, step, nearest, sides,
                     tuple(problems))
    step = min(1e-4 * probe.size, nearest / 16.0)
    coarse = (at(step) - at(-step)) / (2.0 * step)
    fine = (at(step / 2.0) - at(-step / 2.0)) / step
    difference = (4.0 * fine - coarse) / 3.0
    jacobian = _jacobian(probe.shape, name, whole, numeric64, jnp.asarray(probe.point, dtype=jnp.float64))
    scale = max(float(np.max(np.abs(difference))), float(np.max(np.abs(np.asarray(jacobian)))),
                _natural(name, probe.size))
    size_of_value = float(np.max(np.abs(at(0.0))))
    floor = DIFFERENCE_FLOOR_ULPS * float(np.finfo(np.float64).eps) * (scale * reach + size_of_value) / (step / 2.0)
    shift = ROUNDING_SHIFT_ULPS * float(np.finfo(np.float64).eps) * reach
    fines = [(at(base + step / 2.0) - at(base - step / 2.0)) / step for base in (-shift, shift)] + [fine]
    noise = float(np.max(np.ptp(np.stack(fines), axis=0)))
    floor = max(floor, NOISE_FACTOR * noise)
    floor += DIFFERENCE_FLOOR_ULPS * float(np.finfo(dtype).eps) * scale * reach / max(nearest, 1e-300)
    error = float(np.max(np.abs(autodiff - difference)))
    allowed = SHAPE_TOLERANCE[probe.dtype] * scale + floor
    if not error <= allowed:
        problems.append(f"autodiff {autodiff} and the difference {difference} differ by {error:.3g}, beyond "
                        f"{allowed:.3g} (tolerance {SHAPE_TOLERANCE[probe.dtype]} of {scale:.3g}, floor {floor:.3g})")
    return ShapeCheck(probe, name, autodiff, difference, scale, floor, step, nearest, None, tuple(problems))


def _sphere_place(parameters: Parameters, offset: float, draw: np.ndarray) -> np.ndarray:
    return abs(offset) * _direction_from(draw)


def _direction_from(draw: np.ndarray) -> np.ndarray:
    """A unit vector from two uniform numbers: evenly over the sphere."""
    z, angle = 2.0 * draw[0] - 1.0, 2.0 * math.pi * draw[1]
    across = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([across * math.cos(angle), across * math.sin(angle), z])


def _around_x(radius: float, draw: float) -> np.ndarray:
    angle = 2.0 * math.pi * draw
    return np.array([0.0, radius * math.cos(angle), radius * math.sin(angle)])


def _band(scale: str, ulps: float = AXIS_ULPS) -> Callable[[Parameters, Any], float]:
    return lambda parameters, dtype: ulps * float(np.finfo(dtype).eps) * float(parameters[scale])


def _rounding(scale: str, ulps: float = switch.ROUNDING_ULPS,
              factor: float = 1.0) -> Callable[[Parameters, Any], float]:
    return lambda parameters, dtype: ulps * float(np.finfo(dtype).eps) * float(parameters[scale]) * factor


def screw_focus(parameters: Parameters, trough: bool) -> float | None:
    """Where a crest's (or a trough's) nearest points end, on the ray through it from the axis, in the screw's radius:
    the centre of curvature its nearest points reach, where the Hessian of half the squared distance over the surface's
    coordinates is singular. At the crest or the trough the derivative along the section vanishes, and that Hessian's
    determinant for a point at ``r`` is ``rho r + (rho - r) rho'' (1 + rho nu^2 r)``; its root nearest the surface,
    inside a crest, outside a trough. ``None`` for a trough with no root outside (a convex trough of an untwisted
    screw, whose centre lies beyond the axis, where the seam comes first, or a flat one, whose centre is at
    infinity)."""
    epsilon, starts, nu = float(parameters["modulation"]), int(parameters["starts"]), float(parameters["wavenumber"])
    rho = 1.0 - epsilon if trough else 1.0 + epsilon
    second = (1.0 if trough else -1.0) * starts * starts * epsilon
    a, b, c = -second * rho * nu * nu, rho - second + second * rho * rho * nu * nu, rho * second
    if a == 0.0:
        if b == 0.0:  # an untwisted trough with no curvature, whose centre is at infinity
            return None
        roots = [-c / b]
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        q = -(b + math.copysign(math.sqrt(discriminant), b)) / 2.0
        roots = [q / a, c / q]
    found = [r for r in roots if (r > rho if trough else 0.0 < r < rho)]
    return min(found, key=lambda r: abs(r - rho)) if found else None


#: The farthest from the axis, in the screw's radius, a trough's focus is placed: beyond it, a crest's.
FARTHEST_FOCUS = 3.0


def _screw_focus_place(parameters: Parameters, offset: float, draw: np.ndarray) -> np.ndarray:
    """A point where the screw focus's indicator is ``|offset|``: ``R (|offset| / R)^(2/3)`` from the focal set,
    along the ray through a crest (or a trough) at an axial position, towards the surface, the side where that
    crest's point is the one nearest (on the other side the seam runs in from the set). A trough's focus where it lies within ``FARTHEST_FOCUS`` radii of the axis and ``draw[1]``
    chooses it, a crest's otherwise; the lobe and the axial position from ``draw[2]`` and ``draw[0]``."""
    radius, starts = float(parameters["radius"]), int(parameters["starts"])
    trough_at = screw_focus(parameters, trough=True)
    trough = draw[1] >= 0.5 and trough_at is not None and trough_at <= FARTHEST_FOCUS
    at = trough_at if trough and trough_at is not None else screw_focus(parameters, trough=False)
    assert at is not None, "every crest has its focus between the axis and the crest"
    lobe = min(int(draw[2] * starts), starts - 1)
    section = (-1.0 if trough else 1.0) * math.pi / (2.0 * starts) + 2.0 * math.pi * lobe / starts
    x = (draw[0] - 0.5) * float(parameters["length"])
    out = at * radius + (-1.0 if trough else 1.0) * radius * (abs(offset) / radius) ** (2.0 / 3.0)
    angle = section + float(parameters["wavenumber"]) * (x + float(parameters["length"]) / 2.0) / radius
    return np.array([x, out * math.cos(angle), out * math.sin(angle)])


def _length(low: float, high: float) -> st.SearchStrategy[float]:
    """A length drawn evenly in its logarithm, from ``low`` to ``high`` metres."""
    return st.floats(math.log10(low), math.log10(high)).map(lambda exponent: float(10.0 ** exponent))


def _axial(parameters: Parameters, turn: np.ndarray, shift: np.ndarray) -> Parameters:
    """A shape with a centre and an axis, its own along x through the origin, turned and moved."""
    return {**parameters, "centre": tuple(shift), "axis": tuple(_rotation(turn)[:, 0])}


register_shape_case(ShapeCase(
    shape="sphere",
    parameters=st.fixed_dictionaries({"radius": _length(1e-5, 1e-2)}).map(
        lambda held: {"centre": (0.0, 0.0, 0.0), **held}),
    size=lambda parameters: parameters["radius"],
    turned=lambda parameters, turn, shift: {**parameters, "centre": tuple(shift)},
    sets=(SwitchSet("sphere.centre", _sphere_place, _band("radius")),),
    moved=("radius", "centre"),
))

register_shape_case(ShapeCase(
    shape="cylinder",
    parameters=st.fixed_dictionaries({"radius": _length(1e-4, 1e-2)}).map(
        lambda held: {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), **held}),
    size=lambda parameters: parameters["radius"],
    turned=_axial,
    sets=(SwitchSet("cylinder.axis", lambda p, offset, draw: np.array([(2.0 * draw[0] - 1.0) * p["radius"], 0.0, 0.0])
                    + _around_x(abs(offset), draw[1]), _band("radius")),),
    moved=("radius", "centre", "axis"),
    unplaced={"cylinder.axis_norm": "an axis of zero length, which is no direction: no point is at it"},
))

register_shape_case(ShapeCase(
    shape="capsule",
    parameters=st.fixed_dictionaries({"radius": _length(1e-5, 5e-3), "height": st.floats(0.5, 4.0)}).map(
        lambda held: {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": held["radius"],
                      "height": held["height"] * held["radius"]}),
    size=lambda parameters: parameters["radius"] + parameters["height"] / 2.0,
    turned=_axial,
    sets=(
        SwitchSet("capsule.segment", lambda p, offset, draw: np.array([0.8 * (2.0 * draw[0] - 1.0) * p["height"] / 2.0,
                                                                       0.0, 0.0]) + _around_x(abs(offset), draw[1]),
                  _band("radius")),
        SwitchSet("capsule.end.high", lambda p, offset, draw: np.array([p["height"] / 2.0 - offset, 0.0, 0.0])
                  + _around_x((0.2 + 1.8 * draw[2]) * p["radius"], draw[1]), _rounding("height", factor=0.5)),
        SwitchSet("capsule.end.low", lambda p, offset, draw: np.array([-p["height"] / 2.0 + offset, 0.0, 0.0])
                  + _around_x((0.2 + 1.8 * draw[2]) * p["radius"], draw[1]), _rounding("height", factor=0.5)),
    ),
    moved=("radius", "height", "centre", "axis"),
))

register_shape_case(ShapeCase(
    shape="slab",
    parameters=st.fixed_dictionaries({"half_thickness": _length(1e-5, 5e-3)}).map(
        lambda held: {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), **held}),
    size=lambda parameters: parameters["half_thickness"],
    turned=_axial,
    sets=(SwitchSet("slab.mid", lambda p, offset, draw: np.array([abs(offset), 0.0, 0.0])
                    + _around_x(2.0 * draw[0] * p["half_thickness"], draw[1]), _band("half_thickness")),),
    moved=("half_thickness", "centre", "axis"),
))

register_shape_case(ShapeCase(
    shape="dejongh_screw",
    parameters=st.fixed_dictionaries({
        "radius": _length(5e-4, 3e-3), "length": st.floats(2.0, 6.0), "wavenumber": st.floats(-4.0, 4.0),
        "modulation": st.floats(0.02, 0.5), "starts": st.sampled_from((2, 3, 4))}).map(
        lambda held: {"centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), **held,
                      "length": held["length"] * held["radius"]}),
    size=lambda parameters: parameters["radius"],
    turned=lambda parameters, turn, shift: {**parameters, "centre": tuple(shift), "orientation": tuple(turn)},
    sets=(SwitchSet("dejongh_screw.seam", lambda p, offset, draw: np.array([(draw[0] - 0.5) * p["length"], 0.0, 0.0])
                    + _around_x(abs(offset), draw[1]), _rounding("radius")),
          SwitchSet("dejongh_screw.focus", lambda p, offset, draw: _screw_focus_place(p, offset, draw),
                    _band("radius"))),
    moved=("radius", "length", "wavenumber", "modulation", "centre", "orientation"),
    plain=("wavenumber", "modulation"),
    capabilities=("distance", "normal", "nearest", "tangent"),
))


__all__ = ["DIFFERENCE_FLOOR_ULPS", "NOISE_FACTOR", "PLANTED_TANGENT", "PROBE_CLASSES", "ROUNDING_SHIFT_ULPS",
           "SHAPE_CASES", "SHAPE_TOLERANCE", "ShapeCase", "ShapeCheck", "ShapeProbe", "SwitchSet", "capability",
           "check_shape", "moved_rate", "register_shape_case", "screw_focus", "shape_probes", "switch_distances",
           "uncased_shapes"]
