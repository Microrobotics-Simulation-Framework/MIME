"""The shape conformance suite: each registered shape's derivatives equal the finite difference on the point's side of
every switch it declares, at points off, near and on each declared set, under a random rotation and placement.

**Cases.** Each shape has a :class:`ShapeCase` (``SHAPE_CASES``, by its registered name): a strategy for its
parameters in its own frame at physical scales, its size, how it is turned and moved, the scalar parameters a study
may move, and for each switch it declares how a point is placed a signed distance from the set (the indicator's value
there) and the band's width. A registered shape with no case fails the walk (:func:`uncased_shapes`).

**A probe** (:func:`shape_probes`) picks a set and a class: ``off`` (a twentieth of the shape's size or more from the set),
``near`` (4 to 10^4 of the set's bands for the dtype drawn, never more than a tenth of the size) or ``on`` (on the
set, inside its band); a direction for the point and a parameter to move with it, together one curve. Autodiff's
derivative along the curve (a JVP) is taken in the dtype drawn; the reference is the central difference along it,
extrapolated (Richardson), always in float64. Its step is at most a sixteenth of the distance to the nearest switch at
the point, which every owner's indicator gives through a collector (half the gap for the screw's seam, whose
indicator is the gap between its two nearest roots), so no difference straddles a switch. On a set no difference is
taken: both one-sided slopes are reported, not compared, and autodiff must be finite there.

**Dtypes** [#168]. A float32 probe is placed in float32's bands, so it lies outside both dtypes' bands, and its
autodiff (computed in float32 throughout) is compared with the float64 difference: the only float32 against float64
comparison.

**The criterion**: ``|AD - FD| <= SHAPE_TOLERANCE[dtype] S + floor``, ``S`` the derivative's scale (the larger of the
difference's size and the capability's own: 1 for the distance and the nearest point, 1 / size for the normal and the
tangent). The floor is rounding only (``DIFFERENCE_FLOOR_ULPS`` units in the last place, measured to cover a point's
rounding through the shapes' few operations): the difference's own, ``k eps64 (|J| (|p| + size) + |f|) / h``, and in
float32 autodiff's, ``k eps32 S (|p| + size) / d``, ``d`` the distance to the nearest switch. No floor comes from two
differences' difference: ``2 |coarse - fine|`` accepts any tangent across a switch.

**The control**: :func:`check_shape` with ``planted`` scales autodiff's tangent by ``1 + planted`` through a ``custom_jvp``;
:data:`PLANTED_TANGENT` lies above the suite's measured agreement, by the margin its test records."""

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
#: The planted wrong tangent's relative error: above the float64 tolerance, so that a probe placed where the floor is
#: negligible must reject it.
PLANTED_TANGENT = 1e-4
#: The rounding floor's safety factor, in units in the last place.
DIFFERENCE_FLOOR_ULPS = 128.0
PROBE_CLASSES = ("off", "near", "on")
_jit: Any = cast(Any, jax).jit
_jvp: Any = cast(Any, jax).jvp
_custom_jvp: Any = cast(Any, jax).custom_jvp
_tree_map: Any = cast(Any, jax).tree_util.tree_map


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
    turned by a unit quaternion and moved by a translation), the scalar parameters a study may move (``moved``, with
    the dimensionless ones in ``plain``: a probe's curve moves a length by up to its own distance along it, and a
    dimensionless parameter by up to that over the size), its sets, and its capabilities. ``unplaced`` names each
    declared owner no point can be placed at, with why."""

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


def uncased_shapes() -> list[str]:
    """Each registered shape with no case, and each owner a shape declares that its case neither places nor says why it
    cannot."""
    from mime.core import shape_switches

    found: list[str] = []
    for name in shape_names():
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
    moved: Mapping[str, float]
    owner: str
    kind: str
    offset: float
    dtype: str
    size: float


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _rotation(q: np.ndarray) -> np.ndarray:
    return np.asarray(quaternion.rotation_matrix(jnp.asarray(q, dtype=jnp.float64)))


@st.composite
def shape_probes(draw: Any, case: ShapeCase, kinds: Sequence[str] = PROBE_CLASSES, dtypes: Sequence[str] = ("float64", "float32")
           ) -> ShapeProbe:
    """A probe of ``case``: its parameters, a random rotation and placement, a set, a class, a direction and a
    parameter moved with the point."""
    own = draw(case.parameters)
    size = case.size(own)
    turn = _unit(np.array(draw(st.tuples(*(st.floats(-1.0, 1.0),) * 4)), dtype=np.float64) + np.array(
        [1e-3, 0.0, 0.0, 0.0]))
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
    direction = _unit(np.array(draw(st.tuples(*(st.floats(-1.0, 1.0),) * 3))) + np.array([1e-6, 0.0, 0.0]))
    name = draw(st.sampled_from(case.moved))
    rate = draw(st.floats(-1.0, 1.0))
    moved = {name: rate / size if name in case.plain else rate}
    return ShapeProbe(case.shape, case.turned(own, turn, shift), point, direction, moved, chosen.owner, kind, offset,
                 dtype, size)


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
def _value(shape: str, name: str, whole: tuple[tuple[str, int], ...], numeric: dict[str, Any], point: Any) -> Any:
    return capability(shape_named(shape), name)({**numeric, **dict(whole)}, point)


def switch_distances(probe: ShapeProbe) -> dict[str, float]:
    """How far the probe's point is from each switch the shape evaluates there, by owner: half the magnitude of its
    indicator (each MIME shape's indicator is a distance, or the screw seam's gap, at most twice its distance)."""
    numeric, whole = _split(probe.parameters, jnp.float64)
    shape = shape_named(probe.shape)
    held = {**numeric, **dict(whole)}
    point = jnp.asarray(probe.point, dtype=jnp.float64)
    with switch.collecting() as rows:
        shape.signed_distance(held, point)
        shape.outward_normal(held, point)
    found: dict[str, float] = {}
    for row in rows:
        indicator = float(np.min(np.abs(np.asarray(row.indicator, dtype=np.float64))))
        found[row.owner] = min(found.get(row.owner, math.inf), indicator / 2.0)
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

    if probe.kind == "on":
        step = max(1e3 * max(distances.get(probe.owner, 0.0), 1e-300), 1e-6 * probe.size)
        centre = at(0.0)
        sides = ((at(step) - centre) / step, (centre - at(-step)) / step)
        return ShapeCheck(probe, name, autodiff, None, _natural(name, probe.size), 0.0, step, nearest, sides,
                     tuple(problems))
    step = min(1e-4 * probe.size, nearest / 16.0)
    coarse = (at(step) - at(-step)) / (2.0 * step)
    fine = (at(step / 2.0) - at(-step / 2.0)) / step
    difference = (4.0 * fine - coarse) / 3.0
    scale = max(float(np.max(np.abs(difference))), _natural(name, probe.size))
    size_of_value = float(np.max(np.abs(at(0.0))))
    floor = DIFFERENCE_FLOOR_ULPS * float(np.finfo(np.float64).eps) * (scale * reach + size_of_value) / (step / 2.0)
    if dtype == jnp.float32:
        floor += DIFFERENCE_FLOOR_ULPS * float(np.finfo(np.float32).eps) * scale * reach / max(nearest, 1e-300)
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


def _rounding(scale: str, ulps: float = switch.ROUNDING_ULPS, factor: float = 1.0) -> Callable[[Parameters, Any], float]:
    return lambda parameters, dtype: ulps * float(np.finfo(dtype).eps) * float(parameters[scale]) * factor


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
    moved=("radius",),
))

register_shape_case(ShapeCase(
    shape="cylinder",
    parameters=st.fixed_dictionaries({"radius": _length(1e-4, 1e-2)}).map(
        lambda held: {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), **held}),
    size=lambda parameters: parameters["radius"],
    turned=_axial,
    sets=(SwitchSet("cylinder.axis", lambda p, offset, draw: np.array([(2.0 * draw[0] - 1.0) * p["radius"], 0.0, 0.0])
                    + _around_x(abs(offset), draw[1]), _band("radius")),),
    moved=("radius",),
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
    moved=("radius", "height"),
))

register_shape_case(ShapeCase(
    shape="slab",
    parameters=st.fixed_dictionaries({"half_thickness": _length(1e-5, 5e-3)}).map(
        lambda held: {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), **held}),
    size=lambda parameters: parameters["half_thickness"],
    turned=_axial,
    sets=(SwitchSet("slab.mid", lambda p, offset, draw: np.array([abs(offset), 0.0, 0.0])
                    + _around_x(2.0 * draw[0] * p["half_thickness"], draw[1]), _band("half_thickness")),),
    moved=("half_thickness",),
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
                    + _around_x(abs(offset), draw[1]), _rounding("radius")),),
    moved=("radius", "length", "wavenumber", "modulation"),
    plain=("wavenumber", "modulation"),
    capabilities=("distance", "normal", "nearest", "tangent"),
))


__all__ = ["DIFFERENCE_FLOOR_ULPS", "PLANTED_TANGENT", "PROBE_CLASSES", "SHAPE_CASES", "SHAPE_TOLERANCE", "ShapeCase",
           "ShapeCheck", "ShapeProbe", "SwitchSet", "capability", "check_shape", "register_shape_case", "shape_probes",
           "switch_distances", "uncased_shapes"]
