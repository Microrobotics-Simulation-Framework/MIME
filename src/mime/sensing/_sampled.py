"""Route 2, sampled: a body whose geometry kind declares a cheap sign function (``Geometry.sign_function``), imaged
without a closed form, by a march along each ray.

**The march.** Each ray's line through the body's enclosing ball, widened by ``MARGIN_PX`` pixels' footprints, is
sampled at ``samples`` evenly spaced points in one reduction (no loop): the marched function is the first-order
distance ``value / slope`` itself, never read at another function's least point (a Lipschitz bound marched and the
first-order distance read at its least sits off by a quarter of a pixel at any count). Its local minima among the
samples are found, the two least kept, and each refined by the parabola through its neighbours. The search is held:
at a least point the distance does not move with the point along the ray (its derivative there is zero), so its
value's derivative is the live evaluation's at the held point, exact to the first order.

**Two branches, one corner.** Each of the two minima is a piece in route 1's sense (the first-order distance at its
held point, its line in the pixel by ``jvp`` along the ray table), composed as a union (``lesser``). Where both lie
within ``CORNER_REACH`` of the pixel's centre, two branches of the outline cross the pixel (a body occluding itself,
seen as two separated minima along the ray: FL-9's lobes meet in concave notches of about 80 degrees, where one
straight edge misses a quarter of the pixel), and route 1's corner sub-samples take it, each re-evaluating both
pieces at its held points (exact to the second order in the offset). Where there is no second minimum, the second
piece is the first moved out by the enclosing radius, which never reaches the pixel.

**The chord** (for an attenuating body) is the samples' sign changes, each refined by linear interpolation: a body
thinner than the samples' spacing along a ray can be missed in its chord, never in its coverage. The interval is held
and read only by the flags.

**The sample count** is a declaration, ``SAMPLES`` (32, which camera truth 1 found within 1 % of the physical outline
in motion off the crease, at 160 x 120 to 640 x 480 on FL-9), raised for a slender body so that neighbouring samples
lie at most ``SPACING`` of its smallest feature apart (``samples_for``): a structural rule, never a measured time."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax

from mime.core import switch
from mime.core._node import jnp
from mime.registry import EntryKind, Geometry, Solid, implements, lookup, registered
from mime.sensing import _forms as forms
from mime.sensing._forms import Form, Vec

lax: Any = jax.lax
#: JAX's jit, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_jit: Any = cast(Any, jax).jit

#: Samples a ray by default: camera truth 1's count, within 1 % of the physical silhouette in motion off the crease.
SAMPLES = 32
#: The most neighbouring samples may lie apart, as a share of a body's smallest feature (its least radius of
#: curvature): a slender body's count rises to keep it.
SPACING = 0.5
#: Pixels' footprints the march's enclosing ball is widened by, so that a ray grazing the body's outline from outside
#: is sampled where its distance is least.
MARGIN_PX = 2.0
#: Flops-equivalents a transcendental counts as in the structural cost estimate: the count that predicted the
#: renderer's time (camera optimisation 1).
TRANSCENDENTAL_WEIGHT = 20.0
#: Evaluations a ray costs besides its samples: each of the two pieces' value and its line's two derivatives.
PIECE_EVALUATIONS = 6
#: Evaluations a ray costs on a frame whose corner sub-samples run: every pixel's sub-pixels (``CORNER_SAMPLES``
#: squared, 16), each the two pieces' value and line.
CORNER_EVALUATIONS = 16 * PIECE_EVALUATIONS
#: The structural cost estimate above which the build warns, in flops-equivalents a frame: a declaration, about 50 to
#: 70 ms on a four-core runner at the rate camera optimisation 1 measured for this renderer (15 to 20 G a second).
COST_LINE = 1.0e9


@dataclass(frozen=True)
class Sampled:
    """A body route 2 images: its geometry kind (made from its settings), the region it marches, its smallest
    feature's size and its enclosing radius (m), and the samples a ray."""

    kind: Geometry
    region: str
    feature: float
    enclosing: float
    samples: int


def kind_named(name: str) -> type[Geometry] | None:
    """The newest registered version of the geometry kind ``name``, or ``None``."""
    found = [record for record in registered(EntryKind.GEOMETRY) if record.name == name]
    if not found:
        return None
    newest = max(found, key=lambda record: tuple(int(p) if p.isdigit() else 0 for p in record.version.split(".")))
    held = lookup(EntryKind.GEOMETRY, name, newest.version)
    return None if held is None else held.cls  # type: ignore[return-value]


def creased(kind: type[Geometry]) -> bool:
    """Whether a kind's body has creases: a solid whose body is the intersection of two or more regions."""
    return issubclass(kind, Solid) and len(kind.regions) >= 2


def has_sign(kind: type[Geometry]) -> bool:
    return implements(kind, "sign_function")


def samples_for(enclosing: float, feature: float) -> int:
    """The samples a ray for a body of ``enclosing`` radius whose smallest feature is ``feature``: ``SAMPLES``, or
    more where the march's longest line (the enclosing ball's diameter) would leave its samples more than ``SPACING``
    of the feature apart."""
    return max(SAMPLES, math.ceil(2.0 * enclosing / (SPACING * feature)) + 1)


def evaluate(sampled: Sampled, p: Mapping[str, Any], x: Any, y: Any, z: Any) -> Any:
    """The marched function: the region's first-order distance, ``value / slope``."""
    found = sampled.kind.sign_function(p, sampled.region, (x, y, z))
    return found.value / found.slope


class March(NamedTuple):
    """The march's held result on each ray: the least sample's refined point and the least other local minimum's,
    whether there is another, the samples' points and spacing, which spacings lie inside the body, which it is entered
    or left in (each a held mask, one or zero), and the held interval the flags read."""

    ta: Any
    tb: Any
    has2: Any
    t: Any
    dt: Any
    inside: Any
    entering: Any
    leaving: Any
    interval: tuple[Any, Any]


def _margins(v: Any) -> Any:
    """How far each interior sample lies below the lower of its neighbours (positive at a local minimum)."""
    return jnp.minimum(v[..., :-2], v[..., 2:]) - v[..., 1:-1]


def _gap(values: Any) -> Any:
    """How far the least of each row lies below the next least (zero at a tie)."""
    low = jnp.sort(values, axis=-1)
    return low[..., 1] - low[..., 0]


def march(sampled: Sampled, p: Mapping[str, Any], o: Vec, d: Vec, du: Vec) -> March:
    """The held search along rays ``o + t d`` (``du`` the ray table's derivative along a pixel's row, for the
    footprint): the samples across the widened enclosing ball, the least sample (the first branch), the least of the
    other local minima (the second), each refined by the parabola through its neighbours."""
    o, d, du = lax.stop_gradient((o, d, du))
    held_p = {name: value if isinstance(value, int) else lax.stop_gradient(value) for name, value in p.items()}
    n = sampled.samples
    b = forms.dot(o, d)
    footprint = jnp.sqrt(forms.dot(o, o) * forms.dot(du, du))
    reach = sampled.enclosing + MARGIN_PX * footprint
    half = jnp.sqrt(jnp.maximum(b * b - (forms.dot(o, o) - reach * reach), 0.0))
    t0 = jnp.maximum(-b - half, 0.0)
    t1 = jnp.maximum(-b + half, t0)
    dt = (t1 - t0) / (n - 1)
    k = jnp.arange(n, dtype=t0.dtype)
    t = t0[..., None] + dt[..., None] * k
    v = evaluate(sampled, held_p, *(o[i] + t * d[i][..., None] for i in range(3)))

    def resolution() -> Any:
        return switch.rounding(reach)

    first = switch.argmin(v, owner="camera.sampled.least", resolution=resolution, indicator=lambda: _gap(v))
    margins = _margins(v)
    others = jnp.where(jnp.arange(1, n - 1) == first[..., None], -jnp.inf, margins)
    basins = switch.where(margins > 0.0, v[..., 1:-1], jnp.inf, owner="camera.sampled.basin", resolution=resolution,
                          indicator=lambda: margins)
    rest = jnp.where(jnp.arange(1, n - 1) == first[..., None], jnp.inf, basins)
    second = switch.argmin(rest, owner="camera.sampled.second", resolution=resolution, indicator=lambda: _gap(rest)) + 1
    has2 = switch.predicate(jnp.max(others, axis=-1) > 0.0, owner="camera.sampled.branch", resolution=resolution,
                            indicator=lambda: jnp.max(others, axis=-1))

    def refined(at: Any) -> Any:
        def near(offset: int) -> Any:
            return jnp.take_along_axis(v, (at + offset)[..., None], axis=-1, mode="clip")[..., 0]

        before, least, after = near(-1), near(0), near(1)
        curvature = before - 2.0 * least + after
        inner = (at > 0) & (at < n - 1) & (curvature > 0.0)
        shift = 0.5 * (before - after) / switch.domain_guard(inner, curvature, 1.0)
        return t0 + (at.astype(t0.dtype) + jnp.where(inner, shift, 0.0)) * dt

    ta = refined(first)
    tb = refined(second)
    entering = (v[..., :-1] >= 0.0) & (v[..., 1:] < 0.0)
    leaving = (v[..., :-1] < 0.0) & (v[..., 1:] >= 0.0)
    inside = (v[..., :-1] < 0.0) & (v[..., 1:] < 0.0)
    across = t[..., :-1] + dt[..., None] * v[..., :-1] / jnp.where(entering | leaving, v[..., :-1] - v[..., 1:], 1.0)
    interval = (jnp.min(jnp.where(entering, across, forms.FAR), axis=-1),
                jnp.max(jnp.where(leaving, across, -forms.FAR), axis=-1))
    return March(ta, tb, has2, t, dt, inside.astype(v.dtype), entering.astype(v.dtype), leaving.astype(v.dtype),
                 interval)


def chord(sampled: Sampled, p: Mapping[str, Any], o: Vec, d: Vec, held: March) -> Any:
    """The length of each ray inside the body: each spacing between neighbouring samples that the march found inside,
    and where it found the body entered or left, the share inside by linear interpolation of the live values at the
    held samples. The masks are held: a sample crossing the surface moves the crossing from one spacing to its
    neighbour, where the interpolation puts it at the same place, so the chord is continuous across it."""
    v = evaluate(sampled, p, *(o[i] + held.t * d[i][..., None] for i in range(3)))
    a, b = v[..., :-1], v[..., 1:]
    changes = (held.entering + held.leaving) > 0.0
    share = a / switch.domain_guard(changes, a - b, 1.0)
    within = held.inside + held.leaving * share + held.entering * (1.0 - share)
    return jnp.sum(within, axis=-1) * held.dt


def pieces(sampled: Sampled, held: March) -> Any:
    """The two branches' pieces on rays: each the first-order distance at its held point; the second, where there is
    no second minimum, the first moved out by the enclosing radius."""

    def made(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
        def at(t: Any) -> Any:
            return evaluate(sampled, p, *(o[i] + t * d[i] for i in range(3)))

        first = at(held.ta)
        return first, jnp.where(held.has2, at(held.tb), first + sampled.enclosing)

    return made


def _union(parts: Sequence[Any], lesser: Any, greater: Any) -> Any:
    return lesser(parts[0], parts[1])


def form(sampled: Sampled, held: March | None = None) -> Form:
    """Route 2's form: with ``held``, the march done once on these rays, its pieces at the held points (for the
    coverage and its corner sub-samples); without, a form whose interval marches on the rays it is given (a sampled
    body as another subject's parent)."""

    def interval(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, Any]:
        zero = jnp.zeros_like(d[0])
        return march(sampled, p, o, d, (zero, zero, zero)).interval

    def enclosing(settings: Mapping[str, float]) -> float:
        return sampled.enclosing

    if held is None:
        def unmarched(p: Mapping[str, Any], o: Vec, d: Vec) -> tuple[Any, ...]:
            raise TypeError("a sampled form's pieces are made from its march (form(sampled, held))")

        return Form(interval, unmarched, enclosing, _union, corners=True)
    return Form(interval, pieces(sampled, held), enclosing, _union, corners=True)


def evaluation_cost(sampled: Sampled, parameters: Mapping[str, Any], dtype: Any) -> float:
    """The flops-equivalents of one evaluation of the marched function (flops plus ``TRANSCENDENTAL_WEIGHT`` times
    the transcendentals), from XLA's cost analysis of the lowered program on one point: a count of the program, never
    a time."""
    point = jax.ShapeDtypeStruct((), dtype)
    held = {name: (value if isinstance(value, int) else jnp.asarray(value, dtype=dtype))
            for name, value in parameters.items()}
    def one(x: Any, y: Any, z: Any) -> Any:
        return evaluate(sampled, held, x, y, z)

    cost: Mapping[str, float] = _jit(one).lower(point, point, point).cost_analysis() or {}
    return float(cost.get("flops", 0.0)) + TRANSCENDENTAL_WEIGHT * float(cost.get("transcendentals", 0.0))


def frame_cost(sampled: Sampled, parameters: Mapping[str, Any], rays: int, sub_frames: int, dtype: Any) -> float:
    """The structural cost estimate of a body's frame, in flops-equivalents: its rays (its window's) times the
    sub-frames times one evaluation's count times the evaluations a ray takes (its samples, its pieces, and the corner
    sub-samples, counted as running: a frame with a corner in view runs them on every ray)."""
    per_ray = sampled.samples + PIECE_EVALUATIONS + CORNER_EVALUATIONS
    return float(rays) * sub_frames * per_ray * evaluation_cost(sampled, parameters, dtype)


def made_kind(kind: type[Geometry], settings: Mapping[str, Any]) -> Geometry:
    """The kind made from its settings in SI (each dimensioned one a quantity, a list a tuple)."""
    import dataclasses

    from mime.registry import registration_of
    from mime.units import Quantity

    record = registration_of(kind)
    made: dict[str, Any] = {}
    for item in dataclasses.fields(cast(Any, kind)):
        value: Any = settings[item.name]
        value = tuple(cast(list[Any], value)) if isinstance(value, list) else value
        declared = record.settings.get(item.name) if record is not None else None
        made[item.name] = Quantity(value, declared.dimension) if declared is not None and declared.dimension is not None \
            else value
    return kind(**made)


def smallest_length(kind: type[Geometry], settings: Mapping[str, Any]) -> float:
    """The smallest of a kind's length settings (each component of a vector one): the structural stand-in for its
    smallest feature that sets the samples a ray (a sphere's or capsule's radius, a helix's filament, an ellipsoid's
    least semi-axis)."""
    from mime.registry import registration_of
    from mime.units import Length

    record = registration_of(kind)
    lengths: list[float] = []
    for name, declared in (record.settings.items() if record is not None else ()):
        if declared.dimension != Length:
            continue
        value: Any = settings[name]
        held: Sequence[Any] = cast(Sequence[Any], value) if isinstance(value, (list, tuple)) else (value,)
        lengths += [abs(float(v)) for v in held]
    positive = [v for v in lengths if v > 0.0]
    return min(positive) if positive else math.inf


__all__ = ["COST_LINE", "MARGIN_PX", "SAMPLES", "SPACING", "Sampled", "chord", "creased", "evaluate", "evaluation_cost",
           "form", "frame_cost",
           "has_sign", "kind_named", "made_kind", "march", "samples_for", "smallest_length"]
