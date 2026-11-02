"""Switches: every non-smooth operation a node's step evaluates, called through this module so that it has a name.

A **switch** is a relation in a step with the branches it selects; each call of a function here is one of its
**sites**. Every switch has an **owner**, a dotted name ``<namespace>.<switch>`` (``contact.band``), declared once by
the entry or shape whose law it is (:class:`Declaration`) and shared by all its sites. Each call also takes an
**indicator**, the signed value whose sign or band selects the branch, in its own units and positive on the side the
call names first (``where``'s ``a``, ``maximum``'s ``x``), and a **resolution**, the width within which the
indicator counts as zero. A core helper that evaluates a switch takes its owner from its caller.

Each function returns its operation's value and JAX's derivative of it, ties included: ``maximum`` and ``minimum``
take half of each derivative at a tie, ``absolute`` the derivative ``+1`` at zero, ``clip`` half at a bound, and
``where`` the selected branch's; ``searchsorted``'s and ``argmin``'s indices, integers, have none. ``norm`` is MIME's safe norm:
``|x|``, with the derivative zero at ``x = 0`` where ``jnp.linalg.norm``'s is NaN. So writing a site through this
module changes no program.

**The context.** The resolution and the indicator are each a value or a zero-argument callable, evaluated only inside
a collector, so with no context open the traced program is exactly the operation's. A value already computed at the
site may be passed as it is; one not computed there is passed as a callable. The one context is empty by default and
set only by a context manager: :func:`collecting`, which gathers a :class:`Row` per evaluation, or the test-only
setting (:func:`forcing` a branch per owner, :func:`replacing_fallbacks` of every domain guard). It is read while a
step is traced, so whatever reads it traces the step afresh inside it: a step JAX traced before (a ``jit``'s cached
trace) does not run its Python again, and sees nothing.

Four selections are not switches and take no owner: :func:`domain_guard`, which only keeps an undefined value (a
square root's derivative at zero, a division by zero) out of a branch that is discarded; :func:`nonzero_norm`, the
safe norm of a vector zero at no valid state, whose zero branch is a domain guard no valid state selects;
:func:`static_select`, whose mask is fixed when the step is built; and :func:`without_slope`, a value that has no
derivative where its mask says so, the one place a derivative is made NaN on purpose.

This module imports only JAX, jaxtyping and the standard library, so it can be lifted out of MIME whole.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy
from jax import Array

#: ``jax.numpy`` and ``jax.lax``, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
jnp: Any = jax.numpy
lax: Any = jax.lax
_TRACER: type = cast(Any, jax).core.Tracer

#: A value, or a zero-argument callable that computes it, called only inside a collector.
type Lazy = Any | Callable[[], Any]

#: An owner: dotted lower-case words, the first the namespace (``contact.band``, ``mime_lab.nudge.cap``).
OWNER_FORM = re.compile(r"[a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)+")
#: A resolution's width where nothing else sets one, in units in the last place of its indicator's scale: the
#: rounding of an indicator computed in a few operations from values of that scale.
ROUNDING_ULPS = 64


def rounding(scale: Any) -> Array:
    """``ROUNDING_ULPS`` units in the last place of ``|scale|``, in its dtype: a resolution for an indicator computed
    from values of that scale. Passed as a callable, it is evaluated only inside a collector."""
    held = jnp.asarray(scale)
    return ROUNDING_ULPS * jnp.finfo(held.dtype).eps * jnp.abs(held)


@dataclass(frozen=True, kw_only=True)
class Declaration:
    """One switch, as the entry or shape whose law it is declares it: its ``owner``; what its ``indicator`` measures,
    with its units; its smoothness ``orders``, per output group of the node that evaluates it (an entry's) or per
    capability (a shape's): ``-1`` a jump, ``0`` a kink, ``k >= 1`` continuously differentiable ``k`` times; for a
    time switch, whose indicator reads only the step count and settings, the ``settings`` it reads; the ``route``
    to an exact derivative, or ``"none: "`` and why there is none; and, where no output can show its branches, why
    (``unmeasured``): never live, or live with no effect an output shows. The switch harness measures the orders of
    every other switch at the outputs, and finds such a switch's row measuring nothing."""

    owner: str
    indicator: str
    orders: Mapping[str, int]
    route: str
    settings: tuple[str, ...] = ()
    unmeasured: str | None = None

    def __post_init__(self) -> None:
        if not OWNER_FORM.fullmatch(self.owner):
            raise ValueError(f"switch owner {self.owner!r} is not dotted lower-case words, <namespace>.<switch>")
        if not self.indicator.strip() or not self.route.strip() or \
                (self.unmeasured is not None and not self.unmeasured.strip()):
            raise ValueError(f"switch {self.owner}: its indicator, its route and why it is unmeasured are each a "
                             f"sentence")
        orders = cast(Mapping[str, object], self.orders)  # an entry's own declaration, checked like a script's
        if not orders or any(isinstance(order, bool) or not isinstance(order, int) or order < -1
                             for order in orders.values()):
            raise ValueError(f"switch {self.owner}: orders {dict(self.orders)!r} are not an integer from -1 on per "
                             f"output group")

    @property
    def lowest_order(self) -> int:
        """The order a run counts it at: its outputs feed later steps."""
        return min(self.orders.values())

    @property
    def jump(self) -> bool:
        return self.lowest_order < 0


@dataclass(frozen=True)
class Row:
    """One evaluation of a switch, as a collector gathers it: the owner, the function, and the indicator, resolution
    and branch taken (true on the side the call names first), each held out of every derivative."""

    owner: str
    function: str
    indicator: Array
    resolution: Array
    branch: Array


@dataclass
class _Collector:
    rows: list[Row] = field(default_factory=list[Row])


@dataclass
class Reached:
    """What the traces made inside a test-only setting reached: the forced owners evaluated; how many domain guards'
    fallbacks were replaced; each domain guard called from outside this module, by its call site (``file:line``),
    with whether its condition was false anywhere it was evaluated on a value (a tracer's is unknown, and counts as
    not false); and each :func:`nonzero_norm`, by its call site, with whether its vector was zero anywhere. Empty when
    the step that ran was traced before the setting opened."""

    owners: set[str] = field(default_factory=set[str])
    guards: int = 0
    guard_sites: dict[str, bool] = field(default_factory=dict[str, bool])
    nonzero_norms: dict[str, bool] = field(default_factory=dict[str, bool])


@dataclass
class _Setting:
    forced: Mapping[str, bool]
    fallback_scale: float | None
    reached: Reached


_context: ContextVar[_Collector | _Setting | None] = ContextVar("mime_switch_context", default=None)


@contextmanager
def _opened(context: _Collector | _Setting) -> Generator[None]:
    token = _context.set(context)
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def collecting() -> Generator[list[Row]]:
    """Gather a :class:`Row` for every switch evaluated while a step is traced inside it, in evaluation order."""
    collector = _Collector()
    with _opened(collector):
        yield collector.rows


@contextmanager
def forcing(branches: Mapping[str, bool]) -> Generator[Reached]:
    """Test-only: every site of each owner in ``branches`` takes the branch given (true: the one the call names
    first), in steps traced inside it. Yields what those traces reached."""
    setting = _Setting(dict(branches), None, Reached())
    with _opened(setting):
        yield setting.reached


@contextmanager
def replacing_fallbacks(scale: float) -> Generator[Reached]:
    """Test-only: every domain guard's fallback is multiplied by ``scale``, or, where it is zero, which no scale
    changes, replaced by ``scale`` itself, in steps traced inside it. Yields what those traces reached."""
    setting = _Setting({}, scale, Reached())
    with _opened(setting):
        yield setting.reached


def _evaluated(value: Lazy) -> Any:
    return value() if callable(value) else value


def _forced(owner: str) -> bool | None:
    context = _context.get()
    if isinstance(context, _Setting) and owner in context.forced:
        context.reached.owners.add(owner)
        return context.forced[owner]
    return None


def _collect(owner: str, function: str, resolution: Lazy, indicator: Lazy, branch: Callable[[], Any]) -> None:
    context = _context.get()
    if isinstance(context, _Collector):
        held: Callable[[Any], Array] = lambda value: lax.stop_gradient(jnp.asarray(value))
        context.rows.append(Row(owner, function, held(_evaluated(indicator)), held(_evaluated(resolution)),
                                held(branch())))


def _selected(branch: bool, a: Any, b: Any) -> Array:
    return jnp.where(branch, a, b)


def maximum(x: Any, y: Any, *, owner: str, resolution: Lazy, indicator: Lazy) -> Array:
    """``jnp.maximum(x, y)``; the branch ``x`` where ``x > y``."""
    forced = _forced(owner)
    if forced is not None:
        return _selected(forced, x, y)
    _collect(owner, "maximum", resolution, indicator, lambda: jnp.asarray(x) > y)
    return jnp.maximum(x, y)


def minimum(x: Any, y: Any, *, owner: str, resolution: Lazy, indicator: Lazy) -> Array:
    """``jnp.minimum(x, y)``; the branch ``x`` where ``x < y``."""
    forced = _forced(owner)
    if forced is not None:
        return _selected(forced, x, y)
    _collect(owner, "minimum", resolution, indicator, lambda: jnp.asarray(x) < y)
    return jnp.minimum(x, y)


def absolute(x: Any, *, owner: str, resolution: Lazy, indicator: Lazy) -> Array:
    """``jnp.abs(x)``; the branch ``x`` where ``x > 0``, ``-x`` otherwise."""
    forced = _forced(owner)
    if forced is not None:
        return _selected(forced, x, -jnp.asarray(x))
    _collect(owner, "absolute", resolution, indicator, lambda: jnp.asarray(x) > 0)
    return jnp.abs(x)


def clip(x: Any, low: Any, high: Any, *, owner: str, resolution: Lazy, indicator: Lazy) -> Array:
    """``jnp.clip(x, low, high)``: two switches, ``<owner>.low`` (the branch ``x`` where ``x > low``, else ``low``)
    and ``<owner>.high`` (``x`` where ``x < high``, else ``high``). ``indicator`` is the clipped quantity in its own
    units; each bound's indicator is its distance from that bound, positive inside."""
    low_owner, high_owner = f"{owner}.low", f"{owner}.high"
    forced_low, forced_high = _forced(low_owner), _forced(high_owner)
    if forced_low is not None or forced_high is not None:
        above = jnp.maximum(x, low) if forced_low is None else _selected(forced_low, x, low)
        return jnp.minimum(above, high) if forced_high is None else _selected(forced_high, above, high)
    _collect(low_owner, "clip", resolution, lambda: _evaluated(indicator) - low, lambda: jnp.asarray(x) > low)
    _collect(high_owner, "clip", resolution, lambda: high - _evaluated(indicator), lambda: jnp.asarray(x) < high)
    return jnp.clip(x, low, high)


def _norm_of(x: Any, select: Callable[[Any, Any, Any], Array], axis: int | None = None) -> Array:
    """``|x|`` (of each slice along ``axis`` where one is given), its square root taken only of a positive number,
    and ``select``-ed against zero at zero."""
    squared = jnp.sum(x * x, axis=axis)
    positive = squared > 0
    return select(positive, jnp.sqrt(domain_guard(positive, squared, 1.0)), 0.0)


def _safe_norm(x: Any) -> Array:
    """``|x|``, its square root taken only of a positive number and discarded at zero, where its derivative is zero.
    The formula ``norm`` evaluates, and ``mime.core``'s ``safe_norm`` for the callers that evaluate it raw."""
    return _norm_of(x, jnp.where)


def norm(x: Any, *, owner: str, resolution: Lazy, indicator: Lazy) -> Array:
    """MIME's safe norm, ``|x|``, a kink at ``x = 0`` with the derivative zero there; the branch off zero where
    ``|x| > 0``."""
    forced = _forced(owner)
    if forced is not None:
        squared = jnp.sum(x * x)
        return jnp.sqrt(squared) if forced else jnp.zeros_like(squared)
    _collect(owner, "norm", resolution, indicator, lambda: jnp.sum(x * x) > 0)
    return _safe_norm(x)


def where(condition: Any, a: Any, b: Any, *, owner: str, resolution: Lazy, indicator: Lazy) -> Array:
    """``jnp.where(condition, a, b)``; the branch ``a`` where ``condition`` holds."""
    forced = _forced(owner)
    if forced is not None:
        return _selected(forced, a, b)
    _collect(owner, "where", resolution, indicator, lambda: condition)
    return jnp.where(condition, a, b)


def predicate(condition: Any, *, owner: str, resolution: Lazy, indicator: Lazy) -> Any:
    """``condition``, unchanged: a part of a conjunction that a later ``where`` takes, recorded as its own switch. A
    domain guard that takes the same condition follows a forced branch with it."""
    forced = _forced(owner)
    if forced is not None:
        return jnp.full(jnp.shape(condition), forced)
    _collect(owner, "predicate", resolution, indicator, lambda: condition)
    return condition


def argmin(x: Any, *, owner: str, resolution: Lazy, indicator: Lazy) -> Array:
    """``jnp.argmin(x, axis=-1)``: the index of the least of each row of ``x``, the first of those tied. An index has
    no derivative; what it selects is the caller's, and each tie is a jump in it. The branch: the least over the
    next least of its row (forced false, the next least's index); the indicator, the caller's, the next least less
    the least."""
    index = jnp.argmin(x, axis=-1)
    forced = _forced(owner)
    if forced is not None:
        return index if forced else jnp.argsort(x, axis=-1, stable=True)[..., 1].astype(index.dtype)
    _collect(owner, "argmin", resolution, indicator, lambda: jnp.ones(jnp.shape(index), dtype=bool))
    return index


@dataclass(frozen=True, eq=False)
class Buckets:
    """An index of an increasing array of breaks, made with them, that locates a query without a search. The
    query's coordinate (the query, or its logarithm where ``logarithmic``) is mapped to one of ``len(first)`` equal
    buckets, ``floor((coordinate - origin) * scale)``, clamped to them; every break below the bucket's
    ``first`` is below any query the bucket holds, every break past its ``candidates`` above, so the index is
    ``first`` plus the candidates the query is past. Exact for every query a dtype holds, NaN and the infinities
    included, wherever the buckets are made with a margin wider than the mapping's rounding."""

    origin: Any
    scale: Any
    first: Any
    candidates: Any
    logarithmic: bool


cast(Any, jax).tree_util.register_dataclass(Buckets, data_fields=["origin", "scale", "first", "candidates"],
                                            meta_fields=["logarithmic"])


def searchsorted(breaks: Any, x: Any, *, side: str = "left", method: str = "scan", buckets: Buckets | None = None,
                 owner: str, resolution: Lazy, indicator: Lazy) -> Array:
    """``jnp.searchsorted(breaks, x, side=side, method=method)``: the index of the interval of the increasing
    ``breaks`` that holds each ``x``, ``0`` below the first and ``len(breaks)`` above the last, a break itself in the
    interval above it (``side="right"``) or below it (``"left"``). With ``buckets``, an index of ``breaks`` made
    with them, the same index without a search. Each break is a jump in the index; what the index selects is the
    caller's. The branch: the interval above the break nearest ``x``, the one below it otherwise; the indicator,
    ``x``'s offset from that break."""
    index = _sorted_index(breaks, x, side=side, method=method, buckets=buckets)
    forced = _forced(owner)
    if forced is not None:
        return (_nearest(breaks, x) + (1 if forced else 0)).astype(index.dtype)
    _collect(owner, "searchsorted", resolution, indicator, lambda: index > _nearest(breaks, x))
    return index


def _sorted_index(breaks: Any, x: Any, *, side: str, method: str, buckets: Buckets | None) -> Array:
    """The index :func:`searchsorted` returns, by a search or by ``buckets``. In the buckets, a query is past a
    candidate as JAX's search compares them, in its total order (NaN above every number, ``-0.0`` equal to ``0.0``)."""
    if buckets is None:
        return jnp.searchsorted(breaks, x, side=side, method=method)
    held = jnp.asarray(x)
    first, candidates = jnp.asarray(buckets.first), jnp.asarray(buckets.candidates)
    count = first.shape[0]
    if buckets.logarithmic:
        positive = jnp.maximum(held, jnp.finfo(held.dtype).tiny)  # NaN stays NaN; zero and below, the first bucket
        coordinate = (jnp.log(positive) - buckets.origin) * buckets.scale
    else:
        coordinate = (held - buckets.origin) * buckets.scale
    bucket = jnp.floor(jnp.where(coordinate < count - 1, jnp.maximum(coordinate, 0), count - 1)).astype(jnp.int32)
    near = candidates[bucket]
    past = jnp.logical_not(held[..., None] < near) if side == "right" else jnp.logical_not(held[..., None] <= near)
    return first[bucket] + jnp.sum(past, axis=-1, dtype=first.dtype)


def _nearest(breaks: Any, x: Any) -> Array:
    """The index of the break nearest each ``x``."""
    return jnp.argmin(jnp.abs(jnp.asarray(breaks) - jnp.asarray(x)[..., None]), axis=-1)


def _call_site() -> str | None:
    """Where the function of this module that called this one was called from (``file:line``), or ``None`` where
    that is this module itself (a guard inside the safe norm)."""
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
    if caller is None or caller.f_globals.get("__name__") == __name__:
        return None
    return f"{caller.f_code.co_filename}:{caller.f_lineno}"


def _anywhere(value: Any, test: Callable[[Array], Array]) -> bool:
    """Whether ``test`` holds anywhere of ``value``, where it is a value; a tracer's is unknown, and counts as not."""
    return not isinstance(value, _TRACER) and bool(jnp.any(test(jnp.asarray(value))))


def domain_guard(condition: Any, a: Any, b: Any) -> Array:
    """``jnp.where(condition, a, b)`` where ``b``, the fallback, only keeps an undefined value out of a branch that
    is discarded wherever ``condition`` is false: a selection that is no switch, for which a collector records no
    row. Wherever ``condition`` is false, any other fallback must give the same values and derivatives."""
    context = _context.get()
    if isinstance(context, _Setting) and context.fallback_scale is not None:
        context.reached.guards += 1
        site = _call_site()
        if site is not None:
            false = _anywhere(condition, jnp.logical_not)
            context.reached.guard_sites[site] = context.reached.guard_sites.get(site, False) or false
        held = jnp.asarray(b)
        scale = context.fallback_scale
        return jnp.where(condition, a, jnp.where(held == 0, held + scale, scale * held))
    return jnp.where(condition, a, b)


def nonzero_norm(x: Any, axis: int | None = None) -> Array:
    """``|x|`` of an ``x`` that is zero at no valid state (a direction the plan refuses at zero, a quaternion an
    integrator keeps near unit length): the safe norm's formula, whose zero branch is here a domain guard that no
    valid state selects. With ``axis``, the norm of each slice along it (each row's, ``axis=-1``), none of them zero
    at a valid state. No switch: it takes no owner and a collector records no row for it, so a norm whose zero a
    valid state reaches is ``norm``, with its owner. In the test-only setting it records its call site, with whether
    ``x`` (or a slice of it) was zero there, so that a test can show that no valid point reaches its zero."""
    context = _context.get()
    if isinstance(context, _Setting):
        site = _call_site()
        if site is not None:
            zero = _anywhere(x, lambda v: jnp.any(jnp.sum(v * v, axis=axis) == 0))
            context.reached.nonzero_norms[site] = context.reached.nonzero_norms.get(site, False) or zero
    return _norm_of(x, domain_guard, axis)


def static_select(mask: Any, a: Any, b: Any) -> Array:
    """``jnp.where(mask, a, b)`` for a ``mask`` fixed when the step is built (a NumPy array or a Python bool): no
    switch. A traced mask is refused when the step is traced."""
    if isinstance(mask, _TRACER):
        raise TypeError("static_select takes a mask fixed at build time; this one is traced, so it selects by a "
                        "value of the step: write it with where, and declare its owner")
    return jnp.where(mask, a, b)


@jax.custom_jvp
def _slope_scaled(x: Any, factor: Any) -> Array:
    return jnp.asarray(x)


@_slope_scaled.defjvp
def _slope_scaled_jvp(primals: tuple[Any, Any], tangents: tuple[Any, Any]) -> tuple[Array, Array]:
    x, factor = primals
    return jnp.asarray(x), tangents[0] * factor


def without_slope(x: Any, mask: Any) -> Array:
    """``x`` itself, whose derivative is NaN wherever ``mask`` holds (and unchanged elsewhere): no switch. For a value
    read where it is defined at a single point (a table whose readable range is one point), which has no derivative
    there: the evaluator returns none, absent or NaN, rather than a slope of the line through it. The tangent rule is
    linear, so it cannot give NaN along one direction and a number along another (``0 * NaN`` is NaN): reverse mode
    is NaN along the masked input only, and forward mode is NaN in every direction whenever the masked value carries
    a tangent, even a zero one. A masked value that carries no tangent in the mode used (a constant, or a ``jit``
    argument the derivative is not taken in) leaves the other directions finite, so a table read, which takes one
    query per axis, keeps the other axes' derivatives then. Known and kept (``MIME-ANO-115``)."""
    value = jnp.asarray(x)
    return _slope_scaled(value, jnp.where(jnp.asarray(mask), jnp.asarray(jnp.nan, value.dtype),
                                          jnp.ones((), value.dtype)))


#: The functions a node's step evaluates switches through, by name: a call of one of these takes an owner.
SWITCH_FUNCTIONS = ("maximum", "minimum", "absolute", "clip", "norm", "where", "predicate", "searchsorted",
                    "argmin")
#: The selections that are no switch.
SELECTIONS = ("domain_guard", "nonzero_norm", "static_select", "without_slope")

__all__ = ["OWNER_FORM", "ROUNDING_ULPS", "SELECTIONS", "SWITCH_FUNCTIONS", "Buckets", "Declaration", "Lazy", "Reached",
           "Row", "absolute", "argmin", "clip", "collecting", "domain_guard", "forcing", "maximum", "minimum", "nonzero_norm",
           "norm", "predicate", "replacing_fallbacks", "rounding", "searchsorted", "static_select", "where",
           "without_slope"]
