"""The rounding error of the calibrated multipole's forward model, from its own computation path.

The model's rows per ampere at a point are the field and gradient each coil's sources sum to (``source_field``,
summed in ``unit_fields``), the force ``F_j = m_i G_ij`` and the torque ``m x B``; its offset at zero command is the
background and ``m x`` it. Each operation on that path rounds once, ``fl(a op b) = (a op b)(1 + delta)`` with
``|delta| <= u`` (``u = eps / 2``), and the run's dtype rounds its inputs once on the way in (the sources' positions,
directions and coefficients, the point, the moment, the background). To first order an element ``R`` of the rows is
off by ``sum_k (dR/d delta_k) delta_k``, so it is within

    u sum_k |dR/d delta_k|

of the exact value: the bound computed here, each ``dR/d delta_k`` a forward-mode derivative of the path in float64
with every rounding written as a ``(1 + delta_k)``. It is the worst case of the linearised rounding, not a fit. Fed
the roundings the float32 path actually made (read from a bit-exact emulation of it), the same perturbed path
reproduces the measured float32 error of every element at 72 points of Navion_2's region to 1e-6 of itself, and its
linearisation to 0.2 %: the path holds every rounding the model makes, and the dropped second-order terms are below
1e-3 of the bound.

What it sees that ``eps / c`` does not: each rounding weighted by how far the element moves with it, summed over the
sources in absolute value (they round independently, so their errors do not cancel where their values do). Each
source's rounding moves an element by the derivative of that source's term along it, not by the element's size: where
a coil's opposed sources nearly cancel in one element (coil 0's ``G_zz`` near ``(0.045, 0.078, 0.043)`` m is 1/37 of
its ``G_zy``), the element is off by the size of what its sources' terms do under rounding, and a force row ``m_i
G_ij`` carries that into ``F_j`` on whichever axes the moment has. Of the bound, about 40 % is the distance (the
sum of squares, its root and the powers, raised to the third to seventh power), 18 % the inputs' cast, 15 to 23 % the
per-source terms, 10 % the offset ``x - p``, 6 to 9 % ``v . o``, 1 to 4 % the reciprocals, 2 % the sum over sources.

The path, as XLA runs it on CPU (eager and jitted): the offset ``x - p``; the distance the square root of a three-term
sum of squares; ``along = v . o`` a three-term dot product of fused multiply-adds (a fused operation rounds once where
the bound counts two, so the bound holds); powers of the distance by repeated products (``d^5 = d (d^2)^2``, ``d^3 =
d^2 d``); a quotient by a broadcast distance power rewritten as a reciprocal and a product (two roundings), ``3 / d^5``
one quotient; the gradient's four terms added left to right, then scaled; three sources summed per coil; the force a
three-term dot product; each torque component a difference of two products. A three-term sum's first partial is
bounded by its largest pair, so the bound holds whichever order XLA adds them in.

The host's float64 rows (``Calibration.unit_fields``) are computed by the same operations, or fewer roundings of them,
from exact inputs, so the bound of a comparison is ``u_run`` times the inputs' sum plus ``u_run + u_64`` times the
arithmetic's: in float64 the run's inputs are exact. The placement is the array's default, at the origin unturned
(an exact identity on the path); a turned placement adds the rotation's roundings, which this does not count."""

from __future__ import annotations

from functools import cache
from typing import Any

import jax
import numpy as np

from mime.fields._coil_array import calibration_of
from mime.nodes.actuation import Calibration

jnp: Any = jax.numpy

#: The roundings of the run's inputs (the cast to its dtype): exact in float64.
INPUTS = frozenset({"point", "positions", "directions", "coefficients", "moment", "background"})


def _sum3(terms: Any, partial: Any, total: Any) -> Any:
    """Three terms (the first axis) summed with two roundings: the first partial, whichever pair it is (bounded by the
    largest pair's sum), and the total."""
    pairs = jnp.stack([terms[0] + terms[1], terms[0] + terms[2], terms[1] + terms[2]])
    exact = jnp.sum(terms, axis=0)
    return exact + partial * jnp.max(jnp.abs(pairs), axis=0) + total * exact


def _dot3(a: Any, b: Any, products: Any, partial: Any, total: Any) -> Any:
    """``sum_i a_i b_i`` over the first axis, each product rounded, then :func:`_sum3`."""
    return _sum3(a * b * (1 + products), partial, total)


def _source(x: Any, p: Any, direction: Any, beta: Any, d: dict[str, Any]) -> tuple[Any, Any]:
    """One source's field and gradient per ampere (``source_field``), every rounding a ``(1 + d[...])``."""
    p = p * (1 + d["positions"])
    direction = direction * (1 + d["directions"])
    beta = beta * (1 + d["coefficients"])
    v = beta * direction * (1 + d["vectors"])
    o = (x - p) * (1 + d["offset"])
    squared = _dot3(o, o, d["squares"], d["squares_partial"], d["squares_total"])
    distance = jnp.sqrt(squared) * (1 + d["root"])
    along = _dot3(v, o, d["along_products"], d["along_partial"], d["along_total"])
    d2 = distance * distance * (1 + d["d2"])
    d5 = distance * (d2 * d2 * (1 + d["d4"])) * (1 + d["d5"])
    d3 = d2 * distance * (1 + d["d3"])
    over5, over3, over2 = (1 / d5) * (1 + d["over5"]), (1 / d3) * (1 + d["over3"]), (1 / d2) * (1 + d["over2"])
    three = 3 / d5 * (1 + d["three_over5"])
    first = 3 * along * (1 + d["three_along"]) * o * (1 + d["first"]) * over5 * (1 + d["first_scaled"])
    second = v * over3 * (1 + d["second"])
    field = (first - second) * (1 + d["field"])
    eye = jnp.eye(3)
    sums = (along * eye + jnp.outer(v, o) * (1 + d["vo"])) * (1 + eye * d["diagonal"])
    sums = (sums + jnp.outer(o, v) * (1 + d["ov"])) * (1 + d["sums"])
    last = 5 * along * (1 + d["five_along"]) * (jnp.outer(o, o) * (1 + d["oo"])) * (1 + d["last"])
    last = last * over2 * (1 + d["last_scaled"])
    gradient = three * ((sums - last) * (1 + d["difference"])) * (1 + d["gradient"])
    return field, gradient


_SCALARS = ("coefficients", "squares_partial", "squares_total", "root", "along_partial", "along_total", "d2", "d4",
            "d5", "d3", "over5", "over3", "over2", "three_over5", "three_along", "five_along")
_VECTORS = ("positions", "directions", "vectors", "offset", "squares", "along_products", "first", "first_scaled",
            "second", "field")
_MATRICES = ("vo", "ov", "sums", "oo", "last", "last_scaled", "difference", "gradient")


def _zeros(coils: int, sources: int) -> dict[str, Any]:
    per = {name: np.zeros((coils, sources)) for name in _SCALARS}
    per |= {name: np.zeros((coils, sources, 3)) for name in _VECTORS}
    per |= {name: np.zeros((coils, sources, 3, 3)) for name in _MATRICES}
    per["diagonal"] = np.zeros((coils, sources, 3, 3))
    return {"source": per, "point": np.zeros(3), "moment": np.zeros(3), "background": np.zeros(3),
            "field_partial": np.zeros((coils, 3)), "field_total": np.zeros((coils, 3)),
            "gradient_partial": np.zeros((coils, 3, 3)), "gradient_total": np.zeros((coils, 3, 3)),
            "force_products": np.zeros((3, 3, coils)), "force_partial": np.zeros((3, coils)),
            "force_total": np.zeros((3, coils)), "torque": np.zeros((3, 3, coils)), "torque_offset": np.zeros((3, 3))}


def _cross(m: Any, b: Any, d: Any) -> Any:
    """``m x b`` along the first axis, each component ``m_a b_b - m_b b_a`` with its products and difference
    rounded (``d`` the three roundings per component)."""
    index = ((1, 2), (2, 0), (0, 1))
    return jnp.stack([(m[a] * b[c] * (1 + d[0, j]) - m[c] * b[a] * (1 + d[1, j])) * (1 + d[2, j])
                      for j, (a, c) in enumerate(index)])


def _rows(held: Calibration, point: Any, moment: Any, background: Any, d: dict[str, Any]) -> dict[str, Any]:
    """The rows per ampere (``(3, K)``; the gradient ``(3, 3, K)``) and the offsets at zero command, with every
    rounding of the path a ``(1 + d[...])``."""
    x = jnp.asarray(point) * (1 + d["point"])
    per_source = jax.vmap(jax.vmap(_source, in_axes=(None, 0, 0, 0, 0)), in_axes=(None, 0, 0, 0, 0))
    fields, gradients = per_source(x, jnp.asarray(held.positions), jnp.asarray(held.directions),
                                   jnp.asarray(held.coefficients), d["source"])
    field = jax.vmap(_sum3)(fields, d["field_partial"], d["field_total"]).T
    gradient = jnp.moveaxis(jax.vmap(_sum3)(gradients, d["gradient_partial"], d["gradient_total"]), 0, -1)
    m = jnp.asarray(moment) * (1 + d["moment"])
    force = _sum3(m[:, None, None] * gradient * (1 + d["force_products"]), d["force_partial"], d["force_total"])
    torque = _cross(m, field, d["torque"])
    background = jnp.asarray(background) * (1 + d["background"])
    return {"flux_density": field, "gradient": gradient, "force": force, "torque": torque,
            "flux_density_offset": background, "torque_offset": _cross(m, background, d["torque_offset"])}


@cache
def _calibration(system: str) -> Calibration:
    from mime.data import asset

    return calibration_of(asset(system))


@cache
def _sums(system: str) -> Any:
    """The jitted sums of :func:`sensitivities` for ``system``'s calibration, traced under x64."""
    held = _calibration(system)
    zeros = _zeros(held.coils, int(held.positions.shape[1]))

    def sums(point: Any, moment: Any, background: Any) -> dict[str, tuple[Any, Any]]:
        jacobian = jax.jacfwd(lambda d: _rows(held, point, moment, background, d))(zeros)
        found: dict[str, tuple[Any, Any]] = {}
        for quantity, by in jacobian.items():
            size = jnp.ndim(by["point"]) - 1
            inputs, arithmetic = 0.0, 0.0
            for path, leaf in jax.tree_util.tree_flatten_with_path(by)[0]:
                total = jnp.sum(jnp.abs(leaf), axis=tuple(range(size, jnp.ndim(leaf))))
                if str(getattr(path[-1], "key", "")) in INPUTS:
                    inputs = inputs + total
                else:
                    arithmetic = arithmetic + total
            found[quantity] = (inputs, arithmetic)
        return found

    return jax.jit(sums)


def sensitivities(system: str, point: np.ndarray, moment: np.ndarray, background: np.ndarray
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per quantity (and its offset, ``<quantity>_offset``), the sums ``sum_k |dR/d delta_k|`` of each element over
    the inputs' roundings and over the arithmetic's: in units of the element, per unit ``u``."""
    with jax.enable_x64(True):
        found = _sums(system)(*(np.asarray(v, dtype=np.float64) for v in (point, moment, background)))
        return {quantity: (np.asarray(inputs), np.asarray(arithmetic))
                for quantity, (inputs, arithmetic) in found.items()}


def bound(system: str, point: np.ndarray, moment: np.ndarray, background: np.ndarray, dtype: str
          ) -> dict[str, np.ndarray]:
    """Per quantity, each element's first-order rounding bound between the model in ``dtype`` and the host in
    float64: ``u_run`` times the inputs' sum (their cast, none in float64) plus ``u_run + u_64`` times the
    arithmetic's."""
    unit, host = float(np.finfo(dtype).eps) / 2, float(np.finfo(np.float64).eps) / 2
    narrower = np.dtype(dtype) != np.float64
    return {quantity: (unit * inputs if narrower else 0.0) + (unit + host) * arithmetic
            for quantity, (inputs, arithmetic) in sensitivities(system, point, moment, background).items()}


__all__ = ["INPUTS", "bound", "sensitivities"]
