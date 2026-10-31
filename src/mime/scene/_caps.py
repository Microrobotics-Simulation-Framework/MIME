"""A flat cap of a star-shaped section, ``r <= rho(theta)``, sampled in rings: a sample at its centre and rings out to
its rim, each ring's count growing with its radius, so that no ring crowds the centre (MIME-ANO-119: a polar grid puts
as many samples on its innermost ring, a hundredth of the way out, as on its rim).

The rings are homothetic copies of the rim, the ``i``-th at the fraction ``s_i`` of the way out at every angle, and a
ring's samples are equally spaced in the section's angle, so that a sample at fraction ``s`` and angle ``theta`` lies
at ``s rho(theta)`` and carries ``W_i rho(theta)^2 2 pi / m_i`` of the cap's area (``s rho^2 ds dtheta`` is the area
element). Round a ring of ``m_i`` samples the trapezoid rule integrates exactly a trigonometric polynomial of degree
below ``m_i``; a section of ``N`` starts makes the area, first and second moments trigonometric polynomials of degree
``2N``, ``3N + 1`` and ``4N + 2`` round it, so every ring carries more than ``4N + 2`` samples. A section of ``N``
starts is itself after a turn of ``2 pi / N``, and so are the samples: each ring's count is a multiple of ``N``, and a
cap carries one more than a multiple of ``N`` (its centre and whole rings). Along the fraction, ``W_i`` (the centre's
``W_0``) integrate ``s g(s) ds`` exactly for ``g`` of degree two, which the second moments need: the weights nearest
the rings' own shares of the cap (each ring the share between it and its neighbours, in the chi-square sense) that do,
solved in exact rational arithmetic and rounded once, so that every weight is above zero and rounds at half a unit in
its last place.

The rings are equally spaced in the fraction, the first ``a`` gaps out (``a`` the radius, in gaps, at which a ring
carries its fewest samples a gap apart round the crest) and the last ``RIM_GAPS`` gaps in from the rim, where a screw's
lateral surface, turned along the axis, leans over its cap; and a ring of ``a + i`` gaps' radius carries about
``2 pi (a + i)`` samples, as many round it as gaps out, so that at the crest the samples are as far apart round a ring
as across rings. A cap of ``n`` rings then holds about ``1 + 2 pi (n a + n (n - 1) / 2)`` samples, which sets ``n``
from the cap's count."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

#: How far in from the rim the outer ring lies, in ring gaps: the lateral surface of a screw leans over its cap where
#: the section's flank is turned along the axis, and its samples near the end come within a fraction of a gap of a ring
#: on the rim. Lowered for a cap of few rings (to a quarter of the outer ring's gaps out), where the fraction outside it
#: would leave the second moment no rule with weights above zero.
RIM_GAPS = Fraction(3, 2)
#: Pi as the rings' rational layout takes it (22/7): the layout's fractions are rational, so that their weights are
#: exact; its rings' counts are rounded to whole samples anyway.
_PI = Fraction(22, 7)


def ring_floor(starts: int) -> int:
    """The fewest samples a ring carries: the least multiple of ``N`` above ``4N + 2``, the second moments' degree round
    it (12, 15 and 20 for two, three and four starts)."""
    return starts * -(-(4 * starts + 3) // starts)


def symmetric_total(total: int, starts: int) -> int:
    """The most samples at most ``total`` that a cap of ``starts`` starts can carry and keep its section's symmetry:
    one more than a multiple of ``starts`` (its centre and whole rings)."""
    return total - (total - 1) % starts


@dataclass(frozen=True)
class CapRings:
    """One cap's samples, the centre first, then each ring's, innermost first: their fraction of the way out to the
    rim, their angle in the section, and their share of the cap's area divided by ``rho(theta)^2`` (the centre's by
    the section's mean of ``rho^2``, ``R^2 (1 + eps^2 / 2)``, the integral of ``rho^2`` round it over ``2 pi``)."""

    fractions: tuple[Fraction, ...]
    counts: tuple[int, ...]
    fraction: np.ndarray
    angle: np.ndarray
    share: np.ndarray


def _rings(total: int, starts: int) -> tuple[tuple[Fraction, ...], tuple[int, ...]]:
    """The rings' fractions and counts for a cap of ``total`` samples, one of them its centre, each ring's count a
    multiple of ``starts``."""
    if total < 2 or (total - 1) % starts:
        raise ValueError(f"a cap of {starts} starts carries its centre and whole rings, one more than a multiple of "
                         f"{starts} samples and at least {starts + 1}, not {total}")
    orbits, floor = (total - 1) // starts, ring_floor(starts) // starts
    if orbits < floor:
        return (Fraction(3, 4),), (total - 1,)
    first = Fraction(ring_floor(starts)) / (2 * _PI)
    rings = round((-(2 * float(first) - 1) + math.sqrt((2 * float(first) - 1) ** 2 + 4 * (total - 1) / math.pi)) / 2)
    rings = max(1, min(rings, orbits // floor))
    while True:
        if rings == 1:
            fractions: tuple[Fraction, ...] = (Fraction(3, 4),)
        else:
            rim = min(RIM_GAPS, (first + rings - 1) / 4)
            fractions = tuple((first + i) / (first + rings - 1 + rim) for i in range(rings))
        radii = [first + i for i in range(rings)]
        wanted = [float(r / sum(radii)) * orbits for r in radii]
        counts = [max(floor, math.floor(w)) for w in wanted]
        if sum(counts) <= orbits:
            break
        rings -= 1
    while sum(counts) < orbits:
        short = max(range(rings), key=lambda i: (wanted[i] - counts[i], -i))
        counts[short] += 1
    return fractions, tuple(starts * c for c in counts)


def _solved(matrix: list[list[Fraction]], right: list[Fraction]) -> list[Fraction]:
    """``matrix x = right`` in exact arithmetic, by Gauss-Jordan elimination."""
    size = len(matrix)
    rows = [row[:] + [right[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = next(i for i in range(column, size) if rows[i][column] != 0)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for i in range(size):
            if i != column and rows[i][column] != 0:
                factor = rows[i][column] / rows[column][column]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[column], strict=True)]
    return [rows[i][size] / rows[i][i] for i in range(size)]


def radial_weights(nodes: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Weights at ``nodes`` (fractions of the way out, the first the centre's 0) that integrate ``s g(s) ds`` over
    ``[0, 1]`` exactly for ``g`` of degree two (of degree one at two nodes, three at one): of the weights that do, the
    nearest to each node's share, ``(b^2 - a^2) / 2`` between the midpoints ``a`` and ``b`` to its neighbours (the rim
    the last one's outer), in the chi-square sense."""
    degree = min(2, len(nodes) - 1)
    edges = [Fraction(0), *((a + b) / 2 for a, b in zip(nodes, nodes[1:])), Fraction(1)]
    shares = [(b * b - a * a) / 2 for a, b in zip(edges, edges[1:])]
    powers = [[Fraction(1) if q == 0 else s ** q for s in nodes] for q in range(degree + 1)]
    gram = [[sum((p[i] * shares[i] * r[i] for i in range(len(nodes))), Fraction(0)) for r in powers] for p in powers]
    missing = [Fraction(1, q + 2) - sum((powers[q][i] * shares[i] for i in range(len(nodes))), Fraction(0))
               for q in range(degree + 1)]
    multipliers = _solved(gram, missing)
    return tuple(shares[i] * (1 + sum((powers[q][i] * multipliers[q] for q in range(degree + 1)), Fraction(0)))
                 for i in range(len(nodes)))


@lru_cache(maxsize=64)
def cap_rings(total: int, starts: int) -> CapRings:
    """The cap's ``total`` samples (one its centre) in rings, for a section of ``starts`` starts."""
    fractions, counts = _rings(total, starts)
    weights = radial_weights((Fraction(0), *fractions))
    fraction = np.concatenate([[0.0], *(np.full(m, float(s)) for s, m in zip(fractions, counts, strict=True))])
    angle = np.concatenate([[0.0], *(2.0 * math.pi * (np.arange(m) + 0.5 * (i % 2)) / m for i, m in enumerate(counts))])
    rings = zip(weights[1:], counts, strict=True)
    share = np.concatenate([[float(weights[0]) * 2.0 * math.pi], *(np.full(m, float(w) * 2.0 * math.pi / m)
                                                                   for w, m in rings)])
    for array in (fraction, angle, share):
        array.flags.writeable = False
    return CapRings(fractions, counts, fraction, angle, share)


def any_closer(points: np.ndarray, distance: float) -> bool:
    """Whether two of ``points`` (``(n, 2)``) lie closer than ``distance``: each binned into a square of that side,
    every pair within ``distance`` lies in one square or two adjacent, so each point is compared with the points of its
    own square after it and of the four squares after it in the bins' order."""
    if len(points) < 2:
        return False
    cell = np.floor(points / distance).astype(np.int64)
    cell -= cell.min(axis=0)
    width = int(cell[:, 0].max()) + 3
    key = cell[:, 1] * width + cell[:, 0]
    order = np.argsort(key, kind="stable")
    keys, held = key[order], points[order]
    for offset in (0, 1, width - 1, width, width + 1):
        low = np.searchsorted(keys, keys + offset, side="left")
        high = np.searchsorted(keys, keys + offset, side="right")
        if offset == 0:
            low = np.arange(len(keys)) + 1
        for step in range(int(np.max(high - low, initial=0))):
            other = low + step
            within = other < high
            if np.any(np.linalg.norm(held[within] - held[other[within]], axis=1) < distance):
                return True
    return False


__all__ = ["RIM_GAPS", "CapRings", "any_closer", "cap_rings", "radial_weights", "ring_floor", "symmetric_total"]
