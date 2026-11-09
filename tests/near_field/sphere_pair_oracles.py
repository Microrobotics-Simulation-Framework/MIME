"""Two equal spheres in Stokes flow, the near field's analytic oracles [recalled formulas, each checked here against the
method of reflections at large gaps, and the squeeze against lubrication's a/(2h)]: Stimson and Jeffery's series for
equal velocities along the line of centres, Brenner's and Maude's for the squeeze (approaching each other, the
mid-plane a free surface), and Rotne and Prager's cross mobility. Written again from S3's spike (its ``oracles.py``),
in units of the radius and the viscosity: lambda = F / (6 pi mu a U).

Equal velocities squeeze nothing out of the gap, so that oracle cannot see the near field: a far-field check only."""

from __future__ import annotations

import math

import numpy as np


def stimson_jeffery(gap: float, terms: int = 4000) -> float:
    """lambda = F / (6 pi mu a U) on each of two equal spheres of radius a, surface gap ``gap`` (in a), moving with equal
    velocities U along their line of centres (Stimson & Jeffery 1926):

        lambda = 4/3 sinh(alpha) sum_n n(n+1)/((2n-1)(2n+3)) [1 - (4 sinh^2((n+1/2) alpha) - (2n+1)^2 sinh^2 alpha)
                                                               / (2 sinh((2n+1) alpha) + (2n+1) sinh(2 alpha))],

    cosh(alpha) = d/a, d each centre's distance from the plane of symmetry (half the centres' distance). Each bracket is
    evaluated with numerator and denominator divided by exp((2n+1) alpha), so that no term overflows and none cancels:
    with E = exp(-alpha) and p = E^(2n+1),
        bracket = [2p - 2p^2 + p ((2n+1) sinh(2 alpha) + (2n+1)^2 sinh^2 alpha)] / [1 - p^2 + p (2n+1) sinh(2 alpha)].
    """
    d = 1.0 + gap / 2.0
    alpha = math.acosh(d)
    sh2a = math.sinh(2.0 * alpha)
    sha2 = math.sinh(alpha) ** 2
    n = np.arange(1, terms + 1, dtype=np.float64)
    k = 2.0 * n + 1.0
    p = np.exp(-k * alpha)
    numerator = 2.0 * p - 2.0 * p * p + p * (k * sh2a + k * k * sha2)
    denominator = 1.0 - p * p + p * k * sh2a
    bracket = numerator / denominator
    weight = n * (n + 1.0) / ((2.0 * n - 1.0) * (2.0 * n + 3.0))
    series = weight * bracket
    # the tail must be negligible: the last term against the sum
    assert series[-1] < 1e-16 * np.sum(series) or alpha * k[-1] > 700, (gap, series[-1])
    return float(4.0 / 3.0 * math.sinh(alpha) * np.sum(series[::-1]))


def reflections(gap: float) -> float:
    """The method of reflections' lambda for the same motion: equal forces give equal velocities, and the mobility along
    the line of centres is x11 + x12 = 1 - 15/(4 s^4) + 3/(2 s) - 1/s^3 + O(s^-6) (s the centres' distance over a;
    Jeffrey & Onishi 1984's far-field terms [recalled]), so lambda = 1 / (that)."""
    s = 2.0 + gap
    return 1.0 / (1.0 + 1.5 / s - 1.0 / s ** 3 - 15.0 / (4.0 * s ** 4))


def rotne_prager(r: np.ndarray) -> np.ndarray:
    """The Rotne-Prager cross mobility between two spheres of radius 1 at viscosity 1, centres ``r`` apart (r >= 2):
    (1 / (8 pi |r|)) [(1 + 2/(3 |r|^2)) I + (1 - 2/|r|^2) r r^T / |r|^2]."""
    d = float(np.linalg.norm(r))
    e = r / d
    return ((1.0 + 2.0 / (3.0 * d * d)) * np.eye(3) + (1.0 - 2.0 / (d * d)) * np.outer(e, e)) / (8.0 * math.pi * d)


def approaching(gap: float, terms: int = 20000) -> float:
    """lambda = F / (6 pi mu a U) on each of two equal spheres approaching each other along their line of centres, each
    at speed U (the "squeeze" motion): by symmetry the mid-plane is a free surface, so this is a sphere approaching a free
    surface (Brenner 1961; Maude 1961) [recalled formula]:

        lambda = 4/3 sinh(alpha) sum_n n(n+1)/((2n-1)(2n+3)) [(4 cosh^2((n+1/2) alpha) + (2n+1)^2 sinh^2 alpha)
                                                               / (2 sinh((2n+1) alpha) - (2n+1) sinh(2 alpha)) - 1],

    with the same alpha as Stimson and Jeffery's. Evaluated divided through by exp((2n+1) alpha), p = exp(-(2n+1) alpha):
        bracket = (2p + 2p^2 + p k^2 sinh^2 alpha + p k sinh 2 alpha) / (1 - p^2 - p k sinh 2 alpha),
    whose denominator cancels at small alpha (n = 1: O(alpha^3) out of O(alpha)), so it is taken in Python's 50-digit
    decimal arithmetic. Checked below against reflections and the lubrication limit lambda -> a / (2 h)."""
    from decimal import Decimal, getcontext

    getcontext().prec = 50
    d = Decimal(1) + Decimal(repr(gap)) / 2
    alpha = (d + (d * d - 1).sqrt()).ln()
    e = (-alpha).exp()
    sinh_a = (alpha.exp() - e) / 2
    sinh_2a = ((2 * alpha).exp() - e * e) / 2
    total = Decimal(0)
    for n in range(1, terms + 1):
        k = 2 * n + 1
        p = e ** k
        numerator = 2 * p + 2 * p * p + p * k * k * sinh_a * sinh_a + p * k * sinh_2a
        denominator = 1 - p * p - p * k * sinh_2a
        term = Decimal(n * (n + 1)) / Decimal((2 * n - 1) * (2 * n + 3)) * numerator / denominator
        total += term
        if term < Decimal(10) ** -40 * total:
            break
    return float(Decimal(4) / 3 * sinh_a * total)


def approaching_reflections(gap: float) -> float:
    """Reflections for the squeeze: 1 / (x11 - x12) = 1 / (1 - 3/(2 s) + 1/s^3 - 15/(4 s^4)) [recalled terms]."""
    s = 2.0 + gap
    return 1.0 / (1.0 - 1.5 / s + 1.0 / s ** 3 - 15.0 / (4.0 * s ** 4))
