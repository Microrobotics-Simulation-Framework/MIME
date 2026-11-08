"""A phase held in turns as a pair of floating numbers, so that its rounding does not grow with the run.

A rotating field's phase is the sum over steps of ``f_n h`` turns, ``f_n`` the frequency written for the step and ``h``
the timestep. Held as one floating number, or as the product ``f_n t_n`` of a frequency and a time, the phase carries a
relative rounding ``eps`` of its own size, which grows with the turns the run has made: in float32 the field of a 1000 s
run at 40 Hz is off by 0.025 rad. Here the phase is held as a fraction of a turn ``(hi, lo)``, ``hi`` in ``[0, 1)`` and
``lo`` the part of it below ``hi``'s last place, and each step adds the step's turns, ``f_n h``, **as an exact product
of two floating numbers** (a pair ``(p, e)`` with ``p + e = f_n h`` to within ``eps^2`` of it) by an error-free sum:

* the frequency and the timestep (a Python float, held as a float and its remainder) are each split into a high and a
  low half by clearing the low half of the bits (an integer mask, which no floating-point contraction or reordering can
  touch), so that every partial product of the halves is exact in the run's precision;
* ``p`` is the rounded product and ``e`` Dekker's residual, built of the exact partial products;
* ``(hi, lo)`` takes ``p`` by a two-sum (Knuth), whose error term is exact, and ``e`` and the timestep's remainder
  (the part of ``h`` beyond one float32, below ``eps`` of the step's turns) join ``lo``.

What rounds, per step, is one addition into ``lo``, whose size is below a place of ``hi``: at most about
``3 eps^2 / 2`` of a turn, ``2e-15`` turns in float32 and ``1e-32`` in float64. Over ``N`` steps that is ``N`` times
as much in the worst case (``4e-9`` turns at 2e6 steps, 1000 s at 0.5 ms, a fraction of float32's own ``6e-8`` turns
resolution of one turn) and the sum of independent roundings in practice, against ``N eps`` of a step's turns for a
product rounded once a step, whatever the order it is summed in. The same arithmetic in float64 leaves the phase exact
to its last bit."""

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.core._error_free import fast_two_sum, two_sum

_lax: Any = cast(Any, jax).lax
_custom_jvp: Any = cast(Any, jax).custom_jvp

#: The mantissa bits (with the leading one) of the run's floating types.
_BITS = {np.dtype(np.float32): (24, np.uint32), np.dtype(np.float64): (53, np.uint64)}


def _half(dtype: Any) -> tuple[int, Any]:
    """How many low bits a float's split clears (the half of the mantissa, rounded up), and its integer type."""
    bits, uint = _BITS[np.dtype(dtype)]
    return (bits + 1) // 2, uint


def split(x: Array) -> tuple[Array, Array]:
    """``x`` as ``hi + lo`` exactly, ``hi`` with the high half of its mantissa (rounded to nearest) and ``lo`` the rest:
    a product of any two halves, of ``x`` or another float, is exact in ``x``'s precision."""
    clear, uint = _half(x.dtype)
    word = _lax.bitcast_convert_type(x, uint)
    kept = _lax.bitcast_convert_type((word + uint(1 << (clear - 1))) & ~uint((1 << clear) - 1), x.dtype)
    return kept, x - kept


def pieces(h: float, dtype: Any) -> tuple[float, float]:
    """The Python float ``h`` as ``h1 + h2`` with each a float of ``dtype``: ``h1`` rounded to it and ``h2`` the
    remainder, so their sum is ``h`` to within ``eps^2`` of it (exactly, in float64)."""
    first = float(np.asarray(h, dtype=dtype))
    return first, float(np.asarray(h - first, dtype=dtype))


def turns_of_step(frequency: Array, timestep: float) -> tuple[Array, Array]:
    """The turns ``frequency * timestep`` as ``(p, e)`` with ``p + e`` equal to it to within ``eps^2`` of it: ``p`` the
    product with the timestep rounded to the run's precision, ``e`` Dekker's residual of that product (built of exact
    partial products) plus the product with the timestep's remainder (below ``eps`` of ``p``, so its own rounding is
    below ``eps^2``)."""
    first, rest = pieces(timestep, frequency.dtype)
    big = jnp.asarray(first, dtype=frequency.dtype)
    p = frequency * big
    f_high, f_low = split(frequency)
    b_high, b_low = split(big)
    e = ((f_high * b_high - p) + f_high * b_low + f_low * b_high) + f_low * b_low
    return p, e + frequency * jnp.asarray(rest, dtype=frequency.dtype)


def accumulate(hi: Array, lo: Array, frequency: Array, timestep: float) -> tuple[Array, Array]:
    """The phase ``(hi, lo)`` (a fraction of a turn, ``hi`` in ``[0, 1)``) after a step of ``timestep`` at
    ``frequency``. Its derivative with respect to the frequency is the timestep, the phase's, and to ``hi`` and ``lo``
    one (the fraction's reduction is a whole turn, which moves neither the field nor a tangent)."""
    p, e = turns_of_step(frequency, timestep)
    s, r = two_sum(hi, p)
    carried, rest = fast_two_sum(s, (lo + r) + e)
    whole = jnp.floor(carried)
    return carried - whole, rest


def advance_phase(hi: Array, lo: Array, frequency: Array, timestep: float) -> tuple[Array, Array]:
    """:func:`accumulate`, with the derivative the phase has: with respect to the frequency the timestep, and with
    respect to ``hi`` and ``lo`` one (the reduction to a fraction drops whole turns, which moves neither the field nor a
    tangent). The arithmetic is exact and its tangent through it is not, so the tangent is given."""

    @_custom_jvp
    def step(hi: Array, lo: Array, frequency: Array) -> tuple[Array, Array]:
        return accumulate(hi, lo, frequency, timestep)

    def tangent(primals: tuple[Array, Array, Array],
                tangents: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        d_hi, d_lo, d_frequency = tangents
        return step(*primals), (d_hi + timestep * d_frequency, d_lo)

    step.defjvp(tangent)
    return cast(tuple[Array, Array], step(hi, lo, frequency))
