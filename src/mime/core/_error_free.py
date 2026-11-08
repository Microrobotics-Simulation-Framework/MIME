"""Error-free sums, the one home of the two a run keeps a quantity to more than its precision by.

**The two-sum** (Knuth, :func:`two_sum`; Dekker's :func:`fast_two_sum` where the larger operand is known) returns
``a + b`` as ``s + r`` *exactly*: ``s`` the rounded sum and ``r`` the rounding it made. Neither contracts under a
compiler's reassociation or fusion, since they contain no product.

**A pair** ``(total, low)`` holds a sum to about twice the run's precision, ``total`` the rounded value and ``low`` the
part below its last place. :func:`add_to_pair` adds one term, and the pair's value is read as ``total + low``: **in the
run's own dtype that sum rounds to ``total``** (the pair is renormalised, so ``|low| <= eps |total| / 2``), so read the
two apart where the difference matters (a balance that subtracts large terms), or in float64 on the host. What rounds,
per added term, is one addition into ``low``, below a place of ``total``: at most ``eps^2 / 2`` of the larger of the
total and the term, against ``eps`` of the total's own size for a plain sum, whose rounding grows as a mixed-sign
series is summed (about ``15 eps`` of the summed magnitudes by ``1e5`` float32 steps, ``MIME-ANO-149``). The derivative
of the pair's total with respect to the term is one, the plain sum's: the error terms' derivatives are rounding.

:mod:`mime.nodes.actuation._turns` holds a rotating field's phase by the same two-sums."""

from __future__ import annotations

from jax import Array


def two_sum(a: Array, b: Array) -> tuple[Array, Array]:
    """``a + b`` as ``s + r`` exactly (Knuth), for any ``a`` and ``b``."""
    s = a + b
    b_part = s - a
    return s, (a - (s - b_part)) + (b - b_part)


def fast_two_sum(a: Array, b: Array) -> tuple[Array, Array]:
    """``a + b`` as ``s + r`` exactly (Dekker), for ``|a| >= |b|``."""
    s = a + b
    return s, b - (s - a)


def add_to_pair(total: Array, low: Array, term: Array) -> tuple[Array, Array]:
    """The pair ``(total, low)`` after adding ``term``: the term joins ``total`` by an error-free sum, its rounding and
    the old ``low`` join by a plain one (below a place of ``total``), and the result is renormalised by a second
    error-free sum, so that ``total`` is the rounded value of the pair (the two-sum, whatever the operands' sizes:
    ``total`` may be near zero while ``low`` is not)."""
    s, rounded = two_sum(total, term)
    return two_sum(s, low + rounded)
