"""The ledger's running sums do not drift with the steps (``MIME-ANO-149``): each sum of a signed series is a pair, and
the pair's value is the exact sum of the terms to the pair's own rounding.

**The gate.** A synthetic series of 1e5 terms, drawn in the dtype the run computes in, goes through
``mime.core.ledger.recorded`` one step at a time under ``lax.scan`` (the program a run's loop is), for all of the
ledger's paired sums at once: a law's paired work, its hosted terms, the scheme's two terms, the unattributed work, and
the storage's jumps between steps (summed twice, as ``between`` and ``between_own``). The reference is the exact sum of
the same terms, in integer arithmetic. A pair's drift is judged on the summed magnitudes ``S`` of its terms, with no
absolute part, against the line its own rounding sets: one addition into ``low`` a step, below a place of the total,
at most ``eps^2 S / 2``, so ``N eps^2 S`` over ``N`` steps is twice the worst case. In float32 that is ``0.012`` units
of ``eps S`` at 1e5 steps, where a plain sum of a series like these reaches tens (the anomaly's ``15 eps``).

**The control** is the plain running sum of the same terms in the same dtype, the ledger as it was, which must fail the
same line on every seed of every series that grows (a bias of 0.25 or 1.0 of the terms' range), by a margin (the
message of a failure gives the ratio). On the series of no bias the sum stays small and a plain sum drifts under the
line on some seeds (``0.003`` to ``0.14`` units against ``0.012``): there the pair is held to the line and no control is
claimed. The seeds are fixed, so the control's detection rate is 8 of 8 by construction, on this dtype's arithmetic.

**The field alone** is the sum rounded to the run's dtype, for a reader that reads only the field (the account's
per-law lines): half a unit of its last place, and the pair's drift.

**The balance** is a difference of the sums, judged in the run's dtype, so its own arithmetic rounds at the scale of the
sums it subtracts (a few eps of their finals) however they were accumulated: the pair removes the drift of the sums and
cannot remove that. It is held to that line, on a series whose exact balance is zero. This test does not separate
the pair from a plain sum on every seed (the drift and the balance's own rounding are of one size), so the gate and the
control above are the evidence for the fix, and this one says the balance reads the pair's low part."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ledger_cases import DRAG_LAW, in_dtype

from mime.core import ledger as books
from mime.core._error_free import add_to_pair

STEPS = 100_000
SEEDS = tuple(range(8))
DTYPES = ("float32", "float64")
#: The terms are uniform in (-1, 1) about a bias: none (the sum stays small), a quarter of the range (mixed signs, the
#: sum grows) and the whole range (every term positive, the sum grows fastest).
SMALL, MIXED, POSITIVE = 0.0, 0.25, 1.0
GROWING = (MIXED, POSITIVE)
HELD = (DRAG_LAW,)
SCHEME_TERMS = 2
#: The sums the ledger keeps as pairs, and the series each is checked against, by the draw's name.
SUMS = ("work", "hosted", "terms", "unattributed", "between", "between_own")


def eps_of(name: str) -> float:
    return float(np.finfo(np.dtype(name)).eps)


def line(name: str) -> float:
    """``N eps^2 S`` in units of ``eps S``: the pair's rounding over the run, twice its worst case."""
    return STEPS * eps_of(name)


def exact(values: np.ndarray) -> Fraction:
    """The exact sum of ``values`` (floats), by integer mantissas grouped by exponent."""
    mantissa, exponent = np.frexp(np.asarray(values, dtype=np.float64).reshape(-1))
    whole, power = np.ldexp(mantissa, 53).astype(np.int64), exponent - 53
    total, least = 0, int(power.min())
    for each in np.unique(power):
        total += int(whole[power == each].astype(object).sum()) << int(each - least)
    return Fraction(total) * Fraction(2) ** least


def draws(name: str, seed: int, bias: float) -> dict[str, np.ndarray]:
    """The series of a run of ``STEPS`` steps: the storage at each step's start, and the step's terms."""
    rng = np.random.default_rng(1000 * seed + int(100 * bias))
    dtype = np.dtype(name)

    def draw(*shape: int) -> np.ndarray:
        return (rng.uniform(-1.0, 1.0, (STEPS, *shape)) + bias).astype(dtype)

    return {"work": draw(1), "hosted": draw(1, 2), "terms": draw(SCHEME_TERMS), "unattributed": draw(),
            "storage": draw()}


def series_of(drawn: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What each sum adds, step by step: the draws, and the jump between steps, the storage at a step's start less the
    one the step before ended at (rounded once, to the dtype, as the ledger's subtraction does)."""
    storage = drawn["storage"]
    jumps = storage - np.concatenate([np.zeros(1, dtype=storage.dtype), storage[:-1]])
    return {**{field: drawn[field] for field in ("work", "hosted", "terms", "unattributed")},
            "between": jumps, "between_own": jumps}


def run_ledger(name: str, drawn: dict[str, np.ndarray]) -> dict[str, Any]:
    """``recorded`` taken ``STEPS`` times under a scan, one law metered and the scheme's two terms."""
    zero = jnp.zeros((), dtype=jnp.dtype(name))
    start = books.initial_records(HELD, SCHEME_TERMS, zero)

    def step(state: Any, step_draws: Any) -> Any:
        work, hosted, terms, unattributed, storage = step_draws
        scale = jnp.sum(jnp.abs(work)) + jnp.abs(unattributed) + 1.0
        account = books.StepAccount(work, hosted, unattributed, terms, storage, storage, scale, {})
        return books.recorded(state, account, HELD), None

    def run(xs: Any) -> Any:
        return jax.lax.scan(step, start, xs)[0]

    order = ("work", "hosted", "terms", "unattributed", "storage")
    return jax.jit(run)(tuple(jnp.asarray(drawn[field]) for field in order))


def run_plain(name: str, series: dict[str, np.ndarray]) -> dict[str, Any]:
    """The same sums as the ledger kept them before: one float of the run's dtype each, added to a step at a time."""

    def run(xs: Any) -> Any:
        def step(totals: Any, terms: Any) -> Any:
            return tuple(total + term for total, term in zip(totals, terms, strict=True)), None

        zeros = tuple(jnp.zeros(x.shape[1:], dtype=x.dtype) for x in xs)
        return jax.lax.scan(step, zeros, xs)[0]

    found = jax.jit(run)(tuple(jnp.asarray(series[field]) for field in SUMS))
    return dict(zip(SUMS, found, strict=True))


def drifts(name: str, series: dict[str, np.ndarray], held: dict[str, Any], low: dict[str, Any] | None) -> list[
        tuple[str, tuple[int, ...], float]]:
    """Each sum's drift from the exact sum, in units of ``eps S`` on its terms' summed magnitudes: the pair's value
    (``held`` and ``low``) where there is a ``low``, the plain sum's otherwise."""
    found = []
    for field in SUMS:
        for index in np.ndindex(*series[field].shape[1:]):
            terms = series[field][(slice(None), *index)]
            value = Fraction(float(np.asarray(held[field])[index]))
            if low is not None:
                value += Fraction(float(np.asarray(low[field])[index]))
            drift = abs(value - exact(terms)) / exact(np.abs(terms))
            found.append((field, index, float(drift) / eps_of(name)))
    return found


@pytest.mark.parametrize("name", DTYPES)
@pytest.mark.parametrize("bias", (SMALL, MIXED, POSITIVE))
def test_each_paired_sum_is_the_exact_sum_of_its_terms_to_the_pairs_rounding(name: str, bias: float):
    r = books.RECORDS
    with in_dtype(name):
        for seed in SEEDS:
            drawn = draws(name, seed, bias)
            series = series_of(drawn)
            records = run_ledger(name, drawn)
            held = {field: records[f"{r}{field}"] for field in SUMS}
            low = {field: records[f"{r}{field}{books.LOW}"] for field in SUMS}
            for field, index, found in drifts(name, series, held, low):
                assert found <= line(name), (name, bias, seed, field, index, found, line(name))
                terms = series[field][(slice(None), *index)]
                total, magnitudes = exact(terms), exact(np.abs(terms))
                alone = Fraction(float(np.asarray(held[field])[index]))
                assert abs(alone - total) <= Fraction(eps_of(name) / 2) * abs(total) + Fraction(
                    line(name) * eps_of(name)) * magnitudes, (name, bias, seed, field, index, "the field alone")


@pytest.mark.parametrize("name", DTYPES)
@pytest.mark.parametrize("bias", GROWING)
def test_control_the_plain_sum_of_the_same_terms_fails_the_same_line_on_every_seed(name: str, bias: float):
    with in_dtype(name):
        for seed in SEEDS:
            series = series_of(draws(name, seed, bias))
            plain = run_plain(name, series)
            found = drifts(name, series, plain, None)
            worst = max(found, key=lambda each: each[2])
            ratio = worst[2] / line(name)
            assert worst[2] > line(name), (name, bias, seed, worst, "the control missed")
            assert ratio > 10.0, (name, bias, seed, worst, f"only {ratio:.3g} of the line")


@pytest.mark.parametrize("name", DTYPES)
def test_a_sum_of_the_ledger_is_a_pair_beside_every_record_and_the_step_returns_the_same_fields(name: str):
    with in_dtype(name):
        zero = jnp.zeros((), dtype=jnp.dtype(name))
        start = books.initial_records(HELD, SCHEME_TERMS, zero)
        for field in books.PAIRED:
            twin = f"{books.RECORDS}{field}{books.LOW}"
            assert twin in start and start[twin].shape == start[f"{books.RECORDS}{field}"].shape
            assert start[twin].dtype == start[f"{books.RECORDS}{field}"].dtype == jnp.dtype(name)
        account = books.StepAccount(jnp.ones(1), jnp.ones((1, 2)), zero, jnp.ones(SCHEME_TERMS), zero, zero, zero + 1.0,
                                    {})
        assert set(books.recorded(start, account, HELD)) == set(start)


@pytest.mark.parametrize("name", DTYPES)
def test_the_balance_of_a_series_that_cancels_exactly_is_zero_to_the_rounding_of_the_sums_it_subtracts(name: str):
    """The scheme's two terms are the paired work split in two, exactly (terms of nine bits, whose pairwise sum is
    exact in either dtype), so the books close exactly: the balance's own rounding is what is left, a few eps of the
    finals."""
    r = books.RECORDS
    with in_dtype(name):
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            dtype = np.dtype(name)
            first = (rng.integers(-128, 384, (STEPS, 1)) / 256.0).astype(dtype)
            second = (rng.integers(-128, 384, (STEPS, 1)) / 256.0).astype(dtype)
            drawn = {"work": first + second, "hosted": np.zeros((STEPS, 1, 2), dtype=dtype),
                     "terms": np.concatenate([first, second], axis=1), "unattributed": np.zeros(STEPS, dtype=dtype),
                     "storage": np.zeros(STEPS, dtype=dtype)}
            records = run_ledger(name, drawn)
            finals = sum(abs(float(np.asarray(records[f"{r}{field}"]).sum())) for field in ("work", "terms"))
            assert abs(float(books.balance(records))) <= 4 * eps_of(name) * finals, (name, seed)


@pytest.mark.parametrize("name", DTYPES)
def test_the_derivative_of_a_paired_sum_with_respect_to_its_terms_is_the_plain_sums(name: str):
    """A gain on the terms: the field's derivative is the sum of the terms, as a plain sum's is."""
    with in_dtype(name):
        terms = jnp.asarray(np.random.default_rng(0).uniform(-1.0, 1.0, 200).astype(np.dtype(name)))
        zero = jnp.zeros((), dtype=jnp.dtype(name))

        def total(gain: Any) -> Any:
            def step(carry: Any, term: Any) -> Any:
                return add_to_pair(*carry, gain * term), None

            return jax.lax.scan(step, (zero, zero), terms)[0][0]

        found = float(jax.grad(total)(jnp.ones((), dtype=jnp.dtype(name))))
        values = np.asarray(terms)
        assert abs(found - float(exact(values))) <= 8 * eps_of(name) * float(exact(np.abs(values)))
