"""Order invariance as a property: the full toy, declared in any order of its nine calls (binding by name, so an
``attach`` may come before the ``add_*`` of what it binds), builds the same graph text and a bitwise-identical run.

The control: with the canonical order skipped, Hypothesis must find an order that breaks the property.
"""

import pytest
from hypothesis import HealthCheck, find, given, settings

from mime.build import _testing
from mime.testing import (
    declaration_orders,
    declare_in_order,
    history_differences,
    order_invariance_failures,
    order_reference,
    run_history,
)
from mime.testing.toys import full_declarations, toy_experiment
from mime.units import s

DECLARATIONS = full_declarations()
DURATION = s(0.4)

# Hypothesis's default example count, kept: each example builds, compiles and runs a fresh graph.
PROPERTY = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def make():
    return toy_experiment("full_toy")


@pytest.fixture(scope="module")
def reference():
    return order_reference(make, DECLARATIONS, DURATION)


@PROPERTY
@given(order=declaration_orders(DECLARATIONS))
def test_any_order_of_the_calls_gives_the_same_graph_and_run(order, reference):
    assert order_invariance_failures(make, DECLARATIONS, order, reference, DURATION) == []


def test_without_the_canonical_order_hypothesis_finds_an_order_that_breaks_it():
    """Hypothesis searches for an order whose graph or run differs, and then for one whose run alone differs, so the
    control does not rest on the graph text."""
    with _testing.declaration_order():
        reference = order_reference(make, DECLARATIONS, DURATION)

        def breaks(order):
            return order_invariance_failures(make, DECLARATIONS, order, reference, DURATION)

        def changes_the_run(order):
            built = declare_in_order(make, DECLARATIONS, order).build()
            return history_differences(reference.history, run_history(built, DURATION))

        for found_by in (breaks, changes_the_run):
            counterexample = find(declaration_orders(DECLARATIONS),
                                  lambda order, found_by=found_by: bool(found_by(order)),
                                  settings=settings(PROPERTY, database=None))
            failures = found_by(counterexample)
            print(f"{found_by.__name__}: {[DECLARATIONS[i] for i in counterexample]}\n" + "\n".join(failures))
            assert failures
