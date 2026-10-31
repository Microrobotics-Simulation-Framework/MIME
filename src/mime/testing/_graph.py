"""Graph-invariant helpers: the canonical text of a built experiment's graph, and order invariance (the order a
script declares things in must change neither the graph nor a single bit of a run)."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, override

import numpy as np
from hypothesis import strategies as st

from mime.experiment import BuiltExperiment, Experiment
from mime.units import Quantity


@dataclass(frozen=True)
class Declare:
    """One declaration call (``add_*``, ``attach`` or ``observe``) to apply to an experiment, with a label for
    reports. Bind by name rather than by handle, so the call can come before or after the things it names."""

    label: str
    apply: Callable[[Experiment], Any]

    @override
    def __repr__(self) -> str:
        return self.label


def declare_in_order(make: Callable[[], Experiment], declarations: Sequence[Declare],
                     order: Sequence[int] | None = None) -> Experiment:
    experiment = make()
    for index in range(len(declarations)) if order is None else order:
        declarations[index].apply(experiment)
    return experiment


def graph_snapshot(built: BuiltExperiment) -> str:
    """Nodes in the order added with class and rate; edges in the order added with the additive flag, mapping and
    transform; coupling groups with every setting; external inputs. Commit it as golden text."""
    return built.snapshot()


def run_history(built: BuiltExperiment, duration: Quantity) -> dict[str, dict[str, np.ndarray]]:
    history = built.run(duration, history=True).history
    assert history is not None
    return {node: dict(fields) for node, fields in history.items()}


def history_differences(reference: dict[str, dict[str, np.ndarray]],
                        other: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Every field that is not bitwise equal, with its largest difference and the first step it differs at."""
    found: list[str] = []
    for node in sorted(set(reference) | set(other)):
        fields_a, fields_b = reference.get(node), other.get(node)
        if fields_a is None or fields_b is None:
            found.append(f"{node}: only in {'the other run' if fields_a is None else 'the reference'}")
            continue
        for name in sorted(set(fields_a) | set(fields_b)):
            a, b = fields_a.get(name), fields_b.get(name)
            if a is None or b is None or a.shape != b.shape:
                found.append(f"{node}.{name}: missing or of another shape")
            elif not np.array_equal(a, b):
                step = int(np.argmax(np.any((a != b).reshape(a.shape[0], -1), axis=1)))
                largest = float(np.max(np.abs(a - b)))
                found.append(f"{node}.{name}: max |difference| {largest:.3e}, first at step {step}")
    return found


@dataclass(frozen=True)
class OrderReference:
    snapshot: str
    history: dict[str, dict[str, np.ndarray]]


def order_reference(make: Callable[[], Experiment], declarations: Sequence[Declare],
                    duration: Quantity) -> OrderReference:
    built = declare_in_order(make, declarations).build()
    return OrderReference(built.snapshot(), run_history(built, duration))


def order_invariance_failures(make: Callable[[], Experiment], declarations: Sequence[Declare],
                              order: Sequence[int], reference: OrderReference, duration: Quantity) -> list[str]:
    """How the experiment declared in ``order`` differs from ``reference``: its graph text, then every field of a
    run, bit for bit. Empty when it does not."""
    built = declare_in_order(make, declarations, order).build()
    snapshot = built.snapshot()
    if snapshot != reference.snapshot:
        return ["the graph differs:\n" + _first_difference(reference.snapshot, snapshot)]
    return history_differences(reference.history, run_history(built, duration))


def _first_difference(a: str, b: str) -> str:
    for line_a, line_b in itertools.zip_longest(a.splitlines(), b.splitlines(), fillvalue="(none)"):
        if line_a != line_b:
            return f"  reference: {line_a}\n  this order: {line_b}"
    return "  (no line differs)"


def assert_order_invariant(make: Callable[[], Experiment], declarations: Sequence[Declare], duration: Quantity,
                           orders: Iterable[Sequence[int]] | None = None) -> None:
    """Every order (every permutation when ``orders`` is ``None``) gives the declaration order's graph and run."""
    reference = order_reference(make, declarations, duration)
    for order in itertools.permutations(range(len(declarations))) if orders is None else orders:
        failures = order_invariance_failures(make, declarations, order, reference, duration)
        assert not failures, (f"declared as {[declarations[i] for i in order]}, the experiment differs:\n"
                              + "\n".join(failures))


def declaration_orders(declarations: Sequence[Declare]) -> st.SearchStrategy[list[int]]:
    """Hypothesis orders of ``declarations``, for a property test of order invariance."""
    return st.permutations(range(len(declarations)))
