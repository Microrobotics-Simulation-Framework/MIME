"""Harnesses a lab calls from its own pytest files, built on ``maddening.testing`` rather than beside it.

Users add their own tests at three levels:

1. **Their own entries.** An interaction, dynamics, field provider, observation or controller a lab registers
   claims its family's suite by calling that suite's harness from the lab's pytest file; passing in the lab's own
   CI is the claim. Every node an entry builds also passes MADDENING's ``verify_node``.
2. **Their own benchmarks.** MADDENING's ``@verification_benchmark``, under the lab's own id prefix; MIME keeps no
   benchmark registry of its own.
3. **Their own experiments.** The round trip (build, save, load, rebuild), golden outputs and the gradient check,
   each as a harness call.

Here now: the graph-invariant helpers (a built graph's canonical text, and order invariance) and, in
``mime.testing.toys``, toy entries with no physics for tests of the build itself. The family suites arrive with the
first families. Importing this module needs the ``verify`` extra.
"""

try:
    from maddening.testing import verification as _maddening_verification  # noqa: F401
except ImportError as error:
    raise ImportError("mime.testing needs the verify extra: pip install 'mime-engine[verify]'") from error

from mime.testing._graph import (
    Declare,
    OrderReference,
    assert_order_invariant,
    declaration_orders,
    declare_in_order,
    graph_snapshot,
    history_differences,
    order_invariance_failures,
    order_reference,
    run_history,
)

__all__ = [
    "Declare",
    "OrderReference",
    "assert_order_invariant",
    "declaration_orders",
    "declare_in_order",
    "graph_snapshot",
    "history_differences",
    "order_invariance_failures",
    "order_reference",
    "run_history",
]
