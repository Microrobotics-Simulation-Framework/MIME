"""Harnesses a lab calls from its own pytest files, built on ``maddening.testing`` rather than beside it.

Users add their own tests at three levels:

1. **Their own entries.** An interaction, dynamics, field provider, observation or controller a lab registers
   claims its family's suite by calling that suite's harness from the lab's pytest file; passing in the lab's own
   CI is the claim. Every node an entry builds also passes MADDENING's ``verify_node``.
2. **Their own benchmarks.** MADDENING's ``@verification_benchmark``, under the lab's own id prefix; MIME keeps no
   benchmark registry of its own.
3. **Their own experiments.** The round trip (build, save, load, rebuild), golden outputs and the gradient check,
   each as a harness call.

Here now: the family suites' harness (``assert_conforms(law, family=...)`` and the case kinds the families use:
sign, frame, units, material reach, oracle, gradients, swap, equal and opposite, and nodes); MADDENING's node
verification for every node an entry builds (``assert_nodes_verified``: ``verify_node``, and ``verify_node_order``
where a node declares an order), and the comparison of what an entry builds with its ``builds``
(``builds_failures``); the graph-invariant helpers (a built graph's canonical text, and order invariance); the saved
form's round trip through a fresh process and the list of installed entries a registry walk covers; the check that
every parameter leaf reaches the compiled step; and, in ``mime.testing.toys``, toy entries with no physics for tests
of the build itself, with the toy family as the family suites' control. MIME's own families arrive with their
entries. Importing this module needs the ``verify`` extra.
"""

import importlib

from mime._extras import VERIFY_EXTRA_MISSING

try:
    importlib.import_module("maddening.testing.verification")
except ImportError as error:
    raise ImportError(VERIFY_EXTRA_MISSING) from error

from mime.testing._families import (
    At,
    CaseKind,
    Family,
    Oracle,
    Probe,
    Response,
    SignRule,
    assert_conforms,
    conformance,
    family_named,
    opposes,
    probe_law,
    redeclared,
    register_family,
    registered_families,
    turned,
)
from mime.testing._mime_families import (
    CONTACT,
    GRAVITY,
    HYDRODYNAMIC,
    MAGNETIC,
    contact_rig,
    gravity_rig,
    hydrodynamic_rig,
    magnetic_rig,
    positive_semidefinite,
    unit_quaternion,
)
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
from mime.testing._nodes import (
    EntryKey,
    NodeCase,
    OrderLadder,
    assert_nodes_verified,
    builds_failures,
    computing_dtype,
    entry_key,
    node_failures,
    node_results,
    nodes_by_entry,
    nodes_by_rung,
)
from mime.testing._saved import (
    FreshLoad,
    covered,
    first_steps,
    installed_entries,
    load_in_fresh_process,
    param_reach_failures,
    refused_round_trip_failures,
    round_trip_failures,
)
from mime.testing._kind_conformance import KIND_ROUNDING_ULPS, NORMAL_ROUNDING_ULPS, kind_problems, sampling_kinds
from mime.testing._shape_suite import (
    DIFFERENCE_FLOOR_ULPS,
    PLANTED_TANGENT,
    PROBE_CLASSES,
    SHAPE_CASES,
    SHAPE_TOLERANCE,
    ShapeCase,
    ShapeCheck,
    ShapeProbe,
    SwitchSet,
    check_shape,
    register_shape_case,
    shape_probes,
    switch_distances,
    uncased_shapes,
)
from mime.testing._switches import (
    FLOOR_ULPS,
    KINK_LIMIT,
    Classification,
    GuardCheck,
    Measure,
    SwitchRow,
    classify,
    fallback_problems,
)

__all__ = [
    "CONTACT",
    "DIFFERENCE_FLOOR_ULPS",
    "FLOOR_ULPS",
    "GRAVITY",
    "HYDRODYNAMIC",
    "KIND_ROUNDING_ULPS",
    "KINK_LIMIT",
    "MAGNETIC",
    "NORMAL_ROUNDING_ULPS",
    "PLANTED_TANGENT",
    "PROBE_CLASSES",
    "SHAPE_CASES",
    "SHAPE_TOLERANCE",
    "At",
    "CaseKind",
    "Classification",
    "Declare",
    "EntryKey",
    "Family",
    "FreshLoad",
    "GuardCheck",
    "Measure",
    "NodeCase",
    "Oracle",
    "OrderLadder",
    "OrderReference",
    "Probe",
    "Response",
    "ShapeCase",
    "ShapeCheck",
    "ShapeProbe",
    "SignRule",
    "SwitchRow",
    "SwitchSet",
    "assert_conforms",
    "assert_nodes_verified",
    "assert_order_invariant",
    "builds_failures",
    "check_shape",
    "classify",
    "computing_dtype",
    "conformance",
    "contact_rig",
    "covered",
    "declaration_orders",
    "declare_in_order",
    "entry_key",
    "fallback_problems",
    "family_named",
    "first_steps",
    "graph_snapshot",
    "gravity_rig",
    "history_differences",
    "hydrodynamic_rig",
    "installed_entries",
    "kind_problems",
    "load_in_fresh_process",
    "magnetic_rig",
    "node_failures",
    "node_results",
    "nodes_by_entry",
    "nodes_by_rung",
    "opposes",
    "order_invariance_failures",
    "order_reference",
    "param_reach_failures",
    "positive_semidefinite",
    "probe_law",
    "redeclared",
    "refused_round_trip_failures",
    "register_family",
    "register_shape_case",
    "registered_families",
    "round_trip_failures",
    "run_history",
    "sampling_kinds",
    "shape_probes",
    "switch_distances",
    "turned",
    "uncased_shapes",
    "unit_quaternion",
]
