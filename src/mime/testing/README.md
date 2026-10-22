# `mime.testing`

**Harnesses a lab calls from its own tests.** They are built on `maddening.testing`, not beside it. Install them with
`pip install "mime-engine[verify]"`.

## How users add their own tests

1. **Their own entries.** A lab's drag law can call something like
   `mime.testing.assert_conforms(MyDrag, family="hydrodynamic")` from its own pytest files, with its own extra
   invariants. Each family suite covers sign, frame, units, an oracle, gradients and a swap case. Passing the suite in
   your own CI is how a lab claims it.
2. **Their own benchmarks.** These use MADDENING's `@verification_benchmark` under the lab's own id prefix, so they
   land in the same evidence tables.
3. **Their own experiments.** Harness calls cover the round trip, golden outputs and the gradient checks.

## What goes here

- Family conformance suites, as harnesses.
- Graph-invariant helpers: snapshots, and order invariance.
- Differential and gradient helpers, and fixtures.

Every MIME node also passes `maddening.testing.verify_node`.

## Status

Wired to `maddening.testing` in dev1's phase 1. Since phase 2:
- the graph-invariant helpers: `graph_snapshot` (a built graph's canonical text), and order invariance
  (`Declare`, `declare_in_order`, `order_reference`, `order_invariance_failures`, `assert_order_invariant`, and
  `declaration_orders` for a Hypothesis property);
- `mime.testing.toys`: toy entries with no physics under the `toy` namespace, and the toy experiments MIME's build
  tests use. They live here, not in `tests/`, so a lab's tests can build the same toys.

Since phase 3:
- `round_trip_failures(experiment, folder)`: build, save, load in a fresh process, rebuild; every difference in the
  derived hash or, bit for bit, the first steps. `refused_round_trip_failures` does the same for an experiment whose
  build is refused, through its document. `installed_entries()` lists, in a fresh process, every entry MIME and the
  installed `mime.entries` packages register: what a registry walk must cover. `covered(document)` names the entries
  and kinds of thing a document uses;
- `param_reach_failures(built)`: every trainable `params` leaf of a built graph reaches its compiled step, without a
  recompilation. Each leaf is written to a new value and the run must change, while JAX's compilation events and
  MADDENING's scan counter must not move. An initial condition (`initial_*`) reaches the run through the state
  instead, and is not asked.

The family suites arrive with the first families, in phase 4.

Phases and releases are explained in the [package manifest](../README.md#status).
