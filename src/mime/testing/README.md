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

Wired to `maddening.testing` in dev1's phase 1, with no harnesses yet. The graph-invariant helpers arrive in phase 2,
and the family suites with the first families in phase 4.

Phases and releases are explained in the [package manifest](../README.md#status).
