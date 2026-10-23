# `mime.testing`

**Harnesses a lab calls from its own tests.** They are built on `maddening.testing`, not beside it. Install them with
`pip install "mime-engine[verify]"`.

## How users add their own tests

1. **Their own entries.** A lab's drag law calls `mime.testing.assert_conforms(MyDrag(), family="hydrodynamic",
   oracle=...)` from its own pytest files. Each family suite holds the cases its family needs, from sign, frame,
   units, material reach, oracle, gradients, swap, equal and opposite (for two-way laws) and nodes. Passing the suite
   in your own CI is how a lab claims it. The entry's nodes pass MADDENING's battery through
   `assert_nodes_verified(built, MyDrag, NodeCase(...))`.
2. **Their own benchmarks.** These use MADDENING's `@verification_benchmark` under the lab's own id prefix, so they
   land in the same evidence tables.
3. **Their own experiments.** Harness calls cover the round trip, golden outputs and the gradient checks.

## What goes here

- Family conformance suites, as harnesses.
- MADDENING's node verification on every node an entry builds.
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

Since phase 4a:
- **the family suites' harness.** A `Family` states, over the vocabulary's interfaces rather than one law's labels,
  its sign rule (`SignRule`, or `opposes(delivered, reading)`), the settings that stand for a material's properties,
  the pairs a two-way law delivers equal and opposite, a reference member a swap replaces, and a rig, the smallest
  experiment holding one of its laws; `register_family` adds it. `conformance(law, family, oracle=, swap_with=,
  node_case=, cases=)` returns a MADDENING `VerificationResult` per `CaseKind`, and `assert_conforms` raises listing
  every failure. The sampled cases are invariants of `verify_node` on the law's node, as the rig built it, at its
  default 200 examples each: **sign** (the family's rule), **frame** (turning every world-frame vector the law reads
  turns every one it delivers), **oracle** (the law's closed form), **gradients** (forward-mode derivative against
  a central difference, within the difference's own truncation error) and **equal and opposite**. **Units** checks
  each input port's units against its read's and refuses each dimensioned setting in another dimension; **material
  reach** changes each material setting and requires the response to change; **swap** puts the law in place of
  another member and requires the rig's graph to be the same, the law's own node aside; **nodes** runs the battery
  below. Tolerances have no absolute part: each is relative, plus a floor measured from the law's own rounding at
  that point, so an SI torque of 1e-12 N m is not read as zero;
- **the node battery.** `node_failures(built, entry, NodeCase(...))` runs `verify_node` on every node an entry
  builds, drawn in the dtype the node computes in, and `verify_node_order` on each order a node declares, against an
  `OrderLadder` the case gives; a declared order with no ladder fails. `nodes_by_entry(built)` groups a built graph's
  nodes by the entry that built them. MIME's own node walk (`tests/saved/test_node_walk.py`) requires a `NodeCase`
  for every registered entry;
- **the toy family** (`mime.testing.toys.TOY_FAMILY`, name `toy`), the control: the toy drag's two rungs conform, and
  in MIME's tests a planted law fails each case.

Since phase 4b:
- **MIME's families**: `MAGNETIC`, `GRAVITY` and `CONTACT` (names `magnetic`, `gravity`, `contact`), each with its
  rig (`magnetic_rig`, `gravity_rig`, `contact_rig`), its sign rule (the torque turns the moment toward the field;
  weight minus buoyancy lies along gravity with the excess density's sign; a projected body moves no further out),
  the values that stand for a material's properties, and the constraints a drawn reading must meet to be one
  (`unit_quaternion` for a pose, `positive_semidefinite` for a resistance). The swap case waits for each family's
  second member: a swap with itself could not fail. The hydrodynamic family comes with drag in 4c;
- the harness reads values beside settings: a law's oracle and rules see the values its plan reads, by label, and
  **material reach** changes a declared value (`redeclared`) as it changes a setting;
- **frame** turns composites as the vocabulary lays them out (a pose, a pair of vectors, a tensor, an operator:
  `turned`), and the node constants a family names as world-frame (the gravity a law was built with);
- a law that builds no node but confines a body (`Contact`'s projection) is probed through the body's node: it reads
  the state a twin of that node built without its confinement reaches, and delivers what the confined node reaches.
  What it delivers is that state with a part taken away, so a field that cancels is judged on the scale of the state
  it came from: in the oracle case component by component (a pose holds metres beside a unit quaternion), in the
  frame case on the whole field's scale, since turning mixes components;
- MIME's own **family walk** (`tests/families/test_family_walk.py`) requires a family case for every registered
  interaction, and every one in a family passes its whole suite.

Phases and releases are explained in the [package manifest](../README.md#status).
