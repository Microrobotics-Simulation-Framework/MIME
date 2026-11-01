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
  second member: a swap with itself could not fail;
- the harness reads values beside settings: a law's oracle and rules see the values its plan reads, by label, and
  **material reach** changes a declared value (`redeclared`) as it changes a setting;
- **frame** turns composites as the vocabulary lays them out (a pose, a pair of vectors, a tensor, an operator:
  `turned`), and the node constants a family names as world-frame (the gravity a law was built with);
- a law that builds no node but confines a body (`Contact`'s projection) is probed through the body's node: it reads
  the state a twin of that node built without its confinement reaches, and delivers what the confined node reaches.
  What it delivers is that state with a part taken away, so a field that cancels is judged on the scale of the state
  it came from: in the oracle case component by component (a pose holds metres beside a unit quaternion), in the
  frame case on the whole field's scale, since turning mixes components. The rounding floor is still one number for
  the whole state, the largest effect of a 64-ulp nudge on any field, so in float32 a confined position is judged
  against the quaternion's rounding (four times about 4e-6, so about 15 micrometres, on a generic draw) or the
  twist's (up to centimetres). A smaller position error shows only on draws where the nudge moves nothing: a basis
  quaternion, no twist, no input, the position on one axis. Floors per field close this since 4c1 (below);
- the oracle case's closed form for a confining law, a free step projected back, cannot know that a step began in
  the node's contact band (a body on its surface and moving outward is held there inside the node's solve), so the
  case moves a body drawn within four bands of the surface to eight bands inside it (`off_the_contact_band`), and the
  band is tested on purpose in the family walk. A draw of that kind failed the contact oracle once in a local lane
  (about one run in a thousand, from the band's area against the draws' range: estimated, not measured);
- MIME's own **family walk** (`tests/families/test_family_walk.py`) requires a family case for every registered
  interaction, and every one in a family passes its whole suite.

Since phase 4c1:
- **rounding floors per field and per part.** Each field a law delivers has its own floor, and a composite one
  (`parts_of`: a pose's position and its quaternion, a pair of vectors' two, an operator's four blocks) a floor per
  part, each from that part's own movement under the 64-ulp nudge; each input part is nudged on the scale of its own
  largest component, since arithmetic that mixes a part's components (a rotation, a projection on an axis) rounds a
  small component on the scale of the large ones. Closeness is judged part by part too, so a rotational resistance
  of 1e-10 N m s is not judged on a translational one of 1e-3 N s/m. A confined position is now judged on the
  rounding of metres, not of the quaternion beside it: the contact oracle now catches a projection 70 nm off in
  every run (210 of 210 independent runs; 50 nm in 58 of 60), where in 4b a micrometre escaped about one run in
  eight;
- **gradients** step each input part in proportion to its own magnitude: a pose's millimetres move by a fraction of
  a millimetre, not by a fraction of its unit quaternion;
- a family may name constants a member holds only when it reads what they describe (`frame_constants_held`: a
  medium's geometry), displacements its setup is uniform along (`uniform_along`: moving every point the law reads
  along a straight tube changes nothing it delivers), and materials its laws are proportional to (`linear_in`);
- **the hydrodynamic family**, `HYDRODYNAMIC` (name `hydrodynamic`), with its rig (`hydrodynamic_rig`: a de Jongh
  FL-9 screw body in a water-filled tube along x): sign (the resistance symmetric positive definite, judged after
  scaling by its diagonal, so `V^T R V > 0` for every nonzero twist), frame (turning the tube and the body together;
  moving along the tube's axis), units, material reach (in proportion to the viscosity), oracle, gradients, swap and
  nodes. Equal and opposite does not apply: the tube is fixed and a medium at rest takes no reaction. Its draws put
  the body 0.2 to 1.0 mm off the tube's axis (`in_the_lumen`), where script A's contact holds a body and the
  surrogate is smooth; the axis, where the surrogate is discontinuous, and the wall, where its clamp and gap floor
  hold it, are tested on purpose in MIME's drag tests. Its second member, a Stokes sphere, is registered by MIME's
  tests only and passed as `swap_with=`;
- **check_units** reaches a group's settings (`lubrication.epsilon`).

Since phase 7s, **the switch harness**: `classify(SwitchRow(declaration, evaluate, delta), KINK_LIMIT)` places a point
a quarter of a step from an owner's switch (in the indicator's units), forces the owner's branch each way there in a
step traced afresh, and compares each output's forced-branch difference at the step and at a tenth of it: a kink's
shrinks, a jump's does not. A row passes when every group it declares continuous measures so, every group it declares
nothing for is unchanged, its indicator at each placed point is that point's offset, and the forced setting reached
it. A row whose every output stays within its floor measures nothing (`Classification.unmeasured`), and passes only
where its declaration says why no output can show the switch (`Declaration.unmeasured`); a row that measures
something fails where its declaration says none can. `KINK_LIMIT` is measured, on drag's clamp and end, and checked
on an analytic kink and jump; its resolution is stated beside it: a jump below about a third of the smooth part's
growth over a quarter step reads continuous, so each row's `delta` keeps that growth small beside the jump in
question. The rounding floor is each output's own, `FLOOR_ULPS` units in the last place of its largest value in
either branch. `fallback_problems` replaces every domain guard's fallback with other values (a zero by the scale
itself) where its condition is false, finds any value or derivative that changes, and reports each guard and
`nonzero_norm` it reached, by call site. A lab gives each owner it declares a row (MIME's are in
`tests/saved/test_switch_harness.py`).

Since phase 7d, **the geometry suites**. `kind_problems(kind, settings, spacing, counts=None)`, the kind-conformance
harness: each patch's samples lie on the boundary of the region it names and on or inside a solid's other regions
(its body is their intersection), their normals are the shape's (to the rounding of both: a position rounded at the
sample's rounding length, the helix's distance from its axis, carried through the normal's sensitivity, and the
normal's own, `NORMAL_ROUNDING_ULPS` declared with its measurement), their weights sum to the area, and over the
whole surface the divergence theorem gives the closed-form volume and inertia; the registry walk runs it on every kind
that samples its surface, and on the helix at every setting it accepts. The shape conformance suite: `check_shape(probe,
capability)` over `shape_probes(case)`, a probe off, near or on each switch a shape declares (`SHAPE_CASES`, one
`ShapeCase` per registered shape), under a random rotation and placement, with a parameter moved (a scalar, or the
centre, axis or orientation), in float64 or float32, autodiff against the float64 difference on the point's side,
judged on the derivative's scale with a floor of rounding and the difference's estimated truncation error (the Richardson value against its next halving; the step is halved until that error is a quarter of the allowance or less, and a probe where no step gets there is reported non-discriminating, never a pass); with `planted`, the tangent scaled wrong, its
control, by dtype. Near a set the float32 comparison resolves its floor, not its tolerance, as the suite's docstring
states. A registered shape with no case fails (`uncased_shapes`). `toy.region_probe`
evaluates the region a medium fills at a toy body, so that a step evaluates every solid's shape.

Since phase 7e, **the drawing check**, `drawing_problems(kind, settings, segments)`: a kind's drawing is a closed,
consistently oriented mesh facing outward; every vertex lies on the body's boundary to rounding (the kind's regions'
shapes, or for the tube, the box and the ellipsoid a surface the harness writes from their definitions, not from their
drawings); and the mesh's volume and area converge on the closed forms at second order (at twice the segments the
error is at most a third of what it was, and the error with the second-order term cancelled falls at least to its own
change between neighbouring resolutions). The vertices are resolved to rounding, so a drawing scaled by a part in
a hundred trillion is found; the volume and area to the fourth-order term they leave, between a millionth and a
ten-thousandth of the closed form at the writer's resolution (a drawing's permuted axes keep both, and are the
vertices'). `drawing_checks` says which apply to
a kind, and the registry walk fails on a drawn kind that lacks any; it runs the check on every kind that draws itself
and on the helix at every setting it accepts. A kind with a native form is
drawn by USD, and the stage tests hold USD's definition of it against the kind.

**What the harness does not establish.** MIME's rows run in float64, at points placed outside the float64 bands
(or inside one on purpose). The float32 bands (the contact band and the tube's axis band, each some units in the
last place of its scale, far wider in float32) and what a step does inside a band (the design's bound, the band's
resolution times the slope) are measured nowhere: they are unvalidated, not bounded. A row of a switch no valid state
reaches measures the law's order at a forced branch outside the valid states, not one a run meets.

A Hypothesis profile in MIME's `tests/conftest.py` prints the blob that replays a failing property
in every run, and is otherwise the profile Hypothesis would load (its `ci` profile on CI). The family cases run under
MADDENING's `verify_node`, which catches its property's failure and keeps the drawn counterexample, not the exception
the blob is attached to: for those, the counterexample is the reproduction.

Since phase 7a2, the **magnetic family** holds `Magnetic.Force` beside the torque: its sign rule's clause for a law
that reads a gradient pulls the moment up the gradient of its alignment (`F . (G^T m)` is not negative), and its rig takes a
fixed magnet (a point dipole) for such a law. `covered` counts a chain's segments.

Since phase 7m-a, **the sampling conformance suite** (`_sampling_conformance.py`), which
`tests/saved/test_sampling_walk.py` runs on every registered kind, MIME's and every installed namespace's: a kind's
`sampling` spreads its samples evenly (`sampling_problems`: each sample's nearest neighbour over the samples' mean
spacing, its fifth percentile at least `NEAREST_FIFTH` and its least at least `NEAREST_LEAST`; each patch's, the seams'
and any labelled region's share of the samples within `COVERAGE_BAND` of its share of the area; the weights summing to
the area), keeps the MRS matrix conditioned at today's blob and at the larger fractions a smoother blob needs
(`conditioning_problems`, `CONDITION_BOUNDS`), its `reach` holds its body and no more against a dense quadrature
(`reach_problems`: an envelope may overstate by its declared share), and a wall kind's `distance` is its normal's
potential and its closed form's (`distance_problems`). A kind's own quadrature fails the spacing check, and is its
control. The kind-conformance harness checks a patch whose region no registered shape evaluates (the ellipsoid's)
against a level set written from the kind's definition (`_LEVELS`).

Phases and releases are explained in the [package manifest](../README.md#status).
