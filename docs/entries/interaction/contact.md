---
describes: ["0.1.0", "0.4.0", "0.5.0"]
---
# A body against a surface

**What it models.** A rigid body kept on the free side of a fixed structure's surface, every point of its hull at least
`margin` from it: the side the surface faces (`tube.inner_surface` faces the lumen), the body's orientation read, so a
long body turned across a narrow tube meets the wall at its ends and not at its centre.

<!-- mime-fragment: one attach line of an experiment shown in the middle of a sentence; the experiments that attach it are built by tests/differential/ -->
```python
experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
```

It builds no node of its own. Its record, a confinement, goes to the body's dynamics at construction, and the dynamics
holds the body inside its implicit solve each step and projects the pose back after it (`RigidInertialNode` is the
dynamics that takes it). The hull's points, sampled at `spacing` on every patch, are split into groups,
and each group is searched from as many places as its shape can hold local deepest points that its samples do not order
(counted when the experiment is built). Where the hull's kind answers the support query (`Geometry.support`: closed forms
for the sphere, the ellipsoid, the cylinder and the capsule, Newton's steps on its own surface for the screw), the
contact holds the body's own point that a converged search reaches, so the touching point moves continuously along the
body as it turns. At each held point the outward velocity is taken away with the body's rotation coupled in (a turned body
touching at two points takes the couple that holds it), inelastic and frictionless, by projected Gauss-Seidel over the
points. A body with no hull (no parts, its extent declared) is held as a sphere of its extent about its centre, which is
exact for a sphere.

It says where the body may be and the force that keeps it there, and it adds no near-wall lubrication, which a drag
that includes it declares (`Interaction.includes_lubrication`); the plan refuses a second law on the body that includes
it. Its `margin` is where it engages: written, a numerical choice; unwritten (0.5.0), the roughness cutoff of the
body's and the wall's surfaces (`Contact.engagement`, the root of the sum of the squares of each surface's declared
`roughness`, or the declared default where none is), the gap below which the near-contact law's film is floored, so that a
body riding a film above it is held by the film alone.

**Friction** (0.5.0), written as `friction=`:

<!-- mime-fragment: one attach line of an experiment shown in the middle of a sentence; the experiments that attach it are built by tests/oracles/test_friction.py -->
```python
experiment.attach(Contact.Projection(friction=Coulomb(mu=0.15)), body=ball, surface=tube.inner_surface)
```

is solved in the contact's own sweeps: each held point carries a sliding pair of rows `[t; r x t]` beside its normal
row, which measure the slip of the body's own point (`v + omega x r`), bounded on the elliptic cone by `mu` times the
point's normal force. One coefficient serves sticking and sliding; a point sticks exactly (a block on a slope stays put
below `tan(theta) = mu`), and a body rolls without slipping where its sliding rows stick. Each pair's update is a scaled
trust region on its own block, repeated within each visit; after the sweeps the forces are scaled to where the normal block does no work,
and each point's friction power at the solved twist is checked never positive (the guard), the step flagged
`unconverged` where it is. Friction is booked in the energy ledger under its own name (`<contact>.friction`) as its
traction on the slip it acts on, `mu N` times the slip distance while it slides and zero while it sticks, the sweeps'
unconverged remainder its own numerical column; the contact takes the rest. It acts on the contact's own normal force
alone, never on a lubricating film's pressure; the film's coupling of rotation and translation is in the resistance the
rows are solved through. The state adds each point's friction (`contact_traction`), its slip speed (`contact_slip`) and
the guard's reading (`contact_friction_power`). Unwritten, the contact is frictionless and compiles the program it
always did. The seams for what is left out (`mime.interactions.friction`): `FrictionLaw` (a speed-dependent law, a
static coefficient above the kinetic), `NormalLaw` with `Hold()` its default (restitution, soft contact; `Hold`'s
`adhesion` reserved), the row kinds (rolling resistance built, spin declared) and the point sets (line and patch
contacts), each with its conformance suite in `mime.testing`. **Rolling resistance**, `Coulomb(mu=..., rolling=...)`
(a length), adds a pair of angular rows `[0; t]` at each point, a torque about the surface's tangents bounded by the
length times the normal force: a sphere on a flat slows while it rolls, and one on a slope stays put below the
threshold the length over the radius sets. A synchronous rolling speed does not depend on it (the torque budget
absorbs it until step-out), so only data past step-out identify it; at zero its derivative is the slope into the
admissible side. **Identifying mu** (`mime.study.fit_friction`): `Study.fit` on the data a run gives, a
gradient-free search over the register's range on the same loss as its control, and a verdict: `identified`, `bounded
below only` (data that stick or roll without slipping hold only a lower bound) or `confounded`; a fitted value ships
as a named record pinned by its content's and its data's hashes, never a default. One contact per body, against a fixed structure's surface. Its rung is `projection`; `penalty`
is reserved. `Contact.Projection(...)` is `Contact(rung="projection", ...)` and saves identically.

What a run says of it, as flags: `touching` where a point takes a force, `unconverged` where the solve left more of the
free step's speed unmet than its tolerance allows, `held_by_samples` where a kind with no support query is held at its
samples, and `support_short` where the body may reach past a held point. Its derivative is piecewise: it jumps where two
of the body's own deepest points tie or where a step changes the contact's branch, as the switches on this page say.

**When not to use it.**
- Where adhesion, a rebound or a compliant wall matters: it is inelastic and rigid, and adhesion is reserved
  (`Hold(adhesion=...)`); where friction varies with the slip speed or a static coefficient exceeds the kinetic one: a
  `FrictionLaw` of its own.
- For flow near the wall: the lubrication that a close wall adds to the drag is the drag's to include, not the contact's.
- For a hull whose kind has no support query (the box, a lab's mesh), where the held point is a sample and the hold acts
  as a friction and a shaking that the `spacing` sets: the validity lines say so, and `held_by_samples` flags it
  (MIME-ANO-131 holds the measurements).
- For two bodies against each other: it is only against a fixed structure's surface.
- For a study that moves a setting of the hull far: the groups, their starts and their candidates are fixed from the
  hull kind's settings as built, while the points are evaluated each step from the live settings. A derivative in a
  setting keeps them; a far move keeps the old candidates, which may then seed otherwise than all the points would, and
  that is not measured.

**The surface and its free side.** A surface faces the region it bounds, and the contact keeps the body on that side: [the scene note](../../scene.md).

**Versions.** `contact` 0.5.0 runs the hold's sweeps and its search's Newton steps as XLA loops, the same updates in the
same order, where 0.4.0 unrolled them, a call each: the step's program is smaller and compiles faster, and its rounding
is not 0.4.0's. A screw held from its first step in script A's tube moves, in float32, its velocity by about a millionth
of its size; the forces among its held points move by up to two parts in a thousand of their scale, since their split
is not unique, while their net force moves by under a thousandth of the momentum rate it came from, on the runners
measured; in float64 each moves by a hundred-billionth of its scale or less, and a sphere in the tube demo is bit for bit (`tests/oracles/test_contact_loops.py` holds the two
versions to each other). 0.4.0 stays registered (`mime.interactions.KEPT`), so a folder saved with it reproduces to the last bit.
`contact` 0.4.0 chooses each group's point again in the second pass of the projection after the step, at the pose the
first pass left, and holds it beside the first pass's points: a group can hold two of the body's local deepest points (a
screw's rim and the crest pass beside it), and the first pass's turn could leave the one it did not hold past the margin
(MIME-ANO-136). 0.3.0, which was in no release, held the first pass's points alone; it started each group's search from
as many places as the hull's shape needs and held the point only of a search that converged (MIME-ANO-134 and
MIME-ANO-135 record what the earlier search did). 0.2.0, which was in no release, held the hull's points and climbed a
fixed count of steps from the group's nearest sample. 0.1.0 held the body's centre by its extent about its own axis and
never read its orientation, so a body that turned across a narrow tube passed through the wall (MIME-ANO-131); it stays
registered (`mime.interactions.KEPT`), so a folder saved with it loads and reproduces, and a script that writes
`Contact` gets 0.5.0. For a body whose hull no point reaches the margin of, the body steps as the free body does, to the
last bit, in every version.
