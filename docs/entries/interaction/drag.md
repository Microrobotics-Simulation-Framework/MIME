---
describes: ["0.3.0", "0.4.0", "0.5.0"]
---

# `drag`

The medium's resistance to a body's motion: the 6 x 6 operator a rigid body's dynamics solves with implicitly, about
its centre of mass, in the world frame, with the load a flowing medium puts on the body held still beside it (zero in
a medium at rest, the only medium built). The rung is written in the script, and there is no default:
`Drag.Surrogate(...)` or `Drag.MRS(...)`.

## Rung `surrogate`

The learned resistance of a de Jongh screw body in a straight tube: a Cholesky network trained on the MRS
resistances of that family, read at the body's offset from the tube's axis, with the near-wall lubrication correction
blended in when `lubrication` is written. It covers the bodies and tubes it was trained on and refuses others when the
experiment is planned.

**When not to use it.** For any body but a de Jongh screw, or a tube outside its training data; for a body far from
a wall, where the mrs rung needs no training.

## Rung `mrs`

The regularised-Stokeslet resistance of any rigid body whose hull kind gives surface samples (a
sphere, a capsule, the de Jongh helix, a cylinder, a box, an ellipsoid) [@Cortez2005], on a separated quadrature
[@Smith2018; @GallagherSmith2021]: the kind samples its surface on the patches written (its whole closed surface
when none are), neighbouring force points about `spacing` apart, each with one traction, constant over its cell; the
integral is the kind's own quadrature `quadrature_refinement` times finer, each of its points in the cell of the force
point nearest it, so that the blob, `blob` times the force points' mean spacing, can be smaller than the force points'
spacing and still be resolved. The force points and the quadrature are placed by the body's pose at every step, so the
resistance turns with the body, and they are a smooth function of the kind's numbers, so a study's derivative in a
radius or a wavenumber passes through them: its derivative is exact, checked against differences.

**What it costs.** The part of the matrix the body makes with itself is the same at every pose, so it is assembled
once per build, in the body's frame, over the quadrature in chunks (never every force point against every quadrature
point at once), and held: in free space the whole resistance is held and each step only turns it, so a run costs its
build; with a wall, each step adds the wall's correction at the force points and solves, which costs as the cube of the
force points' count. **A wall's correction is evaluated at the force points**: the image system varies over the
distance to the wall, which the force points resolve while the body is more than about a force spacing from it.

**Its defaults are by family, measured** (`docs/choosing_spacing.md` holds the numbers): for a compact body (a sphere,
and any kind not measured) the count of force points 0.3.0 took, the blob a tenth of their spacing and the quadrature
twenty times finer; for a screw the coarsest count that keeps its shape, the blob a quarter and the quadrature eight
times finer. The quadrature's refinement, when not written, is two over the blob. The build refuses a free resistance whose
symmetric part is not positive definite, or which is symmetric only to more than a hundredth of itself (a check for
gross defects only: the point forces are symmetric to rounding however far off they are), and, with a wall, a
resistance at the body's starting pose that is not positive definite (`drag.mrs-resistance`); a refinement written
below its rule's where the rule's quadrature moves the free resistance's significant entries by more than a twentieth
(`drag.mrs-quadrature`: that move estimates the written quadrature's error, and it predicted the free-space misses
measured, the point forces' at a small blob among them); and a force point with no quadrature point in its cell
(`drag.empty-cell`).

**Near a wall** (MIME-ANO-130 holds the numbers). The wall's image is taken at the force points unless `wall_refinement`
is written. Taken there, it left the confined resistance indefinite within half a force spacing of the wall, on the
point forces as on the separated quadrature, its sign changing with micrometres of the pose. Written, the step
integrates the image over each force point's cell on a quadrature that many times finer than the force spacing, each
cell's weights scaled to its area and its points capped at twice the mean, as the free part is integrated (the image is
a kernel between any targets and any source cells, the wall kind's own, and the same kernel will take one body's force
points against another's cells). Integrated, the resistance is definite further in, at a step several times the point
image's. So:

- a confined body that a contact holds off the wall takes, by default, a force spacing no coarser than its closest
  approach over `CONFINED_GAP` force spacings, outside the band the resistance was measured indefinite in (over
  `INTEGRATED_GAP` with `wall_refinement` written), and the build report states what that costs: the force points, the
  unknowns a step solves, the image's reads and the run's memory. On a millimetre screw at a contact's usual margin
  that is thousands of force points, most of a laptop's memory and most of a minute a step on its CPU in float64
  (MIME-ANO-130 has the figures), a price worth reading before the run: write a coarser spacing for a look, and expect
  the stop below if the body reaches the band;
- on a machine with a laptop's memory, write `wall_refinement`. The default was chosen by a rule fixed before its
  measurement that weighed a step's time alone, not memory, and memory is where the two differ: at a
  millimetre screw's usual margin, the integrated image's coarser spacing took a step about as long as the default's
  and under a third of its memory, and was definite on every floor roll measured, as the default was. The build report
  names the refinement to write and the memory below which it is the option, beside the run's memory, and
  MIME-ANO-130 holds the figures;
- each step with a wall judges its resistance before the body takes it: one that is not positive definite (its least
  eigenvalue, each row and column scaled by its diagonal, not above zero) is replaced by NaN and the run stops at that
  step, `drag.mrs-indefinite`, naming the step, the pose, the body's least distance from the wall and a force spacing
  that clears it; a step with the body within one force spacing of the wall raises the flag `near_wall`, and one whose
  resistance is symmetric only to more than a twentieth of itself raises `asymmetric` (its error is at least half that
  share). The check is on the resistance the node delivers, so it holds for any solve that delivers one.

**Several bodies: the medium's near field.** Where two bodies or more in one medium take the mrs rung, the build
solves them together, as one joint node for the medium (`near_field:<medium>`): each step it assembles their joint
system (each body's own block as above; each pair coupled by the source's regularised Stokeslet, integrated over its
cells within a reach of the other body, in each cell's own size, and taken at its force points beyond; the wall's image
between them) and solves it once, densely, for the cluster's grand resistance. Each body is delivered its own coupled
resistance and, as its background load, the others' load on it at their velocities at the step's start: their steps
are **lagged**, each body's own resistance implicit, the others' loads a step behind. That costs a first-order error in
the step, and a step's delay after a sudden change, relaxing at the lagged step's **gain**, which the node checks every
step from each body's mass matrix: at one or above (three overdamped bodies in a triangle near contact) the lagged error
would grow, and the run stops there (`drag.near-field-gain`); the build refuses a configuration that starts there.
Each body's coupled resistance and the cluster's are judged as one body's is near a wall (`drag.near-field-indefinite`),
their margins reported. A body alone in its medium keeps its own node, step for step. The dense solve's memory grows
as the square of the cluster's force points, so the build refuses a cluster past `DENSE_UNKNOWNS` unknowns
(`drag.near-field-size`). With the energy ledger attached, each body's own resistance is its drag's law, and the
cross loads are the near field's own law, `near_field:<medium>`, booked on each body by the one pairing rule and judged
in the cluster's sum by the node (its quadratic form at the step's starting twists), never on one body's row: one
body drags another. A
body on the surrogate rung in the same medium is computed alone, and the build report says so.

`wall_kind` names the wall whose correction the solve adds: `free_space`, an unbounded medium, for a medium that fills
no region, and [`tube`](../wall_kind/tube.md), a straight circular tube's no-slip wall, for one that fills a tube's lumen.

**When not to use it.**
- In a tube or near any wall: the free-space wall leaves it out, and a medium that fills a region is refused with it.
- Where the cost matters near a wall: each step with a wall solves a dense system of three unknowns per force point,
  so its cost grows as the cube of their count; the coarsest spacing that resolves the body's shape is the one to
  write.
- Within a fraction of a force spacing of a wall at a coarse spacing: the resistance can be indefinite there, with
  the image at the force points or integrated over each cell, and a run that reaches one stops (MIME-ANO-130). The point forces (a
  quadrature refinement of one) are no way out: they were measured indefinite near the wall at every spacing mapped.
  Write a finer force spacing, as the refusal says, or hold the body further from the wall. A definite resistance close
  to the wall is not shown accurate.
- On a laptop with less memory than script A's frozen point needs on this rung (MIME-ANO-130 states it).
- For a body whose hull is several parts, or whose kind gives no samples: refused when planned.
- Where the error must be bounded rather than estimated: the regularised method's error is measured, never bounded,
  against the sphere's exact regularised drag and Stokes's law, the prolate spheroid's closed form, Haberman and
  Sayre's confined sphere and the shipped free-space reference of a de Jongh screw by its oracle tests and, at the
  defaults, by
  `scripts/default_spacing_table.py`, whose misses the default states in numbers on the stamp and in `mime pending`:
  write `spacing=` for a number to use. Leaving out a patch, as the de Jongh helix's caps were left out before, is a
  model difference of its own.
- As the converged value of a de Jongh helix at the counts the v0.2 line used: its samples and that line's triangle
  mesh differ there, by how much its validity lines say, and neither has converged; the converged diagonal lies
  between the two (MIME-ANO-116). A surrogate trained on either inherits that error.

## Versions

Version 0.5.0 solves a medium's mrs bodies together, its near field; 0.4.0 (`UncoupledDrag`) stays registered so that a
folder saved with it loads and reproduces, each body's drag computed alone. Version 0.4.0 separates the mrs rung's quadrature from its force points and chooses the rung's defaults
again by measurement; its surrogate rung computes what 0.3.0's did. 0.3.0 stays registered so that a folder saved with
it loads and reproduces (the frozen sim2real register's experiments build it): its mrs rung puts a point force at each
sample, the samples its quadrature, with the blob half their mean spacing, which a quadrature refinement of one on
0.4.0 also gives; a script that writes `Drag.MRS(...)` gets 0.5.0. Version 0.3.0 reads the body's shape from its hull's kind, not
from declared values; 0.2.0 moved the resistance from the end of the body's axis to its centre; 0.1.0 delivered it
about the end (MIME-ANO-112). The mrs rung was added to 0.3.0, changing nothing the surrogate computes.
