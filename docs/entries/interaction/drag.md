---
describes: 0.3.0
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
sphere, a capsule, the de Jongh helix) [@Cortez2005]: the kind samples its surface on the patches written (its whole
closed surface when none are), neighbouring samples about `spacing` apart; a point force at each sample, smoothed by a
blob of `blob` times the samples' mean spacing, makes the flow, and the forces that move the samples rigidly give the
resistance. The samples are placed by the body's pose at every step, so the resistance turns with the body, and they
are a smooth function of the kind's numbers, so a study's derivative in a radius or a wavenumber passes through them:
its derivative is exact, checked against differences by rebuilding the experiment.

`wall_kind` names the wall whose correction the solve adds: `free_space`, an unbounded medium, for a medium that fills
no region, and [`tube`](../wall_kind/tube.md), a straight circular tube's no-slip wall, for one that fills a tube's lumen.

**When not to use it.**
- In a tube or near any wall: the free-space wall leaves it out, and a medium that fills a region is refused with it.
- Where the cost matters: each step solves a dense system of three unknowns per sample, so its cost grows as the cube
  of the samples' count; the coarsest spacing that resolves the body's shape is the one to write.
- For a body whose hull is several parts, or whose kind gives no samples (a cylinder, a box): refused when planned.
- Where the error must be bounded rather than estimated: the regularised method's error is measured, never bounded,
  against the sphere's and the prolate spheroid's closed forms by its oracle tests and, at the default spacing, by
  `scripts/default_spacing_table.py`, whose misses the default states in numbers on the stamp and in `mime pending`
  (a screw's is large): write `spacing=` for a number to use. Leaving out a patch, as the de Jongh helix's caps were
  left out before, is a model difference of its own.
- As the converged value of a de Jongh helix at the counts the v0.2 line used: its samples and that line's triangle
  mesh differ there, by how much its validity lines say, and neither has converged; the converged diagonal lies
  between the two (MIME-ANO-116). A surrogate trained on either inherits that error.

## Versions

Version 0.3.0 reads the body's shape from its hull's kind, not from declared values; 0.2.0 moved the resistance from
the end of the body's axis to its centre; 0.1.0 delivered it about the end (MIME-ANO-112). The mrs rung was added to
0.3.0, changing nothing the surrogate computes.
