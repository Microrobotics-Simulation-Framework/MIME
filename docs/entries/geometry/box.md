---
describes: "0.1.0"
---
# `box`

A solid rectangular box, given by the three lengths of its edges (`size`), along the part frame's axes and centred on its
centroid: a block, a slab, the body of a rectangular magnet. As a body's part it gives the body its volume and inertia
(both closed forms), its extent (half its diagonal about any axis), its reach along a direction and its surface samples
by patch, for a solver that integrates over the surface. At run time a node evaluates it through its regions, three slabs
between the pairs of opposite faces (`span_x`, `span_y`, `span_z`), the body being their intersection; its patches are the
three pairs of faces.

**When not to use it.**
- For a body that is not a box: its extent and anything read from its samples would be a box's.
- Where a consumer needs the body's deepest point in a direction (`support`): the kind answers a reach, not a support, and
  a consumer that needs one is refused when it plans, naming both.
- Where the stage must show it as a USD primitive: it has no native form, so the stage writer draws it (the
  capabilities table says so).

**Its derivative** is its slabs', as their switch declares: at a slab's mid-plane the distance has a kink. Its volume
and inertia are exact.

**Its sampling.** At a spacing, for a solver whose unknowns sit at the samples (the MRS rung), it samples its surface
quasi-uniformly, chosen from its quadrature (`mime.scene.quasi_uniform`); the quadrature is Gauss-Legendre along both
edges of each face, so that its divergence integrals are exact. The choice of samples is discrete, so it is not
continuous in `size`: rebuilt at a slightly moved edge, the sampling can choose again and the MRS resistance step with it
(the numbers are MIME-ANO-127's).

**Regions, surfaces, directions and frames.** What a kind names and how a law or a stage reads it: [the scene note](../../scene.md).

**Versions.** 0.1.0 is its first.
