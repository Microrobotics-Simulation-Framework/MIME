---
describes: ["0.1.0", "0.2.0"]
---

# `sphere`

A solid ball, given by its radius and centred on its centroid: a bead, a particle, or the body of a magnetic sphere
pulled by a field gradient. As a body's part it gives the body its volume and inertia, its extent (its radius, about
any axis) and its surface samples by patch, for a solver that integrates over the surface; at run time a node
evaluates it by the shape `sphere`.

**When not to use it.** For a body that is not round: its extent and anything read from its samples would be a
sphere's.

**Its derivative** is the shape's: at the centre the distance is a cone and the normal has no limit. Within a band
about the centre the distance is minus the radius and the normal zero, each with the derivative zero in the point.

**Its sampling.** At a spacing, for a solver whose unknowns sit at the samples (the MRS rung), it samples its surface
quasi-uniformly: as many samples as its quadrature has at that spacing, chosen from its quadrature at a finer spacing
by farthest points and Lloyd's relaxation (`mime.scene.quasi_uniform`), each a quadrature sample with its cell's area as
its weight, so that no two crowd. Its hull query, its reach in a direction, is its radius times the direction's length.

**Versions.** 0.2.0 (phase 7m-a) samples its surface quasi-uniformly; 0.1.0, the first (phase 7d), sampled it by its
quadrature, Gauss-Legendre in the cosine of the polar angle and even in azimuth, which put a quarter of its samples
within half the mean spacing of another at the poles (MIME-ANO-123). 0.1.0 stays registered so that a folder saved with
it loads and reproduces; a script that writes `Sphere(...)` gets 0.2.0. The quadrature, the shape and every closed form
are unchanged.
