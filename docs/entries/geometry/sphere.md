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

**What its sampling costs.** On the sphere the MRS rung's drag at a given count is further from Stokes's law on the
quasi-uniform sampling than on the grid of its first version, in translation and in rotation, at every count measured
from about a hundred samples to a few thousand: the grid misses by two thirds to three quarters as much. It is not the
grid's symmetry (the same holds on triaxial ellipsoids, and a uniform set that shares no code with the sampler gives the
sampler's numbers); the numbers are MIME-ANO-126's. The sphere's two-resolution extrapolation still meets Stokes's law
within the oracle's tolerance. The sphere samples quasi-uniformly all the same (design #195, a labelled default the
maintainer may overturn): it is the only closed-form drag oracle the default sampler has, and on the screw, the body
without the sphere's smoothness, the same sampling comes far closer to a converged reference. The separated quadrature
(phase 7m-b) chooses the defaults again.

**Versions.** 0.2.0 (phase 7m-a) samples its surface quasi-uniformly; 0.1.0, the first (phase 7d), sampled it by its
quadrature, Gauss-Legendre in the cosine of the polar angle and even in azimuth, which put a quarter of its samples
within half the mean spacing of another at the poles (MIME-ANO-123). 0.1.0 stays registered so that a folder saved with
it loads and reproduces; a script that writes `Sphere(...)` gets 0.2.0. The quadrature, the shape and every closed form
are unchanged.
