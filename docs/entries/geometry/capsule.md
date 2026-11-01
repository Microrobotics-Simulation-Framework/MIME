---
describes: ["0.1.0", "0.2.0"]
---

# `capsule`

A solid capsule: a cylinder of a radius and a height (between the centres of its caps, as USD's `Capsule` counts it)
along an axis, capped by two hemispheres, centred on its centroid. An elongated body pulled by a field gradient, or a
rod. As a body's part it gives the body its volume and inertia, its extent (its radius, about its own axis) and its
surface samples by patch, `lateral` and `caps`; at run time a node evaluates it by the shape `capsule`.

**When not to use it.** For a body whose extent across its axis is not its radius all along: a tapered or a threaded
body. A capsule tilted in a tube reaches further than its radius from the tube's axis, which the extent does not hold.

**Its derivative** is the shape's: the distance from its segment less the radius, so on the segment the distance is a
cone and the normal has no limit; across the planes through the segment's ends only the curvature changes.

**Its sampling.** At a spacing, for a solver whose unknowns sit at the samples (the MRS rung), it samples its surface
quasi-uniformly: as many samples as its quadrature has at that spacing, chosen from its quadrature at a finer spacing
by farthest points and Lloyd's relaxation (`mime.scene.quasi_uniform`), each a quadrature sample with its cell's area as
its weight, so that no two crowd. Its hull query, its reach in a direction, is the sphere's about the end of its segment
that lies that way.

**Versions.** 0.2.0 (phase 7m-a) samples its surface quasi-uniformly; 0.1.0, the first (phase 7d), sampled it by its
quadrature, whose polar grids crowd the caps' poles (MIME-ANO-123). 0.1.0 stays registered so that a folder saved with
it loads and reproduces; a script that writes `Capsule(...)` gets 0.2.0. The quadrature, the shape and every closed form
are unchanged.
