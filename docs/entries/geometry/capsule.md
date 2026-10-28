---
describes: 0.1.0
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

**Versions.** The first, landed with the sphere and the de Jongh helix in phase 7d.
