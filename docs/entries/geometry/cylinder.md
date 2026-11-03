---
describes: "0.1.0"
---
# `cylinder`

A solid circular cylinder, given by its `radius`, its `length` and its `axis`, centred on its centroid: a rod, a disc, the
body of a cylindrical magnet. As a body's part it gives the body its volume and inertia (closed forms), its extent, its
reach along a direction and its surface samples by patch. At run time a node
evaluates it through its regions, the unbounded cylinder `side` and the `span` between its end planes, the body being
their intersection; its patches are the `lateral` side and the two flat `caps`. On a stage it is written as USD's own
`Cylinder`, its `height` holding `length`, and its axis is the prim's z axis.

**When not to use it.**
- For a body that is not round about an axis: a cylinder's extent and samples are a cylinder's.
- Where a consumer needs the hull's radial reach about its axis (`radial_reach`): the kind implements a reach in a
  direction and a support in a direction, not that; a consumer that needs it is refused when it plans, naming both.
- For a hollow tube or a wall: that is the `tube`, which is a structure, not a solid body.

**Its derivative** is its regions' shapes', as their switches declare: on the axis the distance is a cone and the normal
has no limit, and at the mid-plane the distance has a kink. Its support in a direction is a closed form, and along the
axis the farthest set is the whole cap, so the farthest point jumps from the cap's centre to its rim as the direction
leaves the axis (the switch `cylinder.support`, with no route). Its volume and inertia are exact.

**Its sampling.** At a spacing, for a solver whose unknowns sit at the samples (the MRS rung), it samples its surface
quasi-uniformly, chosen from its quadrature (`mime.scene.quasi_uniform`): the trapezoid rule round the axis with
Gauss-Legendre nodes along it, and each cap in rings about a sample at its centre, the layout the screw's caps use, which
no ring crowds. The quadrature is built to integrate the area and the divergence integrals exactly. The choice of
samples is discrete, so it is not continuous in the settings: rebuilt at a slightly moved radius or length, the sampling
can choose again and the MRS resistance step with it (the numbers are MIME-ANO-127's).

**Regions, surfaces, directions and frames.** What a kind names and how a law or a stage reads it: [the scene note](../../scene.md).

**Versions.** 0.1.0 is its first.
