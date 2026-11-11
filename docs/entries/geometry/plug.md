---
describes: "0.1.0"
---
# `plug`

A solid circular cylinder placed in the world, `radius` and `length` along `axis` through `centre` (its midpoint): as a
structure, a region of a vessel that a porous material fills, a clot,
`Structure("clot", geometry=Plug(axis=+Z, radius=mm(0.5), length=mm(1.0), centre=mm((0, 0, 2))))`. Its one region is its
`volume`. It gives no samples, volume or inertia: it is not a body's part. Its role is the law bound to it: with
[Darcy's law](../interaction/darcy.md) attached, a medium's flow holds it as a porous region, reading its centre,
axis, radius and length and sampling its exact distance on the flow's own grid. On a stage it is written as USD's own
`Cylinder`, its `height` holding `length`, its axis the prim's z axis.

**When not to use it.**
- For a body's part: that is the `cylinder` solid, placed in its body by `Part(at=...)`.
- For a region of another shape: a plug is a cylinder (a clot of another shape needs a kind of its own).
- As a solid wall: a plug with no law bound to it changes no flow.

**Its derivative.** The kind evaluates nothing at run time; the flow that samples its distance gives its derivative
(the region's exact distance has a kink on its medial set, inside, and on its axis).

**Regions, surfaces, directions and frames.** What a kind names and how a law or a stage reads it: [the scene note](../../scene.md).

**Versions.** 0.1.0 is its first.
