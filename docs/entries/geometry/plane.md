---
describes: "0.1.0"
---
# `plane`

An unbounded plane wall as a structure: the plane `x . n = offset`, `normal` its direction (towards the medium) and
`offset` its signed distance from the world's origin along it, the plane through the origin unless placed. It names two
regions, the `half_space` its normal points into and the `wall`, the other side; only the `half_space` has a shape, the
half-space (`mime.core.HalfSpace`). A medium fills it: `Medium("water", fills=floor.half_space)`.

What a consumer reads of it, by what it is for:
- **A medium** filling the `half_space` sees an unbounded plane wall.
- **The hydrodynamic wall** is a different entry: the [plane wall kind](../wall_kind/plane.md) is the wall's effect on
  the flow, and it reads the plane's normal and offset from this structure.

**When not to use it.**
- For a body: a plane has no mass properties and gives no samples.
- For a contact: it has no surface yet, so no contact can be held against it; that comes later.
- For a finite wall: it is unbounded, and has no drawing on the stage.

**Its derivative** is the half-space's: its distance is linear in the point and the offset, with no switch.

**Regions, surfaces, directions and frames.** What a kind names and how a law or a stage reads it: [the scene note](../../scene.md).

**Versions.** 0.1.0 is its first.
