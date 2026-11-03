---
describes: "0.1.0"
---
# `tube`

A straight tube as a structure: a wall of thickness `wall` around a circular lumen of radius `inner_radius`, `length`
long, along `axis` through `centre`. It is what a body is confined in and what a medium fills; it is not a solid body, and
it gives no samples, volume or inertia. It names two regions, the `lumen` and the `wall`, but only the `lumen` has a shape: a law reads
the `lumen` as a region (an unbounded cylinder), and the `wall` is a name that the stage draws as an annulus: it has no
shape for a law to read (`region_shapes` holds none for it). Its `inner_surface` faces the lumen.

What a consumer reads of it, by what it is for:
- **A contact** bounds a body against the `inner_surface`, which is evaluated as an **unbounded** cylinder: the body is
  bounded radially, and the tube's ends bound nothing. A body past an end is outside the tube, which nothing checks yet.
- **A medium** filling the `lumen` sees a uniform tube along its axis with no ends: the lumen is that cylinder's inside.
- **A reach** along a direction is the finite tube's, from its centre, and across the axis it is the outer radius (the
  `inner_radius` plus the `wall`), not the lumen's.
- **The hydrodynamic wall** is a different entry: the [tube wall kind](../wall_kind/tube.md) is the wall's effect on the
  flow, read from a table of the unit tube's image system, and it reads the tube's centre, axis and radius from this
  structure. This page is the structure, not the wall's flow.

**When not to use it.**
- For a body: a tube has no mass properties and gives no samples.
- For a tube with ends, a bend or a varying radius: it is straight, of one radius, and its ends bound nothing.

**Its derivative** is the cylinder's: a kink in the distance and a jump in the normal on the axis. Within a band about
the axis, `AXIS_ULPS` units in the last place of the inner radius, the normal is zero with the derivative of the
offset from the axis, the projection across it; a standing test holds that Jacobian, and within the band the status
promises nothing.

**Regions, surfaces, directions and frames.** What a kind names and how a law or a stage reads it: [the scene note](../../scene.md).

**Versions.** 0.1.0 is its first.
