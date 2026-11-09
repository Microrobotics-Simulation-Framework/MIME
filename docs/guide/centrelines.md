# Adding a centreline to your geometry

A slender body's drag along its length (`Drag.RFT`) reads the body's **centreline**: a curve by arc length, with a
point, a unit tangent and a frame across it at each arc length, and the filament's radius. MIME's `Helix`, `Capsule`
and prolate `Ellipsoid` declare theirs, and the straight `Tube` declares its axis as a vessel's. A body whose hull kind
declares none is refused when the experiment is planned (`drag.no-centreline`), and the refusal points here. This
page is how a lab adds one to a kind of its own, a custom geometry or a body that comes from an STL or CAD mesh.

## What a centreline is

A `mime.registry.Centreline` is a tree of `CentrelineSegment`s joined at named `Junction`s; one segment is the common
case. Each field of a segment is a function of the kind's parameters (its settings in SI, numbers or JAX arrays):

- `length(parameters)`, the segment's arc length in metres;
- `frame(parameters, s)`, at arc lengths `s` from its start, the points, the unit tangents and the normals of the
  rotation-minimising frame (the normal turns about the tangent only as the tangent turns, so it does not twist);
- `filament_radius(parameters)`, the one radius a slender law's coefficients take;
- optionally `radius(parameters, s)`, the local radius, and `wavelength(parameters)`, the arc length of one period
  of a periodic segment (a helix's turn), which Lighthill's coefficients read.

Write each one with JAX operations on the parameters, never as a number computed when the experiment is built: then a
study's derivative in a setting flows through the centreline, and a value a study moves is never stale.

## A kind of your own declares its curve

A lab's geometry kind (see [extending MIME](lab_author.md)) implements `centreline(settings)`. For a straight rod, a
segment from `mime.scene.straight`; for any curve, a `CentrelineSegment` with its own functions, as
`Helix` does (`src/mime/scene/_helix.py`):

<!-- mime-fragment: a method of a lab's kind; tests/scene/test_centreline.py builds a kind with this centreline and holds it to the contract -->
```python
from mime.registry import Centreline
from mime.scene import straight


def centreline(self, settings):
    """The rod's axis along its frame's x, from end to end."""
    return Centreline((straight(lambda p: (-p["length"] / 2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                lambda p: p["length"], filament_radius=lambda p: p["radius"]),))
```

`mime.testing.centreline_problems(kind, settings)` checks a kind's centreline against the contract: each segment's
tangent the derivative of its point in arc length, the frame's normal across the tangent and turning only as it turns,
and the junctions' ends meeting.

## A mesh: its skeleton as declared points

For a body that comes from an STL or CAD mesh, extract its centreline (a skeleton, for instance with VMTK's centreline
tools) as points along it, with the local radius at each, and declare them through
`mime.scene.polyline`: straight segments joined at junctions, the frame carried round each corner, exact in arc
length. A lab's kind that holds the mesh returns it:

<!-- mime-fragment: a method of a lab's kind holding a mesh; tests/scene/test_centreline.py builds a polyline and holds it to the contract -->
```python
from mime.scene import polyline


def centreline(self, settings):
    """The mesh's extracted skeleton, its points and radii in the kind's frame, metres."""
    return polyline(self.skeleton_points, radii=self.skeleton_radii)
```

The points are fixed into the program, as undeclared geometry is; a mesh moved by an exact transform (scaled, placed,
turned) carries them with it when the kind applies the same transform to them.

## Later: a derived asset

A centreline derived from a mesh by a builder, held as a derived asset with its record, is the route for meshes once
mesh kinds land; until then, the declared points are.
