# The scene: frames, regions and the stage

The scene is what an experiment says of its world: which way is up, what each structure's geometry names, and the USD stage
that records it. This note holds the definitions; the geometry kinds' pages
([`tube`](entries/geometry/tube.md), [`cylinder`](entries/geometry/cylinder.md), [`box`](entries/geometry/box.md)) say what each
declares, and [the multibody note](multibody.md) says how a chain's frames join the world.

## Frames

The world frame is declared once, on the scene: `Scene.new(up=+Z, units="m")`, given to `Experiment(scene=...)`.

- **Up is +Y or +Z**, the two axes USD's `upAxis` names, so the stage's own metadata says which way is up. Another
  direction raises `ValueError` where it is declared, and a saved document that declares one is refused at load
  (`frame.up-not-y-or-z`). There are no default axes: a scene with no up is refused (`frame.up-missing`).
- **Gravity is derived**: its magnitude is the experiment's `gravity=` (standard gravity by convention, which the report
  says), and it points along `-up`. `up=` given to the experiment is refused: up is the scene's.
- **`units`** (`m`, `cm`, `mm` or `um`) is the length unit the stage is written in; MIME computes in SI whatever it is.
- **A direction is stated once and referred to**: a structure's geometry declares its `axis`, and a setting refers to it
  (`tube.axis`) rather than repeating it.
- A **body's frame** has its origin at the body's centre of mass, and a part of a body is placed in it (`at=`). A segment's
  frames and the parts mounted on them are in the world through the chain. The build report has one orientation line per
  thing.

## Regions, surfaces and directions

A geometry kind (`Tube`, `Box`, `Sphere`, or a lab's, a registered `mime.registry.Geometry`) names its **regions**, its
**surfaces** and its **directions**, and a structure's handle gives each as a typed path from the handle:
`tube.lumen` (a region), `tube.inner_surface` (a surface), `tube.axis` (a direction). A name the kind does not have is an
`AttributeError` on the line that writes it, with the names it has.

- A **region** is what a medium fills: `Medium("water", fills=tube.lumen)`. A law reads a region only if the kind gives it a
  shape (`region_shapes`: a signed distance, an outward normal, the axis and a point on it); a region with an axis is taken
  to be uniform along it. A region without a shape is a name, which the stage can draw and which gives a law no shape to read: the
  tube's `wall` is one.
- A **surface** faces the region it bounds, and a contact binds one: `tube.inner_surface` faces the lumen, so the body is
  kept on the lumen's side.
- Geometry is **static unless an entry declares `moving`**: an entry that refers to the geometry of a thing that has a
  dynamics, and does not declare it follows geometry that moves, is refused (`geometry.moving-read`), since it would keep the
  geometry as it was at the start.

## The stage

The stage is a USD layer written from the experiment's declarations alone, so the same declarations give the same bytes;
the saved form keeps it as `scene/scene.usda`, with the layers of any asset it references below `scene/layers/`, and the
lock records its hash.

- `/World` is the only root prim the writer emits, and it is the `defaultPrim`: every other root prim is reserved for MIME's
  overlays (a ghost of another drag rung could sit at `/Compare/<rung>/<body>`). The layer carries the declared `upAxis`
  and `metersPerUnit`.
- **One prim per thing**, with `mime:thing` (its kind) and `mime:name` (its name in the document): `/World/<name>` for a
  thing written in Python, or the prim an asset is referenced at (`scene.reference(asset(path), at="/World/UMR")`) for a body
  bound to it. A prim path is an experiment name, recorded in the saved form beside the thing's.
- **Each geometry kind is a prim** with its kind and version recorded, placed by its transform. A kind with a native USD
  form (a sphere, a capsule, a cylinder) is that primitive; any other is an `Xform` with its shape settings in SI and its
  drawing as a mesh on a `proxy` child, marked with what it was derived from, so that an edit of the derived mesh is found.
- **A chain**: each segment is a prim below its apparatus's, each frame of the segment a prim below it placed in the world,
  and each part below the frame it is mounted on.
- **The document is the authority** when an experiment is written in Python, and the stage is derived from it. Reading an
  existing stage as the authority (`Scene.open`) is not built: it raises `NotImplementedError` and names the Python way, and
  an asset that authors a geometry kind on one of its prims is refused (`scene.stage-first`).
