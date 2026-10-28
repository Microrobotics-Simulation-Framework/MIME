# `mime.scene`

**The scene layer: a USD stage.**

## What goes here

- **`Scene.new(up=..., units=...)`**, which declares the scene Python-first. Later, `Scene.open(...)` reads an
  existing stage, once OpenMaterica's USD support lands.
- **Frames.**
  - Up is declared once, on the scene, and gravity is derived from it (`-g * up`).
  - Up is +Y or +Z, the two axes USD's `upAxis` names, so the stage's own metadata says which way is up, as it does
    for a stage written elsewhere. Up is a choice of coordinates: a scene with another up is re-expressed with up
    along +Y or +Z (for example by swapping two axes), and importing data in another frame will be a declared frame
    transform.
  - There are no default axes.
  - Every direction is stated once and referred to after that (`tube.axis`).
  - The build report has one orientation line per thing.
- **Prim paths.** These are experiment names, recorded in the saved form beside the things' names.
- **Analytic geometry kinds**, such as `Tube` and `Box`. A `Box` writes OpenMaterica's oriented box primitive.
- **`StageIdentity`**, which pins a stage.
- **The stage writer.** It writes the declared up axis, `metersPerUnit`, `/World` and a `defaultPrim`. Robot part
  prims carry a small `mime:` applied schema for design data, such as a magnet's magnetisation direction.
- **The geometry rule.** Reads of geometry are static unless declared `moving`. A read that would silently go stale
  is refused instead.

## Boundaries

- With `viz`, the only importer of `pxr` (a test holds it to that). usd-core ships no type stubs, so its modules are
  bound once, as `Any`, in `_pxr.py`.
- It speaks generic USD only. Bridges to particular platforms live outside MIME.

## Status

Since phase 4b, the analytic geometry kind **`Tube`** (`tube` 0.1.0): axis, inner radius, wall thickness, length
and centre; regions `lumen` and `wall`, the surface `inner_surface` (facing the lumen, evaluated as an unbounded
cylinder, so a contact with it bounds a body radially and its ends bound nothing), and the direction `axis`, each a
typed path from the structure's handle. A geometry kind is a contract a lab implements: a registered
`mime.registry.Geometry` naming its regions, surfaces and directions, its surfaces and some of its regions
(`region_shapes`) evaluated by a `Shape` (the protocol: signed distance, outward normal, the axis and a point on it)
registered in `mime.core`. `X`, `Y` and `Z` are signed axes (`+X`, `-Z`).

Since phase 6:

- **`Scene.new(up=+Z, units="m")`**, given to `Experiment(scene=...)`. `up` is +Y or +Z (any other direction raises
  `ValueError` at the declaration, and a saved document declaring one is refused at load, `frame.up-not-y-or-z`);
  `units` one of `m`, `cm`, `mm`, `um` (MIME computes in SI whatever it is). `scene.up` is a signed axis, so
  `-scene.up` is down.
- **`scene.reference(asset(path), at="/World/UMR")`** references an asset layer's default prim at a prim directly
  below `/World`. The build reads the file once and pins it by the SHA-256 of its bytes (a pin given with
  `asset(path, sha256=...)` refuses other bytes), and refuses an asset with no `defaultPrim`, with no or another
  `metersPerUnit` than the scene's, or composing other files. `scene["/World/UMR"]` names a prim for a thing to bind.
- **The stage**, written by `stage_text` from the build's plans and saved as `scene/scene.usda`, its assets'
  bytes under `scene/layers/`:
  - `upAxis` is the declared up, and `/World`, the `defaultPrim`, carries no transform. `metersPerUnit` is the
    scene's unit.
  - One prim per thing, with `mime:thing` (its kind) and `mime:name` (its name in the document): `/World/<name>` for
    a thing written Python-first, and overs on the bound prim for a body bound to an asset's prim.
  - A geometry kind's `drawing` (a `mime.registry.Drawing`) is its mesh: the tube is its wall, a closed annular
    mesh of 64 segments in its own frame (axis along +Z), with no subdivision. The kind's settings are written in SI
    as `mime:geometry:<setting>`, each with its unit in `customData`, with `mime:geometry` and
    `mime:geometry:version`: the stage records what the physics used.
  - A body stands at its initial `position` and `orientation`; a body written Python-first is shown as a sphere of
    its declared extent.
  - USD stores points as float32: a mesh's points stay in its own frame, near its origin, and its place is written
    as double-precision `translate` and `orient` ops. Display colours and opacities are primvars, with no material.
  - The same declarations write the same bytes; the lock records the stage's hash and each layer's.
- **The `mime:` applied schema**: `MimeMagnetAPI` in a part prim's `apiSchemas` and `mime:magnet:direction`, a unit
  vector in the part's frame, the direction the part is magnetised along. It is the smallest form usd-core reads
  without a plugin: the schema's name in the prim's `apiSchemas` and its namespaced attribute, with no compiled or
  codeless schema registered. The attribute without the schema, or the schema without the attribute, is refused.
- **`check_stage(path)`** runs every validator usd-core registers (its `UsdValidation` framework, which `usdchecker`
  runs; the older Python `ComplianceChecker` is gone from usd-core 26) and returns their findings.
- The world frame (`WorldFrame`, `world_frame`) moved here from the build.

Since phase 4c1, the tube's `lumen` is a region with a shape, the inside of the cylinder: a law reads the region a
medium fills (`Medium(..., fills=tube.lumen)`) through the protocol, as the drag reads the tube's axis, a point on it
and its radius (minus the signed distance there). A region with an axis is taken to be uniform along it.

Since phase 5, **solids**, the geometry kinds a body's part is: `SolidCylinder` (radius, length, axis),
`SolidEllipsoid` (semi-axes) and `SolidBox` (edges), each a registered `mime.registry.Solid` with a closed-form
volume and inertia per unit mass about its centroid, in the part's frame. `Composition.placement` gives a bound
part's origin and rotation in its body's frame from the asset's transforms, and refuses one that scales or shears.

Since phase 7a2, the stage holds an apparatus's chain: each segment a prim below the apparatus's, placed at its
base's pose at the start, each frame a part is mounted on below its segment's, and each part below that frame with its
authored transform, a magnet part with `MimeMagnetAPI` and its direction. A solid gives the radius of the smallest
sphere about its centroid that holds it (`enclosing_radius`).

Phases and releases are explained in the [package manifest](../README.md#status).
