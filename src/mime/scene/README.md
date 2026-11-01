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

Since phase 5, **solids**, the geometry kinds a body's part is: `Cylinder` (radius, length, axis),
`Ellipsoid` (semi-axes) and `Box` (edges), each a registered `mime.registry.Solid` with a closed-form
volume and inertia per unit mass about its centroid, in the part's frame. `Composition.placement` gives a bound
part's origin and rotation in its body's frame from the asset's transforms, and refuses one that scales or shears.

Since phase 7d, **solid kinds with run-time shapes**: `Sphere` (radius), `Capsule` (radius, height between its caps'
centres as USD's `Capsule` counts it, axis) and `Screw` (de Jongh's screw: radius, length, wavenumber,
modulation and starts, along its frame's x and centred on its centroid, the earlier line's body moved half its length
with its turn kept). Each gives its surface samples by named patch, its area, family parameters and radial reach, and
the helix its thread as a tangent; a body whose one bulk part is one of them takes its extent from that reach, and its
shape values from the kind, never declared beside it. The tube reads `piecewise_jump`: its in-band derivative is held
by a standing test.

Since phase 7a2, the stage holds an apparatus's chain: each segment a prim below the apparatus's, placed at its
base's pose at the start, each frame a part is mounted on below its segment's, and each part below that frame with its
authored transform, a magnet part with `MimeMagnetAPI` and its direction. A solid gives the radius of the smallest
sphere about its centroid that holds it (`enclosing_radius`).

Since phase 7e, **the USD side of the geometry kinds**:

- **One authority per authoring mode.** Python-first, the document is the authority and the stage is derived from it,
  its hash in the lock. Stage-first, where the prim is the authority, comes with OpenMaterica's USD support:
  `Scene.open` raises `NotImplementedError` naming the Python-first way, and an asset that authors a geometry kind on
  one of its prims is refused (`scene.stage-first`).
- **Every geometry kind is a prim** with `MimeGeometryAPI` (detected through `GetPrimTypeInfo`, as `MimeMagnetAPI` is),
  `mime:geometry` and `mime:geometry:version`: a structure's, and each part's of a body written Python-first, below the
  body's prim (such a body is shown by its parts, and by a sphere of its extent only when it has none). Its placement
  settings (`placement`: a centre is its prim's origin, an axis its prim's +Z) are its transform, never attributes.
- **Native forms**, by kind (`native`, a `mime.registry.Native`): `Sphere` as USD's `Sphere`, `Capsule` as its
  `Capsule` (whose `height` excludes the caps, as the kind's does), `Cylinder` as its `Cylinder`, each with its
  native attributes the parameters in the stage's units and no copy among `mime:geometry:<setting>`. The helix, the
  tube, the box and the ellipsoid have none: an `Xform` with their shape settings in SI.
- **Derived prims.** A kind with no native form is drawn (`drawing(settings, segments)`, in its own frame, at the
  writer's resolution, `DRAWING_SEGMENTS` facets a turn) on its `proxy` child, purpose `proxy`; a region whose
  primitive is a sphere or a cylinder has a prim of its own below its kind's, the primitive's USD form (the tube's
  lumen: the lumen region clipped at the tube's length, as the region itself has no ends). Each carries
  `mime:derived_from` (the kind, version and its parameters' hash) and
  `mime:derived_hash` (what it wrote). `derived_findings(stage)` reports each one edited or left stale, on any composed
  stage; `stage_edits` compares a stage with the one its declarations write.
  *What a reader with only the stage sees of a derived prim* is: the content hash (a mesh's points, counts and indices;
  a primitive's radius, height, axis, extent, purpose, visibility and transform ops), a proxy's form (purpose `proxy`,
  right-handed, no subdivision, not hidden, no transform of its own), and that its source's parameters still hash to
  what it records. It does not see a proxy removed (nothing marks that one is expected), a mesh and its recorded hash
  rewritten together, or an attribute neither covers (a mesh's display colour and opacity, its `extent`, `doubleSided`,
  normals and primvars, `mime:representation`). A reader with the stage, the declarations and the kind's version
  registered sees all of these through `stage_edits`, as the load does; a reader with the kind alone could redraw
  `drawing(settings, segments)` from the prim's recorded parameters and segments and compare, which needs the drawing
  to be bit-reproducible across platforms (`np.sin` and `np.cos` on arrays: not checked), so it is not done.
- `read_kind(prim, meters_per_unit)` reads a kind back from its prim, its native attributes or its settings and its
  transform, and its counts (the helix's starts): the round trip's half. Nothing in `src` calls it yet, so its four
  `geometry.*` refusals are raised only by the tests that call it: it is the seam a2's stage-first reader builds on.
  `kind_frame`, `placed` and `shape_settings` carry a kind's own frame.
- **The proxy's size.** The helix's proxy is nearly all of script A's saved stage and grows with the
  square of the drawing's segments (its sections scale with them); the resolution is a writer setting, not yet a
  field of the document, and when it becomes one, its cost in the saved folder is a reason to choose it.

Since phase 7m-a (designs #195, #196), **the geometry protocol's sampling and hull query**:

- **`Geometry.sampling(settings, spacing, patches, density=uniform)`** samples a kind's surface for a solver whose
  unknowns sit at the samples (the MRS rung, which takes it by default and refuses a kind without it by name): a
  `mime.registry.Sampling`, a choice among the kind's quadrature samples (`samples` at counts) fixed when the
  experiment is built, so the samples stay a JAX function of its parameters. `mime.scene.quasi_uniform` gives it to any
  kind with a quadrature, MIME's or a lab's: as many samples as the quadrature has at the spacing, chosen from it at
  `REFINEMENT` times finer by farthest points (ties within `TIE` of the body's size broken by index, so a choice is
  the same on a machine that rounds the dense points' last bits otherwise) and `LLOYD_STEPS` of Lloyd's steps, each
  sample a quadrature sample with its cell's area as its weight. A `density` (a function of points, one where the
  samples are the spacing apart) grades it with no protocol change. The choice is made in float64 whatever the run's
  precision, and cached in the process.
- **The choice is not continuous in the kind's settings**, and is not meant to be: it is discrete (each sample one of
  the quadrature's, chosen by farthest points), so a small change of a setting can choose other samples, and a result
  then steps, by far more than its smooth change with the choice held (MIME-ANO-127 holds the measured steps). The grid
  kept from before keeps its count and moves smoothly, and the sphere's choice scales with its radius exactly, so it
  does not step. Within one build the choice is held (the MRS node holds it as it holds its counts), so a derivative is
  the discretisation's, exact; what sees the steps is a study that builds again at moved settings: an optimisation's
  outer loop, a sweep of rebuilt experiments, a difference taken across two builds. Nothing public yet carries one
  build's choice into another (the rebuild-by-difference checks hold it with a test's helper), so such a study takes
  its derivatives within one build, or reads the steps as the discretisation's error. Carrying the choice, or a
  sampler continuous in its settings (Lloyd's steps with an exact projection, which the helix score tried), is a
  design choice not made here.
- **Every solid kind samples**: the sphere 0.2.0, the capsule 0.2.0 and the screw 0.3.0 quasi-uniformly (their
  quadratures crowded: a quarter of the sphere's samples within half the mean spacing of another, the screw's caps
  several times their share of the area); the cylinder, the box and the ellipsoid gain a quadrature (the cylinder's caps in
  rings, `_caps`; the box's faces by Gauss-Legendre; the ellipsoid's by Gauss-Legendre in the polar angle's cosine)
  and the same sampling, at their versions (nothing of theirs moved). The versions superseded, the sphere 0.1.0, the
  capsule 0.1.0 and the screw 0.2.0, stay registered (`KEPT`, `_previous.py`), their sampling their quadrature, so
  that a folder saved with one loads and reproduces; a script that writes the class gets the current version.
- **`Geometry.reach(parameters, direction)`**, the hull query: the largest `x . d` over the body, homogeneous of degree
  one in `d`, its gradient the farthest point (the middle of a face, a segment or a rim where that is not one point).
  Closed forms for the sphere, the capsule, the cylinder, the box, the ellipsoid and the tube; an envelope for the
  screw, the cylinder of its crests, which may overstate its reach by up to `reach_envelope` of its enclosing radius
  between its axis and across it (a declaration from its supremum, MIME-ANO-128) and whose farthest point is the
  envelope's, not the body's; `sampled_reach` for a kind from its samples, and `body_reach`, the body's own reach from
  its quadrature at a spacing, which never overstates it and whose farthest point is a sample of the body (the point a
  contact acts at). Its kinks are not yet switches of
  `mime.core.switch` (`length_of`, `magnitude_of`): no node's step evaluates a reach until the containment contact
  (phase 7o), which declares them.
- A body's hull may now be a kind that samples but has no radial reach (a box, an ellipsoid): its extent is then not
  the kind's, and is declared if a law reads it.

Phases and releases are explained in the [package manifest](../README.md#status).
