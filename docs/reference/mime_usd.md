# The `mime:` USD schema

The attributes and applied schemas MIME writes on a stage's prims and reads from them, all in its own namespace. An
applied schema is written as a token of a prim's `apiSchemas`, with no plugin, so any USD reads and writes it; an
attribute is a custom attribute in the `mime:` namespace. The stage writer writes them on `scene/scene.usda` from the
declarations alone, and a build reads the design data the schema puts on a referenced asset's prims. The page built
from this source adds, for each name, the constant that names it in the code, and for each registered geometry kind the
settings it writes. The keys MIME writes in a layer's own metadata (`customLayerData`), `mime:recording` on a
recording's layer and `mime:visuals` on a visuals layer, are not prims' names and are not listed here.

Two checks hold this list to the code. `mime docs check` holds it to the constants `scene/_compose.py` and
`scene/_read.py` read a stage with: each is here. (`scene/_derived.py` and `viz/_visuals.py` read names from a stage too,
and are not scanned, so a name only they read is not required here.) `tests/saved/test_usd_reference.py` holds it to
the stages the fixtures write, in both directions: every name a fixture's stage authors is here, and every name here is
authored on some fixture's stage or is listed in `docs/ratchets/usd_read_only.txt` as one MIME only reads. That test
plans the fixtures, which needs their assets, so a run without them does not exercise the second check.

## A thing's prim

- `mime:thing`: the kind of the thing whose prim this is (`structure`, `medium`, `body` or `apparatus`), on every
  thing's prim, whether written by MIME or a prim an asset brings in and a body is bound to.
- `mime:name`: the thing's name in the experiment's document, beside `mime:thing`.
- `mime:fills`: on a medium's prim, the region it fills, written as the document writes it (`tube.lumen`).

## A geometry kind's prim

- `MimeGeometryAPI`: applied on a prim that is a geometry kind, MIME's or a referenced asset's; a prim that writes
  `mime:geometry` attributes without it is refused at build.
- `mime:geometry`: the registered name of the geometry kind the prim is.
- `mime:geometry:version`: the registered version of that kind.
- `mime:geometry:<setting>`: one of the kind's shape settings, in SI with its unit in the attribute's `customData`
  (`units`), on a kind with no native USD form; a whole number the kind holds (a screw's starts) is written as a count,
  of unit one. A kind with a native form writes that primitive's own attributes, in the stage's units, instead. A
  build reads a referenced asset's kind from the same attributes.
- `mime:representation`: on a kind's drawn mesh (its `proxy` child), what the mesh is, in words, for a reader of the
  stage.
- `mime:region`: on a region's own primitive below its kind's prim (the tube's lumen), the region's name.
- `mime:derived_from`: on a derived prim (a kind's drawn mesh, a region's primitive), the kind, the version and the
  parameters it was derived from, the parameters by the hash of their canonical form. A reader with only the stage
  tells a stale prim by it.
- `mime:derived_hash`: on a derived prim, the hash of what it holds as written (a mesh's points, counts and indices;
  a primitive's attributes and transform), by which a reader with only the stage tells an edited one.

## An apparatus's chain and parts

- `mime:segment`: on a chain segment's prim, the segment as its plan describes it.
- `mime:frame`: on a frame's prim below its segment, the frame's own name, which the stage binding maps to its prim.
- `mime:mount`: on a part's prim, where it is mounted, `<segment>.<frame>`, or the world.
- `mime:solid`: on a part's prim, the geometry kind of its solid.
- `MimeMagnetAPI`: applied on a part prim that carries a magnet's design data.
- `mime:magnet:direction`: the unit vector, in the part's own frame, along which the part is magnetised. Written by
  MIME on an apparatus's magnet part, and read from a referenced asset's part prim, which must apply `MimeMagnetAPI`.

## What the stage draws

- `mime:visual:file`: on a mesh visual of a chain's frame, the file its description names, never its bytes.
- `mime:visual:asset`: the manifest that lists the mesh (beside the description, or in an installed package), where
  one does, for the asset cache's overlay.
- `mime:visual:sha256`: the hash of the mesh file's bytes, beside the manifest's name.
- `mime:visual:role`: what the shape stands for on its frame: `visual`, `collision`, or `frames`.
- `mime:vector:name`: on the prim of a field a law reads at a body, the field's interface name.
- `mime:vector:value`: the field's value at the body, set by a recording or a live client, in `mime:vector:units`.
- `mime:vector:units`: the unit of `mime:vector:value`.
- `mime:vector:metres_per_unit`: how long the drawn arrow is per unit of the field.
