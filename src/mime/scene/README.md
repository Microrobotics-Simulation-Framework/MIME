# `mime.scene`

**The scene layer: a USD stage.**

## What goes here

- **`Scene.new(up=..., units=...)`**, which builds the scene Python-first. Later, `Scene.open(...)` reads an existing
  stage, once OpenMaterica's USD support lands.
- **Frames.**
  - Up is declared once, and gravity is derived from it.
  - There are no default axes.
  - Every direction is stated once and referred to after that (`tube.axis`).
  - The build report has one orientation line per thing.
- **Prim paths.** These are experiment names, recorded in the saved form.
- **Analytic geometry kinds**, such as `Tube` and `Box`. A `Box` writes OpenMaterica's oriented box primitive.
- **`StageIdentity`**, which pins a stage.
- **The stage writer.** It writes Z-up, metres, `/World` and a `defaultPrim`. Robot part prims carry a small `mime:`
  applied schema for design data, such as a magnet's magnetisation direction.
- **The geometry rule.** Reads of geometry are static unless declared `moving`. A read that would silently go stale
  is refused instead.

## Boundaries

- With `viz`, the only importer of `pxr`.
- It speaks generic USD only. Bridges to particular platforms live outside MIME.

## Status

Empty. The stage writer and frames land in dev1's phase 6.

Phases and releases are explained in the [package manifest](../README.md#status).
