---
describes: "0.1.0"
---
# `tube_tree`

A vessel as a tree of straight tubes: its `points` (three lengths each), the `segments` joining two of them by index,
each segment's `radii` at its start and its end (two lengths each, the radius linear between them: the tube is the
hull of its two end balls), a `fillet` on every edge the solid has between two walls (its radius a share of the
smallest radius at a junction), and the names of its free `ends`. It names two regions, the `lumen` and the `wall`;
the `lumen`'s shape is the tree's union of tubes, its signed distance exact at every junction (the seam-aware rule of
`mime.scene.ExactUnion`), not the minimum of the tubes' distances, which reads a false wall at each seam. A medium
fills it: `Medium("water", fills=vessel.lumen)`.

<!-- mime-fragment: a declaration's shape, the tree's settings alone; tests/flow/test_fvm_tree_declaration.py builds one -->
```python
vessel = TubeTree(points=mm((0, 0, -3, 0, 0, 0, 1.5, 0, 2.6, -1.5, 0, 2.6)),
                  segments=((0, 1), (1, 2), (1, 3)), radii=mm((0.5, 0.5, 0.4, 0.4, 0.3, 0.3)))
```

What a consumer reads of it, by what it is for:
- **A medium's flow** on the FVM (`FVM(domain=PeriodicBox(length=...))`) reads the lumen's exact distance where the
  wall's band reads it; the tree is periodic along z with the box's period and reaches past its two ends.
- **Its ends** are named by geometry, never by the role a law gives them: the first segment's start is `root` where
  no other segment meets it, and every other free end takes the name `ends` gives it, in the order of its points,
  `end_0`, `end_1`, ... by default. Each is a surface of the structure (`vessel.root`) and an opening of its lumen.
- **Its fillet** is refused where its closing would fuse two tubes that share no point (the solid between them thinner
  than its diameter): the refusal names a smaller fillet or a thicker wall.
- **Its centreline** is a segment a tube and a junction where two ends or more meet.

**When not to use it.**
- For a curved vessel drawn with many short segments: its distance evaluates every tube at every point, so its cost
  grows with the segments; a smooth curve has no kind of its own yet.
- For a body: a tree has no mass properties and gives no samples, no drawing and no surface a contact holds.

**Its derivative** is not verified as a kind's (see its gradient record): the lumen's distance has a kink where two of
its candidates tie. A flow's derivative through the band in a tree's radius is held to a central difference by the
FVM's gate.

**Regions, surfaces, directions and frames.** What a kind names and how a law or a stage reads it: [the scene note](../../scene.md).

**Versions.** 0.1.0 is its first.
