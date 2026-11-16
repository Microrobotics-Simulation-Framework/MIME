- **A surface may declare `display="mirrored"`** (G19). `SurfaceMesh.display` takes `mirrored` beside `revolved` and
  `extruded`: a symmetric part solved as a quarter or a half, drawn with its reflections across its planes of symmetry.
  The entry draws the reflections as it draws a revolved section, the stage writes `mime:display` as `mirrored`, and the
  solve holds the part alone; MIME declares no symmetry planes and reflects nothing itself.
