- **Core's shapes are named for their role and live in `mime.core.shapes`** (CONTRIBUTING's rule 3, one name one
  concept): `SphereShape`, `CapsuleShape`, `CylinderShape` (unbounded), `SlabShape`, `HalfSpaceShape` and `ScrewShape`
  (unbounded along its axis), each a subclass of `Shape`; a script meets only the scene's `Sphere`, `Capsule`,
  `Cylinder` and `Screw`. The singletons (`CYLINDER`, `SCREW`, ...), `Shape`, `register_shape` and `shape_named` stay in
  `mime.core`; `SignValue` is `mime.core.shapes`'s, still exported by `mime.registry`. The sphere's, capsule's and
  cylinder's camera sign functions are no longer written again in the scene: each kind takes its regions' shapes'
  closed forms, so the box (three slabs) and a lab's solid whose regions are shapes (the lab fixture's bead) now have a
  sign function, and such a solid that draws nothing now has a contoured drawing.
