- **One `Shape` protocol, its signed distance required and the rest derived** (`mime.core.shapes`). A shape gives its
  exact signed distance and its parameter names; its outward normal (by autodiff, zero on its medial set, where the
  gradient's length falls below `KINK`: the switch `<name>.kink`, declared when it is registered), its axis (none) and
  the camera's sign function (`Shape.sign_function`: the distance, slope one) are derived unless it gives its own, and
  MIME's closed forms are those overrides. A solid whose every region is a registered shape derives its camera sign
  function and, where it places nothing, its drawing: its body's distance contoured by marching tetrahedra
  (`mime.core.shapes.contour`), a closed mesh within about a cell of its surface whose representation says it is a
  drawing. A shape registered without subclassing `Shape` keeps its own methods and takes the derived forms of those
  it lacks. The extensibility test: the toy plugin shape `toy.capsule` (`mime.testing.toys.ToyCapsuleShape` and its kind
  `ToyCapsule`) gives only a distance and is held, in contact, the camera and the drawing, to MIME's capsule in closed
  form; a derived normal turned a milliradian and a sign function offset a tenth of a pixel are each caught.
