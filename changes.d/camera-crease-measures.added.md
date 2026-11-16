- **Crease measures on the camera's sampled route** (`mime.sensing`; `Geometry.crease_measure`, `CreaseMeasure` with its
  `faces`; `Cylinder.crease_measure`; `mime.testing.crease_measure_problems`; dev2, camera PR 5b-i). A creased kind (a
  solid whose body is two or more regions' intersection) takes the sampled route with its crease measure: its caps
  exact, each branch of the march read with its other regions at its least, each corner cell the exact share of their
  composition (`composed_covered`). `camera.crease-unmeasured` now refuses only a creased kind without the measure (the
  box; the screw until its measure comes). The cylinder declares its measure; its default stays the closed form, and
  `route="sampled"` images it by the march (against the exact image within lines fixed before the gate's draws,
  `scripts/camera_route2/crease_line.py`: the scan's largest error at its 360 poses, by an exact numpy re-implementation
  of the scheme, times 1.25, a margin declared, not derived, which the drawn views test). A face narrower than two
  sub-pixels raises `thin_cap`. Its derivative is the rendered image's: each branch's read place carries the derivative
  of the image's own held point (the march's points and spacing as they move, the quartic's least through its live
  values), as a custom derivative whose value is the held point, so an image is bit for bit what it was (its
  forward program's structural count is 1.0 % larger with it, measured, its cause not traced); G17's derivative holds within the central difference's own error. The rendered image's own
  derivative error against the exact geometry is known and pinned, not removed: a sliver at the rod's end where a
  sub-ray passes within 0.0025 px of the side's axis (up to 14.9 px² a px), a ripple at caps turned under about 8
  degrees (up to 0.88; 0.49 px a px on G17's aggregates), thin caps by their flag wherever the axis passes; CI holds
  the ripple to a line by band of the turn (1.25 times the 360-pose scan's largest in the band) and the sliver to one
  line under 8 degrees (its largest there times 1.25, since one pose's sliver moves by about a quarter between
  machines) and by band above. Under 8 degrees the sliver's line is a machine-robust placeholder with a control of its
  own at one pose only (a sliver grown past about 4 px² a px at turn 6); the follow-up "creased image accuracy" (the
  route to exact) removes the sliver, after which these lines tighten. New switches:
  `camera.composed.{end,low,high,empty}`, `camera.sampled.body`, `cylinder.cap_side`. Known gap, recorded and not
  changed: camera PR 5a's `union_covered` reads 0.125 more than the exact share for a line repeated along a cell's side
  (a case of measure zero; `composed_covered` is exact there).
