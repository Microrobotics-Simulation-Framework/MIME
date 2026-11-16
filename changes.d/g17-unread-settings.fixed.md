- **A setting an entry declares in a form the registration does not read is refused at registration, never left out** (G17,
  from the 3-D tutorial's friction log). `half_angle: Annotated[Quantity, Angle]` was left out of the entry's settings with no
  refusal (`mime.units.Angle` was a type alias for what a rate counts, not a marker), so the entry's own
  `setting_problems` raised `KeyError` and the build reported a defect of the entry. `Angle` is now the marker of a plane
  angle, in radians (`deg(30)`, `rad(0.52)`; a bare number, a length and a number of turns are refused, as a
  dimensionless marker would take the bare number), and the old alias is `AngleCount`. Every other form the reader left out
  without a word is refused naming the setting: metadata that is no dimension, `Annotated[...] | None` and `Optional`
  around the `Annotated`, a `type` alias, a container of quantities, and a setting group in a container. Quantities in a
  container an entry checks itself are declared so, `Annotated[..., CheckedByEntry]` (the arm's `armature` and
  `actuators`), a declaration and not a check.
