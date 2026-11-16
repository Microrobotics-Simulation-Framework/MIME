- **`.to()` takes a unit written as text, and `mN` and `mm2` are units** (G18). `series.values.to("N")`, `.to("mN")`,
  `.to("mm/s")` and `.to("kg m/s^2")` read a unit as `--override` writes one (a name of `mime.units`, the symbol it prints
  with, or a label of SI base units); text that is none of them is refused as `units.label`, naming the nearest units.
  `mime.units.mN` (millinewtons) and `mime.units.mm2` (square millimetres) join `cm` and `um2`.
