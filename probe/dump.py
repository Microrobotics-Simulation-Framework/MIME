"""A released program's arrays for bit_compare.py: ``dump.py <program@dtype> <out>``, run in the tree it compares (its
tests/saved/cases.py): 20 ms, every recorded field at every step."""
import pathlib
import sys

program, out = sys.argv[1], sys.argv[2]
name, _, dtype = program.partition("@")
root = pathlib.Path.cwd()
sys.path.insert(0, str(root / "tests" / "saved"))
sys.path.insert(0, str(root / "tests"))
import jax

jax.config.update("jax_enable_x64", dtype == "float64")
import numpy as np
import cases
from mime.units import ms

by = {c.name: c.make for c in cases.CASES}
make = by.get(name) or getattr(cases, name)
result = make().build().run(ms(20.0), history=True)
arrays = {}
for node, fields in result.history.items():
    for field, value in fields.items():
        a = np.asarray(value)
        if a.dtype.kind in "fciub":
            arrays[f"{node}|{field}"] = a
np.savez(out, **arrays)
print(program, len(arrays), "arrays")
