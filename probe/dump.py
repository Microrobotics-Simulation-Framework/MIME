"""One program's first steps, every recorded array, to an NPZ: ``python dump.py <program>:<dtype>[:ulp] <out> <tree>
<head>``, run from the tree under test. Programs: script A (``script_a``, saved), A0 closed by its controller as a
callback (``tracked_rpm``, saved) and inside the compiled step (``a0_graph``, the switch configurations'), the coil
array (``coil_array``, saved), the gantry magnet (``gantry_magnet``, saved) and ``examples/sphere_settling.py``. None
calls allocate() or the exact allocation. ``:ulp`` moves every leaf the step reads one ulp up in the head's tree only
(``mime.build._maddening.params_one_ulp_up``): the one-ulp control, whose arrays must then differ from the base's."""

import contextlib
import importlib.util
import os
import sys

program, out, tree, head = sys.argv[1:5]
name, dtype, *control = program.split(":")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", dtype == "float64")
sys.path.insert(0, os.path.join(os.getcwd(), "tests", "saved"))
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
import numpy as np  # noqa: E402

from mime.testing import first_steps  # noqa: E402

STEPS = 8


def make():
    if name == "a0_graph":
        from chain_cases import tracked_rpm, tracking

        return tracked_rpm("tracked_rpm_in_graph", mode="graph",
                           controller=tracking(orientation_feedback=True, iterations=1))
    if name == "sphere_settling":
        spec = importlib.util.spec_from_file_location("example_sphere_settling",
                                                      os.path.join(os.getcwd(), "examples", "sphere_settling.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.sphere_settling(10.0)
    from cases import CASES

    return next(c for c in CASES if c.name == name).make()


built = make().build()
moved = control == ["ulp"] and os.path.realpath(tree) == os.path.realpath(head)
if moved:
    from mime.build._maddening import params_one_ulp_up

    context = params_one_ulp_up(built.graph)
else:
    context = contextlib.nullcontext()
with context:
    history = first_steps(built, STEPS)
arrays = {f"{node}/{field}": np.asarray(value) for node, fields in history.items() for field, value in fields.items()}
np.savez(out, **arrays)
print(f"{program}: {len(arrays)} arrays{' (one ulp up)' if moved else ''}")
