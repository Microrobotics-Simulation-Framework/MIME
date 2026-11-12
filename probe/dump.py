"""One saved program's first steps, every recorded array, to an NPZ:
``python dump.py <case>:<dtype>[:ulp] <out> <tree> <head>``, run from the tree under test. ``:ulp`` moves every leaf
the step reads one ulp up (``mime.build._maddening.params_one_ulp_up``) in the head's tree only: the one-ulp control,
whose arrays must then differ from the base's."""

import os
import sys

program, out, tree, head = sys.argv[1:5]
name, dtype, *control = program.split(":")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", dtype == "float64")
sys.path.insert(0, os.path.join(os.getcwd(), "tests", "saved"))
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
import contextlib  # noqa: E402

import numpy as np  # noqa: E402

from cases import CASES  # noqa: E402
from mime.testing import first_steps  # noqa: E402

STEPS = 8
def porous_plug_above():
    """The saved porous plug at a permeability above the hand-over's line at D/h 8 (sqrt(K)/h 0.016, line 0.01): its
    route resolved two-way in both trees, so its program is one without a plug below the line."""
    from mime.dynamics import FVM, Scheme
    from mime.experiment import Experiment
    from mime.interactions import BodyForce, Darcy
    from mime.scene import PeriodicBox, Plug, Scene, Tube, Z
    from mime.things import Medium, Structure
    from mime.units import Pa_s, kg_per_m3, m2, m_per_s2, mm, ms

    experiment = Experiment("porous_plug", scene=Scene.new(up=Z, units="m"), dt=ms(20.0))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                             length=mm(10.0))))
    experiment.add_structure(Structure("clot", geometry=Plug(axis=+Z, radius=mm(0.5), length=mm(0.25))))
    flow = FVM(domain=PeriodicBox(length=mm(1.0), margin=4), cells_across=8, stokes=True, scheme=Scheme.Steady())
    experiment.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.declare("clot.permeability", m2(4e-12), reason="above the line at this spacing")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    experiment.attach(Darcy(), medium="water", structure="clot")
    return experiment


built = (porous_plug_above() if name == "porous_plug_above" else next(c for c in CASES if c.name == name).make()).build()
moved = control == ["ulp"] and os.path.realpath(tree) == os.path.realpath(head)
if moved:
    from mime.build._maddening import params_one_ulp_up

    context = params_one_ulp_up(built.graph)
else:
    context = contextlib.nullcontext()
with context:
    history = first_steps(built, STEPS)
# The fields this PR adds to a flow holding a porous plug are left out: the base has none of them.
NEW = {"porous_flux", "porous_seepage", "forced_flagged", "porous_series", "porous_not_finite"}
arrays = {f"{node}/{field}": np.asarray(value) for node, fields in history.items() for field, value in fields.items()
          if field not in NEW}
np.savez(out, **arrays)
print(f"{program}: {len(arrays)} arrays{' (one ulp up)' if moved else ''}")
