"""Float32 on a PERIODIC tube tree at the default tolerance (the coordinator's ask on decision 63.3): does the solve
reach 1e-5 or run to its cap? The released tree_flow (steady Stokes, 0.1h, 12 across) as cases.py writes it, and
variants: the 0.01h interior; IMEX Euler with convection (a pulsing drive at Re about 20). Per run: each step's
report (iterations, true residual, at cap, stagnated) and the lumen flux; one JSON line a variant."""
import json
import sys
import time

sys.path.insert(0, "tests/saved")
sys.path.insert(0, "tests")
import jax

jax.config.update("jax_enable_x64", sys.argv[1] == "64")
import numpy as np

from mime.dynamics import FVM, Band, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.scene import PeriodicBox, Scene, TubeTree, Z
from mime.things import Medium, Structure
from mime.units import Hz, Pa_s, kg_per_m3, m_per_s2, mm, ms


def made(interior, imex, cells):
    exp = Experiment("tree_flow", scene=Scene.new(up=Z, units="m"), dt=ms(0.05 if imex else 5.0))
    tree = TubeTree(points=mm((0.0, 0.0, -0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.75, 0.0, 0.0, 1.5)),
                    segments=((0, 1), (1, 2), (2, 3), (3, 4)), radii=mm((0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5)))
    exp.add_structure(Structure("vessel", geometry=tree))
    flow = FVM(domain=PeriodicBox(length=mm(1.0), margin=4), cells_across=cells, stokes=not imex,
               scheme=Scheme.IMEXEuler() if imex else Scheme.Steady(), wall=Band(delta_in=interior))
    exp.add_medium(Medium("water", fills="vessel.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    if imex:
        exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0), amplitude=m_per_s2(0.5), frequency=Hz(10.0)),
                   medium="water")
    else:
        exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(0.01)), medium="water")
    return exp


for interior, imex, cells, span in ((0.1, False, 12, 20.0), (0.01, False, 12, 20.0), (0.1, False, 16, 20.0),
                                    (0.1, True, 12, 1.0), (0.01, True, 12, 1.0)):
    start = time.time()
    built = made(interior, imex, cells).build()
    node = built.graph.get_node("water.flow")
    result = built.run(ms(span), history=True)
    held = result.history["water.flow"]
    reports = np.asarray(held["solve_report"], dtype=np.float64)
    row = {"dtype": str(np.asarray(held["lumen_flux"]).dtype), "interior": interior, "imex": imex, "cells": cells,
           "tolerance": float(node.constants()["tolerance"]), "at_floor": bool(node.constants()["at_floor"]),
           "box": [int(node.constants()[f"cells_{x}"]) for x in "xyz"], "steps": len(reports),
           "iterations": reports[:, 0].tolist(), "residual": reports[:, 1].tolist(),
           "at_cap": int(reports[:, 2].sum()), "stagnated": int(reports[:, 3].sum()),
           "lumen_flux_last": float(np.asarray(held["lumen_flux"])[-1]), "seconds": time.time() - start}
    print(json.dumps(row), flush=True)
