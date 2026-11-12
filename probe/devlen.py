"""The development length past a prescribed opening (the maintainer's inflow decision (A)): a steady Inflow into an open
straight tube (tests/flow/open_cases.py's open_node: D = nu = rho = 1, Re_D the mean speed), its face the developed
profile, IMEX Euler at 0.9 of its CFL line at 4U (twice the centreline's 2U: the start's transient), started from the profile laid along the tube and run
in chunks to steady state; read along z: the centreline's w (bilinear at the axis) and the section's largest difference
from its developed profile, against the developed state (their mean over the tube from 0.55 to 0.8 of its length).
Predictions: tests/flow/predictions_prescribed_inlet.json (gate_development_length).

``python probe/devlen.py --reynolds 300 --cells 16 --output probe/out/devlen_300_16.json``"""
import argparse
import json
import math
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "tests/flow")
sys.path.insert(0, "tests")
from open_cases import open_node, tube_resistance  # noqa: E402


def estimate(reynolds: float = 600.0, cells: int = 16):
    """The memory estimate's case (``mem_estimate.py --case``): the largest arm's tube as an experiment, in SI at the
    same Reynolds number and cells (a 1 mm tube of water, (4 + 0.05 Re) D long)."""
    from mime.dynamics import FVM, Scheme
    from mime.experiment import Experiment
    from mime.interactions import Inflow, Windkessel
    from mime.scene import Scene, Tube, VesselRegion, Z
    from mime.things import Medium, Structure
    from mime.units import Pa_s, Pa_s_per_m3, kg_per_m3, m3_per_Pa, m3_per_s, mm, s

    reynolds, cells = float(reynolds), int(cells)
    speed = reynolds * 1e-6 / 1e-3
    exp = Experiment("devlen", scene=Scene.new(up=Z, units="m"), dt=s(0.9 * 0.4 * (1e-3 / cells) / (2.0 * speed)))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                      length=mm(4.0 + 0.05 * reynolds))))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": FVM(domain=VesselRegion(), cells_across=cells,
                                                                           scheme=Scheme.IMEXEuler())}))
    exp.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(Inflow(steady=m3_per_s(speed * math.pi * 0.25e-6)), medium="water", surface="tube.start")
    exp.attach(Windkessel(proximal=Pa_s_per_m3(1e8), compliance=m3_per_Pa(1e-13), distal=Pa_s_per_m3(1e9)),
               medium="water", surface="tube.end")
    return exp


def lengths(series: np.ndarray, z: np.ndarray, line: float) -> float:
    """The largest z at which ``series`` (a relative difference along z) is above ``line``: the development length."""
    above = np.nonzero(series > line)[0]
    return float(z[above[-1]]) if above.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reynolds", type=float, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--settled", type=float, default=1e-7, help="the centreline's change over a chunk, relative")
    parser.add_argument("--budget", type=float, default=19_000.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = time.time()
    re, cells = args.reynolds, args.cells
    length = 4.0 + 0.05 * re
    speed = re
    flux = speed * math.pi / 4.0
    resistance = tube_resistance(length)
    rows = [[flux, 0.0, 0.0, 0.0], [resistance, 1.0 / resistance, resistance, 0.0]]
    probe = open_node(cells, length=length, parameters=rows, dt=1.0, convection=True)
    p = probe._step_constants(None)  # noqa: SLF001
    grid = probe.open_grid(p)
    fields = probe.fields(p, grid)
    face = np.asarray(probe.inlet_profile(p, grid, fields).at(jnp.asarray(0.0)), dtype=np.float64)
    h = float(grid.spacing)
    dt = 0.9 * 0.4 * h / (4.0 * speed)  # the start's transient passed 2U at Re 300 (run 43252920973)
    laid = np.broadcast_to(face[..., None], grid.shape(2)).copy()
    node = open_node(cells, length=length, parameters=rows, dt=dt, convection=True, initial_velocity_z=laid)
    p = node._step_constants(None)  # noqa: SLF001
    centre = np.asarray(node.constants()["tube_centre"], dtype=np.float64)[:2] / h - 0.5
    i, j = int(math.floor(centre[0])), int(math.floor(centre[1]))
    fx, fy = centre[0] - i, centre[1] - j

    def centreline(w: np.ndarray) -> np.ndarray:
        return ((1 - fx) * (1 - fy) * w[i, j] + fx * (1 - fy) * w[i + 1, j] + (1 - fx) * fy * w[i, j + 1]
                + fx * fy * w[i + 1, j + 1])

    stepper = jax.jit(lambda s: jax.lax.scan(lambda q, _: (node.advance(p, q, {}, dt), None), s, None,
                                             length=args.chunk)[0])
    state = dict(node.initial_state())
    before = centreline(np.asarray(laid))
    chunks, settled, change = 0, False, float("nan")
    while time.time() - start < args.budget:
        state = dict(stepper(state))
        chunks += 1
        w = np.asarray(state["velocity_z"], dtype=np.float64)
        if not np.all(np.isfinite(w)) or int(state["stopped_at"]) >= 0:
            break
        now = centreline(w)
        change = float(np.max(np.abs(now - before)) / np.max(np.abs(now)))
        before = now
        if change < args.settled:
            settled = True
            break
    w = np.asarray(state["velocity_z"], dtype=np.float64)
    z = np.arange(w.shape[2]) * h
    cl = centreline(w)
    far = (z >= 0.55 * length) & (z <= 0.8 * length)
    developed = float(np.mean(cl[far]))
    section = np.mean(w[..., far], axis=-1)
    along = np.abs(cl - developed) / abs(developed)
    profile = np.max(np.abs(w - section[..., None]), axis=(0, 1)) / float(np.max(np.abs(section)))
    out = {"reynolds_diameter": re, "cells_across": cells, "length_diameters": length, "dt": dt,
           "dtype": "float64" if jax.config.jax_enable_x64 else "float32",
           "steps": chunks * args.chunk, "settled": settled, "last_change": change,
           "stopped_at": int(state["stopped_at"]), "finite": bool(np.all(np.isfinite(w))),
           "developed_centreline_over_2U": developed / (2.0 * speed),
           "profile_centreline_over_2U": float(centreline(face[..., None])[0]) / (2.0 * speed),
           "mismatch_at_plane": float(cl[0] / developed - 1.0),
           "L_1_centreline": lengths(along, z, 0.01), "L_01_centreline": lengths(along, z, 0.001),
           "L_1_profile": lengths(profile, z, 0.01), "L_01_profile": lengths(profile, z, 0.001),
           "centreline_over_developed": [float(x) for x in cl[: int(min(len(cl), 2 * (0.6 * length) / h))] / developed],
           "z_step": h, "seconds": time.time() - start}
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(json.dumps({k: v for k, v in out.items() if k != "centreline_over_developed"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
