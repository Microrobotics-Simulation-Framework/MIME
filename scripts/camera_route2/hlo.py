"""Route 1's compiled programs, their text with source locations stripped, as uint8 arrays in an NPZ (bit_compare.py
compares them base against head, the command run from each tree's root): python scripts/camera_route2/hlo.py PROGRAM OUT.npz, PROGRAM one of step32, step64, run32, run64,
jvp32, jvp64, render32, render64. BUMP=1 moves the camera one ulp (of the program's dtype) along y (the control: every program's text differs)."""

from __future__ import annotations

import math
import os
import re
import sys

program, out = sys.argv[1], sys.argv[2]
dtype = "float64" if program.endswith("64") else "float32"
import jax  # noqa: E402

jax.config.update("jax_enable_x64", dtype == "float64")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "tests/sensing")
BUMP = os.environ.get("BUMP") == "1"
_one = np.float64 if dtype == "float64" else np.float32  # one ulp in the program's own dtype
Y = float(np.nextafter(_one(-1e-3), _one(0.0))) if BUMP else -1e-3


TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")


def text(compiled) -> np.ndarray:
    """The compiled text with every source location stripped: the metadata of each instruction and the tables of file
    names (whose paths differ between two checkouts), function names, locations (whose lines move with any edit of the
    source) and stack frames."""
    kept, skipping = [], False
    for line in compiled.as_text().split("\n"):
        if line in TABLES:
            skipping = True
        elif skipping and line == "":
            skipping = False
        elif not skipping:
            kept.append(line)
    stripped = re.sub(r", metadata=\{[^}]*\}", "", "\n".join(kept))
    return np.frombuffer(stripped.encode(), dtype=np.uint8)


if program.startswith("render"):
    from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render

    s = math.sqrt(0.5)
    lens = Pinhole(64, 48, 900.0, 900.0, 31.5, 23.5)
    pose = CameraPose(jnp.asarray([0.0, Y * 30.0, 0.0]), jnp.asarray([s, -s, 0.0, 0.0]))
    rod = Subject("rod", "cylinder", {"radius": 0.31e-3, "length": 2.07e-3, "axis": (1.0, 0.0, 0.0)},
                  appearance=Appearance.opaque(), moving=True)
    ball = Subject("ball", "sphere", {"radius": 0.4e-3}, appearance=Appearance.attenuating(500.0), moving=True,
                   window=20)

    def frame(at):
        found = render([rod, ball], lens, pose, {"rod": (at, jnp.asarray((1.0, 0.0, 0.0, 0.0))),
                                                  "ball": (at + 1e-3, jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
        return found.expected

    at = jnp.asarray([0.3e-3, 0.0, 0.1e-3])
    compiled = jax.jit(lambda a: jax.jvp(frame, (a,), (jnp.ones_like(a),))).lower(at).compile()
    np.savez(out, text=text(compiled))
    sys.exit(0)

from in_loop_cases import LOOK_ACROSS, scene_without_camera  # noqa: E402

from mime.build._maddening import _held, differentiated_run, full_state, graph_params  # noqa: E402
from mime.observations import Camera, World  # noqa: E402
from mime.units import m, ms  # noqa: E402

exp = scene_without_camera()
exp.observe(Camera(detector=__import__("in_loop_cases").DETECTOR, mount=World(position=m((0.0, Y, 0.0)),
                                                                               orientation=LOOK_ACROSS),
                   frame_period=ms(3), absorbers=(("tube", 3000.0),)))
built = exp.build()
graph = built.graph
graph.reset_state()
state, params = full_state(graph), graph_params(graph)
held = _held(graph)
fed = held._resolve_external_inputs({})
if program.startswith("step"):
    step = held._build_step_fn()
    compiled = jax.jit(lambda st, p: step(st, fed, p)).lower(state, params).compile()
else:
    run = differentiated_run(graph, 6, (), (), checkpoint=False)

    def loss(fy):
        camera = {**params["nodes"]["camera:"], "fy": fy}
        final, _, _ = run(state, {**params, "nodes": {**params["nodes"], "camera:": camera}})
        return jnp.sum(final["camera:"]["expected"])

    fy = jnp.asarray(params["nodes"]["camera:"]["fy"])
    if program.startswith("run"):
        compiled = jax.jit(loss).lower(fy).compile()
    else:
        compiled = jax.jit(lambda f: jax.jvp(loss, (f,), (jnp.ones_like(f),))).lower(fy).compile()
np.savez(out, text=text(compiled))
