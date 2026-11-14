# SPDX-License-Identifier: Apache-2.0
"""Camera 6a's bit comparison: the compiled text of the in-loop camera's programs for a scene with no static mesh and no
drawn kind (the graph's step, the forward run's value and its forward-mode derivative in ``fy``), source locations
stripped, written as bytes to an NPZ for ``.github/probe/bit_compare.py``. A scene that traces no mesh must compile to
the programs it compiled to before the mesh was there.

    python .github/probe/bit_compare.py --base origin/release/v0.3.0 \\
        --programs step_f32 step_f64 value_f32 value_f64 jvp_f32 jvp_f64 ulp_f32 ulp_f64 \\
        --command 'python {head}/scripts/camera_mesh_bits.py {program} {out} {tree} {head}' --output probe/out/bit_compare.json

A program is ``<kind>_<dtype>``. ``ulp_<dtype>`` is the control: on the head's side only, the tube's attenuation (a constant
the maker bakes into the program) is one unit in the last place up **in the dtype the program runs in**, so that it
survives that dtype's rounding (``np.nextafter(np.float32(3000.0), np.inf)`` in float32: float64's next number rounds back
to 3000.0 in float32 and moves nothing), and the text must differ in both dtypes."""

from __future__ import annotations

import math
import os
import re
import sys

name, out, tree, head = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
kind, dtype = name.rsplit("_", 1)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", dtype == "f64")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mime import Body, Experiment, State, Structure  # noqa: E402
from mime.build._maddening import _held, differentiated_run, full_state, graph_params  # noqa: E402
from mime.dynamics import RigidInertial  # noqa: E402
from mime.observations import Camera, Detector, World  # noqa: E402
from mime.scene import Scene, Sphere, Tube, X, Z  # noqa: E402
from mime.things import Part  # noqa: E402
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um  # noqa: E402

attenuation = 3000.0
if kind == "ulp" and os.path.realpath(tree) == os.path.realpath(head):
    run_type = np.float64 if dtype == "f64" else np.float32
    attenuation = float(np.nextafter(run_type(3000.0), run_type(np.inf)))
    assert run_type(attenuation) != run_type(3000.0), "the control's change must survive the dtype's rounding"
half = math.sqrt(0.5)
exp = Experiment("ballistic", scene=Scene.new(up=Z, units="m"), dt=ms(1))
exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60), wall=um(10), length=um(480))))
exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=um((-40.0, 0.0, 15.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((2e-3, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(20)))}))
exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
exp.observe(Camera(detector=Detector(width=64, height=48, fx=300.0, fy=300.0, cx=31.5, cy=23.5),
                   mount=World(position=m((0.0, -1e-3, 0.0)), orientation=(half, -half, 0.0, 0.0)), frame_period=ms(3),
                   absorbers=(("tube", attenuation),)))
built = exp.build()
graph = built.graph
graph.reset_state()
state, params = full_state(graph), graph_params(graph)


def text(compiled: object) -> np.ndarray:
    return np.frombuffer(re.sub(r", metadata=\{[^}]*\}", "", compiled.as_text()).encode(), dtype=np.uint8)  # type: ignore[attr-defined]


held = _held(graph)
fed = held._resolve_external_inputs({})  # noqa: SLF001
step = held._build_step_fn()  # noqa: SLF001
run = differentiated_run(graph, 6, (), (), checkpoint=False)


def loss(fy):
    camera = {**params["nodes"]["camera:"], "fy": fy}
    final, _, _ = run(state, {**params, "nodes": {**params["nodes"], "camera:": camera}})
    return jnp.sum(final["camera:"]["expected"])


fy = jnp.asarray(params["nodes"]["camera:"]["fy"])
arrays = {}
if kind in ("step", "ulp"):
    arrays["step"] = text(jax.jit(lambda s, p: step(s, fed, p)).lower(state, params).compile())
if kind in ("value", "ulp"):
    arrays["value"] = text(jax.jit(loss).lower(fy).compile())
if kind == "jvp":
    arrays["jvp"] = text(jax.jit(lambda f: jax.jvp(loss, (f,), (jnp.ones_like(f),))).lower(fy).compile())
np.savez(out, **arrays)
print(name, {k: v.size for k, v in arrays.items()})
