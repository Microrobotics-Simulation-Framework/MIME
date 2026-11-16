# SPDX-License-Identifier: Apache-2.0
"""Camera 6a's costs through the real node: a closed tube-wall mesh of 99 328 triangles (256 around, 96 along, 72/80 um,
400 um) referenced as scenery and attenuating, over the in-loop camera's ballistic scene, against the same scene with none.
Usage: camera_mesh_cost.py <none|mesh|none-nf|mesh-nf> <float64|float32> <width> <out.json>. The ``-nf`` variants have a
frame period of a second (no frame in the second run's 30 steps): the base step's cost beside the frames', so that a
mesh's cost per frame is the second run's time less the ``-nf`` run's. Run on a runner, one variant a process, twice
(``run_second_s`` is the steady 30 steps, 10 frames; its ratio mesh over none is the cost a frame has)."""

import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", sys.argv[2] == "float64")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mime import Body, Experiment, State, Structure  # noqa: E402
from mime.build._maddening import _held, full_state, graph_params  # noqa: E402
from mime.dynamics import RigidInertial  # noqa: E402
from mime.observations import Camera, Detector, World  # noqa: E402
from mime.assets import asset  # noqa: E402
from mime.scene import Scene, Sphere, Tube, X, Z  # noqa: E402
from mime.things import Part  # noqa: E402
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um  # noqa: E402
from mime.viz import Mesh, write_mesh  # noqa: E402


def tube_mesh(inner, outer, length, around, along):
    """A closed annular tube along x of ``4 around along + 4 around`` triangles, wound outward (camera spike 2's and the
    first step's generator), as a triangle soup: its points are each triangle's corners, repeated exactly where shared."""
    phi = np.arange(around) * 2.0 * np.pi / around
    xs = np.linspace(-length / 2, length / 2, along + 1)

    def ring(r, x):
        return np.stack([np.full(around, x), r * np.cos(phi), r * np.sin(phi)], -1)

    tris = []
    nxt = np.roll(np.arange(around), -1)
    for k in range(along):
        for r, outward in ((outer, True), (inner, False)):
            a, b = ring(r, xs[k]), ring(r, xs[k + 1])
            t1 = np.stack([a, b[nxt], a[nxt]], 1) if outward else np.stack([a, a[nxt], b[nxt]], 1)
            t2 = np.stack([a, b, b[nxt]], 1) if outward else np.stack([a, b[nxt], b], 1)
            tris += [t1, t2]
    for x, outward in ((xs[0], True), (xs[-1], False)):
        o, i = ring(outer, x), ring(inner, x)
        t1 = np.stack([o, o[nxt], i[nxt]], 1) if outward else np.stack([o, i[nxt], o[nxt]], 1)
        t2 = np.stack([o, i[nxt], i], 1) if outward else np.stack([o, i, i[nxt]], 1)
        tris += [t1, t2]
    corners = np.concatenate(tris, 0)[:, [0, 2, 1]]
    return corners.reshape(-1, 3), np.arange(len(corners) * 3).reshape(-1, 3)


def experiment(variant: str, width: int, record: dict) -> Experiment:
    """The in-loop camera's ballistic scene at ``width``, with the mesh (``variant`` ``mesh``) or without."""
    half = math.sqrt(0.5)
    scene = Scene.new(up=Z, units="m")
    mesh_path = "/World/Vessel"
    absorbers = [("tube", 3000.0)]
    if variant.split("-")[0] == "mesh":
        points, faces = tube_mesh(72e-6, 80e-6, 400e-6, 8, 2)
        from mime.sensing._mesh import closed_problem, signed_volume, triangles_of

        record["triangles"] = int(len(faces))
        record["closed_problem"] = closed_problem(points, faces)
        record["signed_volume"] = signed_volume(triangles_of(points, faces))
        layer = write_mesh(Path(tempfile.mkdtemp()) / "vessel.usda", Mesh(points, faces), "Vessel")
        scene.reference(asset(layer), at=mesh_path)
        absorbers.append((mesh_path, 2000.0))
    scale = width / 160
    detector = Detector(width=width, height=width * 3 // 4, fx=750.0 * scale, fy=750.0 * scale, cx=(width - 1) / 2,
                        cy=(width * 3 // 4 - 1) / 2)
    exp = Experiment("ballistic", scene=scene, dt=ms(1))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60), wall=um(10), length=um(480))))
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((-50.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((2e-3, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(20)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    exp.observe(Camera(detector=detector, mount=World(position=m((0.0, -1e-3, 0.0)), orientation=(half, -half, 0.0, 0.0)),
                       frame_period=ms(1000 if variant.endswith("-nf") else 3), absorbers=tuple(absorbers)))
    return exp


def largest() -> Experiment:
    """The memory estimate's case: the largest arm's, the mesh at 640 x 480."""
    return experiment("mesh", 640, {})


def main() -> None:
    variant, dtype, width, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    record = {"variant": variant, "dtype": dtype, "width": width, "jax": jax.__version__}
    try:
        record["cpu"] = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                             if line.startswith("model name"))
    except (OSError, StopIteration):
        record["cpu"] = ""
    record["usable_cpus"] = len(os.sched_getaffinity(0))
    exp = experiment(variant, width, record)
    t0 = time.perf_counter()
    built = exp.build()
    record["build_s"] = time.perf_counter() - t0
    graph = built.graph
    graph.reset_state()
    node = graph.get_node("camera:")
    intrinsics = jnp.asarray(np.asarray(node.build_intrinsics()), dtype=jnp.zeros(()).dtype)
    inner = node._maker()[0].__wrapped__
    fresh = jax.jit(lambda *a: inner(*a))
    args = node._meshes(None, intrinsics.dtype)
    t0 = time.perf_counter()
    compiled = fresh.lower(intrinsics, *args).compile()
    record["make_compile_s"] = time.perf_counter() - t0
    compiled(intrinsics, *args).block_until_ready()
    t0 = time.perf_counter()
    structure = compiled(intrinsics, *args).block_until_ready()
    record["make_run_s"] = time.perf_counter() - t0
    held = _held(graph)
    state, params = full_state(graph), graph_params(graph)
    fed = held._resolve_external_inputs({})
    stepper = held._build_step_fn()
    t0 = time.perf_counter()
    lowered = jax.jit(lambda s, p: stepper(s, fed, p)).lower(state, params)
    record["step_lower_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    program = lowered.compile()
    record["step_compile_s"] = time.perf_counter() - t0
    record["step_text_chars"] = len(re.sub(r", metadata=\{[^}]*\}", "", program.as_text()))
    try:
        record["step_serialized_bytes"] = len(program.runtime_executable().serialize())
    except Exception as error:  # noqa: BLE001
        record["step_serialized_bytes"] = f"unavailable: {type(error).__name__}"
    t0 = time.perf_counter()
    built.run(ms(30))
    record["run_first_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    built.run(ms(30))
    record["run_second_s"] = time.perf_counter() - t0
    record["expected_mean"] = float(np.asarray(built.current_state()["camera:"]["expected"]).mean())
    print(json.dumps(record, indent=1))
    with open(out, "w") as f:
        json.dump(record, f, indent=1)


if __name__ == "__main__":
    main()
