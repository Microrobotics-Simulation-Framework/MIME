"""Where a confined step on the MRS rung spends its time (the profiling pass's stage 2b), on drag 0.4.0 at its
defaults, on whatever device JAX runs on; and the integrated wall image's (``wall_refinement=2``, W2) structure and
cost against the image at the force points.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/mrs_step_split.py <out.json> split floor|axis [--repeats N]
    uv run --no-sync python scripts/mrs_step_split.py <out.json> w2 <spacing_mm> [--chunk-values P ...] [--repeats N]

Not collected by pytest; run by hand. The tube's cases read the asset ``unit_tube_R0.997`` (``mime assets fetch`` it
first). Every time is wall-clock, the least of ``--repeats`` after a first call that compiles, in the precision the
process runs in (float32, or float64 under ``JAX_ENABLE_X64=1``).

- ``split floor``: FL-9 resting on the floor of the 1/4 in tube (script A's operating point, every kind and law at its
  current version, ``Drag.MRS(wall_kind="tube")`` with nothing written: 0.181 mm, blob 0.25, the quadrature eight
  times finer), at the floored pose 1.0002 mm below the axis; ``split axis``: ``scripts/default_spacing_table.py``'s
  T2, FL-9 on the tube's axis at the rung's default. Each piece of the step jitted alone, in the step's order: the
  body's samples placed; the free part's matrix, held in the body's frame since the build, turned to the pose; the
  wall's image at the force points, each column times its cell's area; their sum; the solve with the wrench; the
  judging (the eigenvalue, the asymmetry, the gap). Then the node's whole update, XLA's account of the image's and
  the update's temporaries, the graph's step in a run (``RUN_STEPS``: MADDENING's scan over steps, whose body XLA
  compiles as a loop), and the same graph with the drag's step returning its state (what the rest of the graph costs).
- ``w2``: FL-9 on the floor at a written spacing with ``wall_refinement=2``: the cells the step integrates (its
  budget), the widest a cell is padded to, the chunks its integration is cut in and their size, and the table reads
  against the image at the force points'; the two images timed at the floored pose. With ``--chunk-values P``, W2
  again with its chunks cut for ``P`` floating values a pair (``Chunks.of``'s ``pair_values``) in place of the
  table's ``TABLE_PAIR_VALUES``, and whether it is W2 as it stands bit for bit (it is not where a cell's points fall
  in two chunks: their sums are added in other groups).
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

#: The steps of the runs the graph's step is timed on, by case: a step on the floor takes tens of seconds on a CPU.
RUN_STEPS = {"floor": 2, "axis": 200}
#: FL-9's centre below the tube's axis on the floor: the tube's radius less FL-9's extent (2.0748 mm) and the contact's
#: margin (0.1 mm), script A's floored pose.
FLOORED_OFFSET = 1.0002e-3


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def floor_experiment(**settings: Any) -> Any:
    """Script A's operating point with every kind and law current and the MRS rung at drag 0.4.0's defaults (or at
    ``settings``): FL-9 in the 1/4 in tube under gravity, a 10 Hz field, the contact's 0.1 mm margin."""
    import script_a

    from mime import Apparatus, Body, Experiment, Medium, State, Structure
    from mime.dynamics import RigidInertial
    from mime.fields import UniformRotating
    from mime.interactions import Contact, Drag, Gravity, Magnetic
    from mime.observations import SwimSpeed
    from mime.scene import Scene, Tube, X, Z
    from mime.things import Part
    from mime.units import A_m2, Hz, kg_per_m3, m, m_per_s, m_per_s2, mm, mPa_s, ms, mT, rad_per_s

    e = Experiment("step_split", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))
    tube = e.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                           length=mm(100.0))))
    water = e.add_medium(Medium("water", fills=tube.lumen))
    umr = e.add_body(Body("umr", dynamics=RigidInertial(), parts={"shell": Part(script_a.fl9())}, initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    field = e.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotating(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(10.0), "amplitude": mT(1.2)}))
    for path, value in (("umr.inertia", script_a.INERTIA), ("umr.mass", script_a.MASS),
                        ("umr.buoyant_volume", script_a.VOLUME),
                        ("umr.magnetic_moment", A_m2((0.0, 2 * 8.4e-4, 0.0))), ("water.density", kg_per_m3(1000.0)),
                        ("water.viscosity", mPa_s(1.0))):
        e.declare(path, value, reason=script_a.REASON)
    e.declare("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0)), reason="one density: the centre of mass")
    e.attach(Drag.MRS(wall_kind="tube", **settings), body=umr, medium=water)
    e.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    e.attach(Gravity(), body=umr, medium=water)
    e.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    e.observe(SwimSpeed(umr, along=tube.axis))
    return e


def axis_experiment() -> Any:
    return _load("_split_default_spacing_table", ROOT / "scripts" / "default_spacing_table.py").screw_task(None)


def pose_of(case: str, dtype: Any) -> Any:
    import jax.numpy as jnp

    return jnp.asarray([0.0, 0.0, -FLOORED_OFFSET if case == "floor" else 0.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)


def timed(function: Callable[..., Any], *arguments: Any, repeats: int) -> tuple[dict[str, float], Any]:
    """The first call (its compile included) and the least of ``repeats`` more, in seconds, and the result."""
    import jax

    started = time.perf_counter()
    found = jax.block_until_ready(function(*arguments))
    first = time.perf_counter() - started
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        found = jax.block_until_ready(function(*arguments))
        times.append(time.perf_counter() - started)
    return {"first_s": first, "least_s": min(times), "median_s": sorted(times)[len(times) // 2]}, found


def held_bytes(function: Callable[..., Any], *arguments: Any) -> dict[str, int]:
    """XLA's account of a program: its arguments, its result and its temporaries, in bytes."""
    import jax

    analysis = jax.jit(function).lower(*arguments).compile().memory_analysis()
    return {name: int(getattr(analysis, f"{name}_size_in_bytes")) for name in ("argument", "output", "temp")} \
        if analysis is not None else {}


def unchecked_build(experiment: Any) -> Any:
    """``experiment`` built with the MRS node's start check lifted: a step's cost does not depend on whether its
    resistance is definite."""
    from mime.nodes.hydro.mrs import MRSDragNode

    held = MRSDragNode.built_problems
    MRSDragNode.built_problems = lambda self, start=None: []  # type: ignore[method-assign]
    try:
        return experiment.build()
    finally:
        MRSDragNode.built_problems = held  # type: ignore[method-assign]


def split(case: str, repeats: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from mime.build._maddening import node_state, parameter_leaves
    from mime.core import quaternion
    from mime.nodes.hydro.mrs import HELD, MRSDragNode
    from mime.nodes.hydro.mrs._node import judged, least_distance
    from mime.nodes.hydro.mrs._quadrature import separated_resistance, turned
    from mime.nodes.hydro.mrs._solve import PRECISION
    from mime.units import s

    make = floor_experiment if case == "floor" else axis_experiment
    started = time.perf_counter()
    built = make().build()
    found: dict[str, Any] = {"case": case, "build_s": time.perf_counter() - started}
    node = built.graph.get_node("drag:umr")
    assert isinstance(node, MRSDragNode) and node.separated
    leaves, state = parameter_leaves(built.graph)["drag:umr"], node_state(built.graph, "drag:umr")
    pose = pose_of(case, state["resistance"].dtype)
    p = node._step_constants(leaves)  # noqa: SLF001 - what the node's step reads
    wall = node._wall()  # noqa: SLF001
    parameters = {name: p[f"wall_{name}"] for name in wall.parameters}
    theta = {key: p[key] for key in node.body_leaf_names()}
    found.update({"force_points": node.sample_count, "quadrature_points": node.quadrature_count,
                  "unknowns": 3 * node.sample_count, "blob": float(node.constants()["blob"]),
                  "force_spacing_m": node.constants().get("force_spacing"), "dtype": str(pose.dtype)})

    def placed(theta: Any, pose: Any) -> Any:
        turn = quaternion.rotation_matrix(pose[3:])
        body = node.body(theta)
        return tuple(pose[:3] + jnp.matmul(x, turn.T, precision=PRECISION)
                     for x in (body.force, body.centroid, body.points)) + (body.area,)

    def free_turned(held: Any, pose: Any, viscosity: Any) -> Any:
        return turned(held, quaternion.rotation_matrix(pose[3:])) / viscosity

    def image(force: Any, area: Any, parameters: Any, viscosity: Any) -> Any:
        corrected = wall.correction(force, parameters, viscosity)
        return corrected.matrix * jnp.repeat(area, 3)[None, :]

    def judging(resistance: Any, points: Any, parameters: Any, pose: Any) -> Any:
        return judged(resistance, jnp.asarray(True), least_distance(wall.distance(points, parameters)), 1e-4, state,
                      pose)

    def update(leaves: Any, pose: Any) -> Any:
        return node.update(state, {"pose": pose}, 1e-3, params=leaves)

    pieces: dict[str, Any] = {}
    pieces["samples_placed"], (force, centroid, points, area) = timed(jax.jit(placed), theta, pose, repeats=repeats)
    pieces["free_part_turned"], free = timed(jax.jit(free_turned), p[HELD], pose, p["viscosity"], repeats=repeats)
    pieces["wall_image"], matrix = timed(jax.jit(image), force, area, parameters, p["viscosity"], repeats=repeats)
    pieces["sum"], matrix = timed(jax.jit(lambda a, b: a + b), free, matrix, repeats=repeats)
    del free
    pieces["solve_and_wrench"], resistance = timed(jax.jit(separated_resistance), matrix, force, area, centroid,
                                                   pose[:3], repeats=repeats)
    del matrix
    pieces["judging"], _ = timed(jax.jit(judging), resistance, points, parameters, pose, repeats=repeats)
    pieces["whole_update"], _ = timed(jax.jit(update), leaves, pose, repeats=repeats)
    found["pieces"] = pieces
    found["held_bytes"] = {"wall_image": held_bytes(image, force, area, parameters, p["viscosity"]),
                           "whole_update": held_bytes(update, leaves, pose)}
    print(json.dumps({key: found[key] for key in ("pieces", "held_bytes")}), flush=True)

    steps = RUN_STEPS[case]
    dt = float(built.graph.get_node("umr").delta_t)

    def per_step(run_built: Any) -> dict[str, float]:
        started = time.perf_counter()
        run_built.run(s(steps * dt))
        first = time.perf_counter() - started
        times = []
        for _ in range(max(1, repeats - 1)):
            started = time.perf_counter()
            run_built.run(s(steps * dt))
            times.append(time.perf_counter() - started)
        return {"steps": steps, "first_run_s": first, "least_per_step_s": min(times) / steps}

    found["run"] = per_step(built)
    held = MRSDragNode.advance
    MRSDragNode.advance = lambda self, p, state, inputs, dt: state  # type: ignore[method-assign]
    try:
        found["run_without_the_drag"] = per_step(unchecked_build(make()))
    finally:
        MRSDragNode.advance = held  # type: ignore[method-assign]
    return found


def w2(spacing_mm: float, chunk_values: list[int], repeats: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mime.core import quaternion
    from mime.nodes.hydro.mrs import Corrected, MRSDragNode
    from mime.nodes.hydro.mrs._node import near_wall_image, on_the_cells
    from mime.nodes.hydro.mrs._quadrature import Chunks
    from mime.units import mm

    built = unchecked_build(floor_experiment(spacing=mm(spacing_mm), wall_refinement=2))
    node = built.graph.get_node("drag:umr")
    assert isinstance(node, MRSDragNode) and node.wall_chunks is not None and node.wall_capped is not None
    assert node.wall_table is not None and node._wall_quadrature is not None  # noqa: SLF001
    wall = node._wall()  # noqa: SLF001
    p = node._step_constants(None)  # noqa: SLF001
    parameters = {name: p[f"wall_{name}"] for name in wall.parameters}
    dtype = jnp.asarray(p["viscosity"]).dtype
    viscosity = jnp.asarray(p["viscosity"], dtype=dtype)
    kept, runs, owners = node.wall_capped
    chunks, table, every = node.wall_chunks, node.wall_table, node.wall_every
    padded = int(chunks.points.size)
    found: dict[str, Any] = {
        "spacing_mm": spacing_mm, "force_points": node.sample_count, "kept_wall_points": int(len(kept)),
        "budget": node.wall_budget, "widest": int(table.shape[1]), "every": bool(every),
        "chunk_points": int(chunks.points.shape[1]), "chunks": int(chunks.points.shape[0]), "dtype": str(dtype),
        "reads_against_the_point_image": 2 * padded / node.sample_count}

    theta = {key: jnp.asarray(p[key], dtype=dtype) for key in node.body_leaf_names()}
    body = node.body(theta)
    cells = node._quadrature_samples({**theta, "viscosity": theta["blob"]},  # noqa: SLF001
                                     node._wall_quadrature[0])  # noqa: SLF001
    pose = pose_of("floor", dtype)
    turn = quaternion.rotation_matrix(pose[3:])
    force = pose[:3] + jnp.matmul(body.force, turn.T, precision="highest")
    wall_points = pose[:3] + jnp.matmul(cells.points[jnp.asarray(kept)], turn.T, precision="highest")
    wall_weights = on_the_cells(jax.ops.segment_sum(cells.weights, jnp.asarray(runs), num_segments=len(kept)), owners,
                                body.area)
    gap = wall.distance(force, parameters) / float(node.constants()["force_spacing"])

    def point_image(force: Any, parameters: Any) -> Any:
        return wall.correction(force, parameters, viscosity).matrix * jnp.repeat(body.area, 3)[None, :]

    def integrated(chunks: Chunks) -> Callable[..., Any]:
        def image(force: Any, wall_points: Any, wall_weights: Any, gap: Any, parameters: Any) -> Any:
            point = wall.correction(force, parameters, viscosity)
            point = Corrected(point.matrix * jnp.repeat(body.area, 3)[None, :], point.accepted)
            return near_wall_image(wall, point, force, wall_points, wall_weights, gap,
                                   None if every else table, node.wall_budget, chunks, parameters, viscosity)[0].matrix
        return image

    arguments = (force, wall_points, wall_weights, gap, parameters)
    found["point_image"], _ = timed(jax.jit(point_image), force, parameters, repeats=repeats)
    found["w2_image"], reference = timed(jax.jit(integrated(chunks)), *arguments, repeats=repeats)
    reference = np.asarray(reference)
    print(json.dumps(found), flush=True)
    for values in chunk_values:
        cut = Chunks.of(np.repeat(np.arange(node.wall_budget, dtype=np.int32), table.shape[1]), node.wall_budget,
                        pair_values=values) if not every else Chunks.of(owners, node.sample_count, pair_values=values)
        timing, matrix = timed(jax.jit(integrated(cut)), *arguments, repeats=repeats)
        matrix = np.asarray(matrix)
        moved = float(np.max(np.abs(matrix - reference)) / np.max(np.abs(reference)))
        found[f"w2_chunks_cut_for_{values}_values"] = {**timing, "chunks": int(cut.points.shape[0]),
                                                      "chunk_points": int(cut.points.shape[1]),
                                                      "bitwise": matrix.tobytes() == reference.tobytes(),
                                                      "largest_move_over_largest_entry": moved}
        print(json.dumps({values: found[f"w2_chunks_cut_for_{values}_values"]}), flush=True)
    return found


def main(arguments: list[str]) -> None:
    import jax

    out, part, subject, *rest = arguments
    repeats = int(rest[rest.index("--repeats") + 1]) if "--repeats" in rest else 3
    values = [int(rest[i + 1]) for i, word in enumerate(rest) if word == "--chunk-values"]
    found: dict[str, Any] = {"machine": {"cpu": platform.processor(), "threads": os.cpu_count(),
                                         "devices": [str(device) for device in jax.devices()],
                                         "jax": jax.__version__, "load_1min": os.getloadavg()[0]}}
    if part == "split":
        if subject not in RUN_STEPS:
            raise SystemExit(f"split takes floor or axis, not {subject!r}")
        found["split"] = split(subject, repeats)
    elif part == "w2":
        found["w2"] = w2(float(subject), values, repeats)
    else:
        raise SystemExit(f"the parts are split and w2, not {part!r}")
    found["machine"]["load_1min_after"] = os.getloadavg()[0]
    Path(out).write_text(json.dumps(found, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
