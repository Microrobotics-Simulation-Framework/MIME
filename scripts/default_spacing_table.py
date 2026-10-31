"""The boundary-element rung's default spacing, measured: for each body family the default serves, how far its headline
number is from a reference at a ladder of sample counts, and how long a task of the first-hour walk's size takes at
each on this machine's CPU. ``Drag``'s default spacing (``mime.interactions.hydrodynamic``) is chosen from this table,
and its basis quotes the misses at the chosen count.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/default_spacing_table.py miss <out.json>
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/default_spacing_table.py time <out.json>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/default_spacing_table.py default-miss <out.json> \
        <miss.json>
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/default_spacing_table.py default-time <out.json>

The two ``default-`` modes measure the same at the spacing the default rule takes (``Drag.BEM()`` with no spacing),
against the references ``miss`` measured: ``default-miss <out.json> <miss.json>``.

Not collected by pytest. The two families and what each measures:

- **sphere**: the walk's T1, a steel sphere 100 um across settling from rest in still water in free space. Its miss is
  the drag against Stokes's law, translation (``6 pi mu a``) and rotation (``8 pi mu a^3``), from the rung's resistance
  at the start; its task is T1 as the walk wrote it, 0.05 s at a step of 0.05 ms (1000 steps).
- **screw**: de Jongh's FL-9 on the axis of the walk's T2 tube (3.175 mm radius, the tube wall kind), and in free
  space. Its miss is the force-free speed along the axis per turn about it, from the 6 x 6 resistance, against the
  finest spacing measured (no closed form exists); its task is T2 with the boundary-element rung in place of the
  surrogate, 3 s at a step of 0.5 ms (6000 steps) under a 5 Hz rotating field.

``miss`` measures the misses (in float64, so that the number is the discretisation's and not float32's rounding);
``time`` the tasks' run times in the precision a script runs in by default (float32): the build, then the run, timed
whole where it is short and otherwise estimated from short runs after the first, which compiles (the time per step
from two lengths' difference, the run's fixed cost and the compile from the rest); where both are measured, the
estimate is held against the whole run. The estimate is noisy and reads low (14 s against a whole run's 22 s at the
sphere's 338 samples, 40 s against 51 s at 512), so an estimated time over the budget is a lower bound, and the
default's own time is a whole run's (``whole_s``). The tube case reads the asset
``unit_tube_R0.997`` (``mime assets fetch`` it first)."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.build._maddening import node_state, parameter_leaves
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Drag, Gravity, Magnetic
from mime.nodes.hydro.bem import BEMDragNode
from mime.observations import SwimSpeed
from mime.scene import Scene, Screw, Sphere, Tube, X, Z
from mime.things import Part
from mime.units import (
    A_m2,
    Hz,
    dimensionless,
    kg_m2,
    kg_per_m3,
    m,
    m_per_s,
    m_per_s2,
    mm,
    mPa_s,
    ms,
    mT,
    rad_per_s,
    s,
    um,
)

MU = 1.0e-3
#: T1's sphere: its radius, and the spacings measured, as fractions of it.
SPHERE_RADIUS = 50e-6
SPHERE_FRACTIONS = (0.4, 0.3, 0.25, 0.2, 0.133, 0.1, 0.08)
#: The sphere's spacings timed: past 0.133 R a walk-sized run is minutes, far over the budget.
SPHERE_TIMED = (0.4, 0.3, 0.25, 0.2, 0.133)
#: FL-9 (de Jongh et al. 2025, Table 1), and the spacings measured (m); the finest is each case's reference.
FL9 = {"radius": 1.56e-3, "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33}
SCREW_SPACINGS = (0.83e-3, 0.7e-3, 0.6e-3, 0.5e-3, 0.4e-3, 0.3e-3)
SCREW_TIMED = (0.83e-3, 0.7e-3, 0.6e-3, 0.5e-3, 0.4e-3)
TUBE_FINEST, FREE_FINEST = 0.16e-3, 0.15e-3
TUBE_RADIUS = 3.175e-3
#: Each task's duration and step: T1's and T2's, as the walk wrote them.
SPHERE_TASK, SPHERE_DT = 0.05, 0.05e-3
SCREW_TASK, SCREW_DT = 3.0, 0.5e-3
#: A run expected to take longer than this is timed from two short runs instead of whole.
WHOLE_LIMIT_S = 60.0
SHORT_STEPS = (4, 16)


def fl9() -> Screw:
    return Screw(radius=m(FL9["radius"]), length=m(FL9["length"]), wavenumber=dimensionless(FL9["wavenumber"]),
                 modulation=dimensionless(FL9["modulation"]), starts=2)


def at_rest(position: Any) -> State:
    return State(position=position, orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
                 angular_velocity=rad_per_s((0.0, 0.0, 0.0)))


def bem(spacing: float | None, wall_kind: str) -> Drag:
    """The rung at ``spacing``, or at its default (``None``), the blob at its default's value either way."""
    return Drag.BEM(wall_kind=wall_kind) if spacing is None else Drag.BEM(spacing=m(spacing), blob=0.5,
                                                                           wall_kind=wall_kind)


def sphere_task(spacing: float | None) -> Experiment:
    """The walk's T1 at ``spacing`` (``None``: the default's)."""
    exp = Experiment("t1", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.05))
    water = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=at_rest(um((0.0, 0.0, 0.0))),
                             parts={"sphere": Part(Sphere(radius=m(SPHERE_RADIUS)))}))
    exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
    exp.attach(bem(spacing, "free_space"), body=ball, medium=water)
    exp.attach(Gravity(), body=ball, medium=water)
    exp.observe(SwimSpeed(ball, along=-Z, after=s(0.04)))
    return exp


def screw_task(spacing: float | None, *, tube: bool = True) -> Experiment:
    """The walk's T2 with the boundary-element rung at ``spacing`` (``None``: the default's), in its tube or in free
    space."""
    exp = Experiment("t2", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(0.5))
    if tube:
        held = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=m(TUBE_RADIUS), wall=mm(1.6),
                                                                   length=mm(100.0))))
        water = exp.add_medium(Medium("water", fills=held.lumen))
    else:
        water = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(1000.0), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    umr = exp.add_body(Body("umr", dynamics=RigidInertial(), initial=at_rest(mm((0.0, 0.0, 0.0))),
                            parts={"shell": Part(fl9())}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
    exp.declare("umr.inertia", kg_m2((1.0331989974018431e-10,) * 3), reason="m R^2/2 of a solid cylinder")
    exp.declare("umr.magnetic_moment", A_m2((0.0, 1.68e-3, 0.0)), reason="two magnets")
    field = exp.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotating(rotation_axis=X, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(5.0), "amplitude": mT(1.2)}))
    exp.attach(bem(spacing, "tube" if tube else "free_space"), body=umr, medium=water)
    exp.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    exp.observe(SwimSpeed(umr, along=X, after=s(2.0)))
    return exp


def resistance(built: Any, body: str) -> tuple[np.ndarray, int]:
    """The rung's 6 x 6 resistance with the body at its starting pose, and the samples' count."""
    node = built.graph.get_node(f"drag:{body}")
    assert isinstance(node, BEMDragNode), type(node)
    params = parameter_leaves(built.graph)[f"drag:{body}"]
    held = node_state(built.graph, f"drag:{body}")
    pose = node_state(built.graph, body)["pose"]
    out = node.update(held, {"pose": jnp.asarray(pose, dtype=held["resistance"].dtype)}, 1e-3, params=params)
    return np.asarray(out["resistance"], dtype=np.float64), int(node.samples(params).points.shape[0])


def speed_per_turn(r: np.ndarray) -> float:
    """The force-free speed along x per turn about x (m), from the resistance in the body's frame."""
    a, b = r[:3, :3], r[:3, 3:]
    return float(-np.linalg.solve(a, b @ np.array([2.0 * math.pi, 0.0, 0.0]))[0])


def misses() -> dict[str, Any]:
    rows: dict[str, Any] = {"sphere": [], "screw_tube": [], "screw_free": []}
    a = SPHERE_RADIUS
    for fraction in SPHERE_FRACTIONS:
        started = time.perf_counter()
        r, count = resistance(sphere_task(fraction * a).build(), "ball")
        rows["sphere"].append({"spacing_m": fraction * a, "spacing_per_radius": fraction, "samples": count,
                               "translation_vs_stokes": r[0, 0] / (6 * math.pi * MU * a) - 1.0,
                               "rotation_vs_stokes": r[3, 3] / (8 * math.pi * MU * a ** 3) - 1.0,
                               "seconds": time.perf_counter() - started})
        print("sphere", rows["sphere"][-1], flush=True)
    for key, tube, finest in (("screw_tube", True, TUBE_FINEST), ("screw_free", False, FREE_FINEST)):
        for spacing in (*SCREW_SPACINGS, finest):
            started = time.perf_counter()
            r, count = resistance(screw_task(spacing, tube=tube).build(), "umr")
            rows[key].append({"spacing_m": spacing, "samples": count, "speed_per_turn_m": speed_per_turn(r),
                              "seconds": time.perf_counter() - started})
            print(key, rows[key][-1], flush=True)
        reference = rows[key][-1]["speed_per_turn_m"]
        for row in rows[key]:
            row["vs_finest"] = row["speed_per_turn_m"] / reference - 1.0
    return rows


def _timed(run: Callable[[], object]) -> float:
    started = time.perf_counter()
    run()
    return time.perf_counter() - started


def timed_task(make: Callable[[], Experiment], duration: float, dt: float) -> dict[str, Any]:
    """A task's build and run times: whole when short enough, otherwise estimated from short runs after the first
    (which compiles): the time per step from two lengths' difference, a run's fixed cost and the compile from the
    rest."""
    steps = round(duration / dt)
    started = time.perf_counter()
    built = make().build()
    build = time.perf_counter() - started
    first, second = SHORT_STEPS
    warm = _timed(lambda: built.run(s(first * dt)))
    t_first = min(_timed(lambda: built.run(s(first * dt))) for _ in range(2))
    t_second = min(_timed(lambda: built.run(s(second * dt))) for _ in range(2))
    per_step = max(t_second - t_first, 0.0) / (second - first)
    fixed = max(t_first - first * per_step, 0.0)
    compile_s = max(warm - t_first, 0.0)
    estimate = build + compile_s + fixed + steps * per_step
    row: dict[str, Any] = {"steps": steps, "build_s": build, "compile_s": compile_s, "fixed_s": fixed,
                           "per_step_s": per_step, "estimated_s": estimate, "whole_s": None}
    if estimate <= WHOLE_LIMIT_S:
        started = time.perf_counter()
        fresh = make().build()
        fresh.run(s(duration))
        row["whole_s"] = time.perf_counter() - started
    return row


def times() -> dict[str, Any]:
    rows: dict[str, Any] = {"sphere": [], "screw_tube": []}
    for fraction in SPHERE_TIMED:
        spacing = fraction * SPHERE_RADIUS
        row = timed_task(lambda: sphere_task(spacing), SPHERE_TASK, SPHERE_DT)
        rows["sphere"].append({"spacing_m": spacing, "spacing_per_radius": fraction, **row})
        print("sphere", rows["sphere"][-1], flush=True)
    for spacing in SCREW_TIMED:
        row = timed_task(lambda: screw_task(spacing), SCREW_TASK, SCREW_DT)
        rows["screw_tube"].append({"spacing_m": spacing, **row})
        print("screw_tube", rows["screw_tube"][-1], flush=True)
    return rows


def taken_spacing(built: Any, body: str) -> float:
    return float(next(d.value.value for d in built.report.profile.defaults if d.path == f"drag:{body}.spacing"))


def default_misses(references: dict[str, Any]) -> dict[str, Any]:
    """The misses at the default's spacing, against ``miss``'s references (Stokes's law; the finest spacing)."""
    a = SPHERE_RADIUS
    built = sphere_task(None).build()
    r, count = resistance(built, "ball")
    rows: dict[str, Any] = {"sphere": {"spacing_m": taken_spacing(built, "ball"), "samples": count,
                                       "translation_vs_stokes": r[0, 0] / (6 * math.pi * MU * a) - 1.0,
                                       "rotation_vs_stokes": r[3, 3] / (8 * math.pi * MU * a ** 3) - 1.0}}
    for key, tube in (("screw_tube", True), ("screw_free", False)):
        built = screw_task(None, tube=tube).build()
        r, count = resistance(built, "umr")
        finest = references["rows"][key][-1]
        speed = speed_per_turn(r)
        rows[key] = {"spacing_m": taken_spacing(built, "umr"), "samples": count, "speed_per_turn_m": speed,
                     "vs_finest": speed / finest["speed_per_turn_m"] - 1.0, "finest_spacing_m": finest["spacing_m"],
                     "finest_samples": finest["samples"]}
    return rows


def whole_task(make: Callable[[], Experiment], duration: float, dt: float) -> dict[str, Any]:
    """A task built and run whole, as a first run is: its time from the build's start to the run's end."""
    started = time.perf_counter()
    make().build().run(s(duration))
    return {"steps": round(duration / dt), "whole_s": time.perf_counter() - started}


def default_times() -> dict[str, Any]:
    """The tasks at the default's spacing, each run whole (the number the default states)."""
    return {"sphere": whole_task(lambda: sphere_task(None), SPHERE_TASK, SPHERE_DT),
            "screw_tube": whole_task(lambda: screw_task(None), SCREW_TASK, SCREW_DT)}


def main(argv: list[str]) -> int:
    modes = ("miss", "time", "default-miss", "default-time")
    if len(argv) not in (2, 3) or argv[0] not in modes or (argv[0] == "default-miss") != (len(argv) == 3):
        print(__doc__, file=sys.stderr)
        return 2
    started = time.perf_counter()
    if argv[0] == "default-miss":
        rows = default_misses(json.loads(Path(argv[2]).read_text(encoding="utf-8")))
    else:
        rows = {"miss": misses, "time": times, "default-time": default_times}[argv[0]]()
    out = {"measure": argv[0], "x64": bool(jax.config.read("jax_enable_x64")), "machine": platform.processor() or
           platform.machine(), "cpus": __import__("os").cpu_count(), "devices": [str(d) for d in jax.devices()],
           "seconds": time.perf_counter() - started, "rows": rows}
    Path(argv[1]).write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
