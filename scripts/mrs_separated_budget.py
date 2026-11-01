"""The separated quadrature's budget (drag 0.4.0, phase 7m-b): what the MRS rung's assembly costs once per build at
the defaults, and what a step costs on script A's MRS arm (A0's is the same body, tube and rung) on drag 0.4.0 against
the point forces of 0.3.0 at the same spacing, with the compile.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/mrs_separated_budget.py <out.json> [assembly] [arm]

Not collected by pytest. Each time is wall-clock, on this machine's CPU, in the precision a script runs in by default
(float32) for a run, and in float64 for the assembly, which the build always makes in float64:

- ``assembly``: the held body-frame quantity (``MRSDragNode.held_in_body_frame``) of each family's default body (the
  sphere of radius 1 mm and FL-9, each in free space and in the tube, where it is the free part's matrix), the first
  call (which compiles the assembly's program) and a second on a node built again with the same settings; and the
  build of the experiment whole.
- ``arm``: script A's MRS arm (``scripts/sim2real/operating_point.py``'s ``experiment(rung="bem")``, FL-9 on the floor
  of the 1/4" tube) at ``ARM_SPACINGS_MM``, with its laws and kinds unpinned (the register's own arm stays drag 0.3.0 on
  the screw 0.2.0): its build, its first run of short run (the compile and the steps), the time per step from warm
  short and long runs (``STEPS``) (each length compiles a program of its own, so both are run once first),
  and the least eigenvalue of the resistance at the start, scaled, with the wall in it.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts" / "sim2real"))

#: The arm's spacings: the coarse one 7m-a measured, and the frozen point's.
ARM_SPACINGS_MM = (0.4, 0.178)
#: Steps of the arm's short and long runs, by spacing: fewer at the frozen point, where a step takes tens of seconds.
STEPS = {0.4: (3, 8), 0.178: (1, 3)}


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assembly() -> list[dict[str, Any]]:
    from mrs_cases import free_space
    from script_a import fl9

    from mime import Body, Experiment, Medium, State, Structure
    from mime.dynamics import RigidInertial
    from mime.interactions import Drag
    from mime.nodes.hydro.mrs import MRSDragNode
    from mime.scene import Scene, Sphere, Tube, X, Z
    from mime.things import Part
    from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

    def in_tube(kind: Any) -> Experiment:
        e = Experiment("budget_tube", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
        tube = e.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                               length=mm(100.0))))
        e.add_medium(Medium("water", fills=tube.lumen))
        e.declare("water.viscosity", mPa_s(1.0), reason="water")
        e.add_body(Body("body", dynamics=RigidInertial(), parts={"shell": Part(kind)}, initial=State(
            position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
            angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        e.declare("body.mass", kg(1e-7), reason="a millimetre body's order")
        e.declare("body.inertia", kg_m2((1e-14, 1e-14, 1e-14)), reason="a millimetre body's order")
        e.attach(Drag.MRS(wall_kind="tube"), body="body", medium="water")
        return e

    rows = []
    for label, kind in (("sphere", Sphere(radius=mm(1.0))), ("screw", fl9())):
        for where in ("free_space", "tube"):
            make = (lambda: free_space("budget", {"body": kind}, Drag.MRS(wall_kind="free_space"))) \
                if where == "free_space" else (lambda: in_tube(kind))
            started = time.perf_counter()
            built = make().build()
            build_s = time.perf_counter() - started
            node = built.graph.get_node("drag:body")
            assert isinstance(node, MRSDragNode)
            again = MRSDragNode(node.name, node.delta_t, **dict(node.constants()))
            started = time.perf_counter()
            _ = again.held_in_body_frame
            second_s = time.perf_counter() - started
            rows.append({"body": label, "wall": where, "force_points": node.sample_count,
                         "quadrature_points": node.quadrature_count, "blob": node.constants()["blob"],
                         "build_s": build_s, "assembly_again_s": second_s,
                         "asymmetry": node.free_asymmetry})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def timed(run: Any) -> float:
    started = time.perf_counter()
    run()
    return time.perf_counter() - started


def least_eigenvalue(built: Any, node: Any) -> float:
    """The least eigenvalue of the resistance at the body's starting pose, the wall's correction in it, each row and
    column scaled by its diagonal: below zero, the confined resistance is not positive definite there."""
    import jax.numpy as jnp
    import numpy as np

    from mime.build._maddening import node_state, parameter_leaves

    held = node_state(built.graph, node.name)
    pose = node_state(built.graph, "umr")["pose"]
    out = node.update(held, {"pose": jnp.asarray(pose, dtype=held["resistance"].dtype)}, 1e-3,
                      params=parameter_leaves(built.graph)[node.name])
    found = np.asarray(out["resistance"], dtype=np.float64)
    scale = np.sqrt(np.abs(np.diag(found)))
    return float(np.linalg.eigvalsh(found / np.outer(scale, scale))[0])


def arm() -> list[dict[str, Any]]:
    from mime.build import BuildRefused
    from mime.nodes.hydro.mrs import MRSDragNode
    from mime.units import s

    op = _load("_budget_operating_point", ROOT / "scripts" / "sim2real" / "operating_point.py")
    rows = []
    for spacing in ARM_SPACINGS_MM:
        for version in ("0.4.0", "0.3.0"):
            if version == "0.4.0":
                op.pinned_law = lambda law: law
                op.pinned = lambda geometry: geometry
                blob = 0.25
            else:
                op.pinned_law = _load("_budget_pins", ROOT / "scripts" / "sim2real" / "operating_point.py").pinned_law
                op.pinned = lambda geometry: geometry
                blob = 0.5
            started = time.perf_counter()
            refusal: dict[str, Any] = {}
            try:
                built = op.experiment(rung="bem", spacing_mm=spacing, blob=blob).build()
            except BuildRefused as refused:
                # Refused at build (the arm's floored pose): recorded, and the step timed with the check bypassed,
                # since what a step costs does not depend on whether its resistance is definite.
                refusal = {"refused": refused.codes(), "why": refused.refusals[0].message}
                held_check = MRSDragNode.built_problems
                MRSDragNode.built_problems = lambda self, start=None: []  # type: ignore[method-assign]
                try:
                    started = time.perf_counter()
                    built = op.experiment(rung="bem", spacing_mm=spacing, blob=blob).build()
                finally:
                    MRSDragNode.built_problems = held_check  # type: ignore[method-assign]
            build_s = time.perf_counter() - started
            node = built.graph.get_node(op.DRAG_NODE)
            least = least_eigenvalue(built, node)
            dt = float(op.DT_MS) * 1e-3
            short, long = STEPS[spacing]
            started = time.perf_counter()
            built.run(s(short * dt))
            first_s = time.perf_counter() - started
            built.run(s(long * dt))  # each run's length compiles a program of its own: both are warm before timing
            short_s, long_s = (min(timed(lambda n=n: built.run(s(n * dt))) for _ in range(2)) for n in (short, long))
            per_step = max(long_s - short_s, 0.0) / (long - short)
            rows.append({"spacing_mm": spacing, "drag": version, "force_points": node.sample_count,
                         "quadrature_points": node.quadrature_count if node.separated else None, "blob": blob,
                         "least_scaled_eigenvalue_at_start": least, **refusal,
                         "build_s": build_s, "first_run_s": first_s, "per_step_s": per_step,
                         "compile_s": max(first_s - short * per_step, 0.0)})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main(out: Path, parts: tuple[str, ...] = ("assembly", "arm")) -> None:
    import os

    found: dict[str, Any] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    found["machine"] = {"cpu": platform.processor(), "threads": os.cpu_count(), "python": platform.python_version(),
                        "jax": jax.__version__, "load_1min": os.getloadavg()[0]}
    if "assembly" in parts:
        found["assembly"] = assembly()
    if "arm" in parts:
        found["arm"] = arm()
    found["machine"]["load_1min_after"] = os.getloadavg()[0]
    out.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), tuple(sys.argv[2:]) or ("assembly", "arm"))
