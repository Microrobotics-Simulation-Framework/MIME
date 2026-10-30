"""Script A's operating point (FL-9 on the floor of the 1/4" tube, a uniform field turning at 10 Hz) and every rung
of phase 7j-a's ladder built from it: the experiments, the static resistances at the floored pose, the swim speeds
read from a resistance, and the steady speed of a run. Imported by the phase's scripts; it computes nothing itself.

**The experiment** is ``tests/script_a.py``'s reproduction arm, declared again here with every declared value and
setting as an argument (the drag's rung, the surrogate's weights, the lubrication, the contact margin, the time step,
the moment, the field, the densities, the viscosity, the inertia), so that a variant is a configuration and not new
code. :func:`same_as_script_a` checks that at its defaults it builds the reproduction arm's graph.

**The floored pose.** On the floor the contact holds the body's centre at the tube's radius less the body's extent
and the margin from the axis (script A's run sits there to rounding, its radial offset 1.0002 mm). A static rung
reads its resistance there, the offset computed from the graph's own leaves (``umr.surface_radius``, ``umr.extent``,
``umr.margin``), the body's axis along the tube's and turned about it by ``phase``. The de Jongh screw with two
starts is unchanged by a half turn about its axis, so a phase average is taken over ``[0, pi)``.

**The swim speeds from a resistance** ``R`` (6 x 6 about the centre of mass, rows and columns ``Ux, Uy, Uz, Ox, Oy,
Oz``; the body receives ``-R [U; O]``), the body turning at ``2 pi f`` about the tube's axis (+x):

- ``axial``: only ``Ux`` free, every other velocity prescribed (zero); the force along the axis zero;
- ``lateral_free``: ``Ux, Uy, Uz`` free, the transverse turning zero; the force zero;
- ``wall_held``: ``Ux, Uy`` and the transverse turning ``Oy, Oz`` free and ``Uz = 0`` (the floor's normal); the
  force along the wall and the transverse torque zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _where in (ROOT / "tests", ROOT / "tests" / "differential"):
    if str(_where) not in sys.path:
        sys.path.insert(0, str(_where))

import numpy as np  # noqa: E402
import script_a as A  # noqa: E402  (tests/script_a.py: script A's declared values, with their provenance)

#: The surrogate's weights the ladder's surrogate rows use until phase 7i ships its retrained ones: pass
#: ``--weights <asset>`` to every script to re-run those rows with another.
DEFAULT_WEIGHTS = "dejongh_mlp_v2"
FREQUENCY_HZ = 10.0
MARGIN_MM = 0.1
DT_MS = 0.5
TUBE_RADIUS_M = 3.175e-3
EPSILON_MM = 0.15
#: The BEM's default discretisation at the floored pose: the sample spacing of the 0.2 line's training mesh's scale and
#: its blob; the closed surface (lateral and caps) and the lateral surface alone are both reported (neither is chosen).
SPACING_MM = 0.178
BLOB = 0.5
PATCHES = {"closed": None, "lateral": ("lateral",)}
#: Phases of the body's turn about its axis at which a static rung is read, over the half turn that repeats.
PHASES = tuple(math.pi * k / 8 for k in range(8))
DURATION_S = 5.0
VARIANTS = ("axial", "lateral_free", "wall_held")
DRAG_NODE, BODY_NODE = "drag:umr", "umr"
#: The field node's state field the torque reads (``tests/differential/script_a_runs.py``'s ``FIELD``).
FIELD = "flux_density_magnetic_torque_umr"


def experiment(rung: str = "surrogate", *, name: str = "script_a_7ja", weights: str = DEFAULT_WEIGHTS,
               lubrication: bool = True, epsilon_mm: float = EPSILON_MM, margin_mm: float = MARGIN_MM,
               dt_ms: float = DT_MS, frequency_hz: float = FREQUENCY_HZ, amplitude_mT: float = 1.2,
               moment: float = 2 * 8.4e-4, body_density: float | None = None, water_density: float = 1000.0,
               viscosity_mPa_s: float = 1.0, inertia_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
               tube_radius_mm: float = 3.175, spacing_mm: float = SPACING_MM, blob: float = BLOB,
               patches: str = "closed", start_mm: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> Any:
    """Script A's reproduction arm with every declared value and setting as an argument; at the defaults it is that
    arm. ``rung`` is ``"surrogate"`` (with ``lubrication`` on or off) or ``"bem"`` (the tube's wall kind, at
    ``spacing_mm``, ``blob`` and ``patches``). ``body_density`` (kg/m^3), when given, sets the mass from the body's
    volume, which is otherwise the reference's ``m_eff``."""
    from mime import Apparatus, Body, Experiment, Medium, State, Structure
    from mime.data import asset
    from mime.dynamics import RigidInertial
    from mime.fields import UniformRotating
    from mime.interactions import Contact, Drag, Gravity, Lubrication, Magnetic
    from mime.observations import SwimSpeed
    from mime.scene import Scene, Tube, X, Z
    from mime.things import Part
    from mime.units import A_m2, Hz, kg, kg_m2, kg_per_m3, m, m3, m_per_s, m_per_s2, mm, mPa_s, ms, mT, rad_per_s

    reason = A.REASON
    built = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(dt_ms))
    tube = built.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(tube_radius_mm),
                                                               wall=mm(1.6), length=mm(100.0))))
    water = built.add_medium(Medium("water", fills=tube.lumen))
    umr = built.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm(start_mm), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(A.fl9())}))
    field = built.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotating(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(frequency_hz), "amplitude": mT(amplitude_mT)}))
    volume = float(A.VOLUME.value)
    mass = float(A.MASS.value) if body_density is None else body_density * volume
    inertia = tuple(float(i) * s for i, s in zip(A.INERTIA.value, inertia_scale, strict=True))
    declared = [("umr.inertia", kg_m2(inertia)), ("umr.mass", kg(mass)), ("umr.buoyant_volume", m3(volume)),
                ("umr.magnetic_moment", A_m2((0.0, moment, 0.0))), ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))),
                ("water.density", kg_per_m3(water_density)), ("water.viscosity", mPa_s(viscosity_mPa_s))]
    for path, value in declared:
        built.declare(path, value, reason=reason if path != "umr.centre_of_buoyancy" else
                      "the reference's body has one density, so its centre of buoyancy is its centre of mass")
    if rung == "surrogate":
        law = Drag.Surrogate(weights=asset(weights), lubrication=Lubrication(epsilon=mm(epsilon_mm))
                             if lubrication else None)
    elif rung == "bem":
        law = Drag.BEM(spacing=mm(spacing_mm), blob=blob, patches=PATCHES[patches], wall_kind="tube")
    else:
        raise ValueError(f"rung {rung!r} is neither 'surrogate' nor 'bem'")
    built.attach(law, body=umr, medium=water)
    built.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    built.attach(Gravity(), body=umr, medium=water)
    built.attach(Contact.Projection(margin=mm(margin_mm)), body=umr, surface=tube.inner_surface)
    built.observe(SwimSpeed(umr, along=tube.axis))
    return built


def leaves_of(built: Any) -> dict[str, dict[str, Any]]:
    from mime.build._maddening import parameter_leaves

    return parameter_leaves(built.graph)


def same_as_script_a(built: Any) -> list[str]:
    """Where ``built`` (an :func:`experiment` at its defaults) differs from ``tests/script_a.py``'s reproduction arm:
    its nodes in order, its edges, and every parameter leaf, bitwise. Empty when it is that arm."""
    from mime.build._maddening import graph_edges, graph_schedule

    reference = A.reproduction_arm().build()
    found: list[str] = []
    if graph_schedule(built.graph) != graph_schedule(reference.graph):
        found.append(f"schedule {graph_schedule(built.graph)} != {graph_schedule(reference.graph)}")
    if graph_edges(built.graph) != graph_edges(reference.graph):
        found.append("edges differ")
    mine, theirs = leaves_of(built), leaves_of(reference)
    for node in sorted(set(mine) | set(theirs)):
        a, b = mine.get(node, {}), theirs.get(node, {})
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                found.append(f"{node}.{key} present in one only")
            elif not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
                found.append(f"{node}.{key}: {np.asarray(a[key]).tolist()} != {np.asarray(b[key]).tolist()}")
    return found


# ---- the floored pose and the static resistance ---------------------------------------------------------------------
def floored_offset(body_leaves: dict[str, Any]) -> Any:
    """The centre's distance from the axis where the contact holds a floored body: the tube's radius less the extent
    and the margin, from the graph's leaves."""
    return body_leaves["surface_radius"] - body_leaves["extent"] - body_leaves["margin"]


def floored_pose(offset: Any, phase: float) -> Any:
    """The pose on the floor (below the axis, up is +z), the body's axis along the tube's, turned by ``phase``."""
    import jax.numpy as jnp

    zero = 0.0 * offset
    return jnp.stack([zero, zero, -offset, zero + math.cos(phase / 2), zero + math.sin(phase / 2), zero, zero])


def static_resistance(built: Any, pose: Any, leaves: dict[str, Any] | None = None) -> Any:
    """The drag node's resistance at ``pose``, with the drag's leaves ``leaves`` (the graph's own when none)."""
    from mime.build._maddening import node_state

    node = built.graph.get_node(DRAG_NODE)
    held = node_state(built.graph, DRAG_NODE)
    used = leaves if leaves is not None else leaves_of(built)[DRAG_NODE]
    return node.update(held, {"pose": pose}, 1e-3, params=used)["resistance"]


def bem_resistance(built: Any, pose: Any, leaves: dict[str, Any] | None = None, *, correction_scale: float = 1.0,
                   extra: Any = None) -> Any:
    """The BEM node's resistance at ``pose`` computed as its step computes it, with the wall's correction multiplied
    by ``correction_scale`` and ``extra`` (a 6 x 6, the lubrication's) added: at a scale of one and no extra it is
    the node's (:func:`static_resistance`); the comparison is a control in ``bem.py``."""
    import jax.numpy as jnp

    from mime.core import quaternion
    from mime.nodes.hydro.bem import blob_squared, resistance, wall_correction_named

    node = built.graph.get_node(DRAG_NODE)
    # What the node's step reads as its constants: its own, the leaves over them, the wall's table rebuilt from its
    # leaves (an argument of the compiled step, never a constant of the program).
    p = node._step_constants(leaves if leaves is not None else leaves_of(built)[DRAG_NODE])  # noqa: SLF001
    held = node.samples(p)
    points = pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision="highest")
    wall = wall_correction_named(node.constants()["wall_kind"], node.constants()["wall_kind_version"])
    corrected = wall.correction(points, {name: p[f"wall_{name}"] for name in wall.parameters}, p["viscosity"])
    found = resistance(points, blob_squared(held.weights, p["blob"]), p["viscosity"], pose[:3],
                       correction_scale * corrected.matrix)
    return found if extra is None else found + extra


def lubrication_matrix(surrogate_built: Any, pose: Any) -> Any:
    """The surrogate rung's lubrication term (6 x 6, SI) at ``pose``, from its own node: what script A adds to the
    surrogate, here to be added to the BEM's resistance by a script, since the rung that carries it is the surrogate's
    alone (``Drag.BEM`` takes no ``lubrication``)."""
    node = surrogate_built.graph.get_node(DRAG_NODE)
    p = node._step_constants(leaves_of(surrogate_built)[DRAG_NODE])  # noqa: SLF001 - what the node's step reads
    _, frame = node.frame(p, pose[:3], pose[3:])
    side = node.end(p, pose[3:])
    import jax.numpy as jnp

    frame = frame * jnp.stack([jnp.ones_like(side), side, side])
    return node.lubrication(p, pose[:3], frame)


# ---- swim speeds ---------------------------------------------------------------------------------------------------
def speeds(resistance: Any, frequency_hz: float = FREQUENCY_HZ) -> dict[str, Any]:
    """The swim speed along the tube (m/s) for each variant of the module's docstring, the body turning at
    ``2 pi frequency_hz`` about +x. Traceable: ``resistance`` may be a tracer."""
    import jax.numpy as jnp

    r = resistance
    omega = 2.0 * math.pi * frequency_hz
    axial = -r[0, 3] * omega / r[0, 0]
    lateral = -jnp.linalg.solve(r[:3, :3], r[:3, 3] * omega)
    free = jnp.asarray([0, 1, 4, 5])
    held = -jnp.linalg.solve(r[free][:, free], r[free, 3] * omega)
    return {"axial": axial, "lateral_free": lateral[0], "wall_held": held[0], "lateral_free_sideways": lateral[1],
            "wall_held_sideways": held[1], "wall_held_turning_y": held[2], "wall_held_turning_z": held[3]}


def phase_mean(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()),
            "spread_over_mean": float((arr.max() - arr.min()) / abs(arr.mean()))}


# ---- runs ----------------------------------------------------------------------------------------------------------
def run_steady(built: Any, dt_ms: float, duration_s: float = DURATION_S, frequency_hz: float = FREQUENCY_HZ
               ) -> dict[str, Any]:
    """Run ``built`` for ``duration_s`` and read its steady quantities by phase 0's rule
    (``tests/differential/script_a_steady.py``: the second half, trimmed to whole field periods), with a hash of its
    trajectory for the bit-for-bit check."""
    import time

    from script_a_steady import SCRIPT_A_FRAME, History, speed_between, steady

    from mime.units import s

    dt = dt_ms * 1e-3
    body, field = built.graph.get_node_state("umr"), built.graph.get_node_state("field.magnetic")
    started = time.perf_counter()
    result = built.run(s(duration_s), history=True)
    seconds = time.perf_counter() - started
    history = result.history
    assert history is not None

    def stacked(first: Any, rest: Any) -> np.ndarray:
        return np.concatenate([np.asarray(first)[None], np.asarray(rest)])

    pose = stacked(body["pose"], history["umr"]["pose"])
    twist = stacked(body["twist"], history["umr"]["twist"])
    held = History(time=np.arange(len(pose)) * dt, position=pose[:, :3], velocity=twist[:, :3],
                   orientation=pose[:, 3:], angular_velocity=twist[:, 3:],
                   field=stacked(field[FIELD], history["field.magnetic"][FIELD]))
    found: dict[str, Any] = dict(steady(held, SCRIPT_A_FRAME, frequency_hz))
    found["transient_speed_0_to_0.5_s_m_s"] = speed_between(held, SCRIPT_A_FRAME, 0.0, min(0.5, duration_s / 2))
    found["steps"] = int(len(pose) - 1)
    found["run_seconds"] = seconds
    found["trajectory_sha256"] = hashlib.sha256(np.ascontiguousarray(pose, dtype=np.float64).tobytes()
                                                + np.ascontiguousarray(twist, dtype=np.float64).tobytes()).hexdigest()
    found["tilt_max_rad"] = float(np.max(np.arccos(np.clip(np.abs(
        1 - 2 * (pose[:, 5] ** 2 + pose[:, 6] ** 2)), -1.0, 1.0))))
    return found


# ---- provenance ----------------------------------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    import jax

    info: dict[str, Any] = {"python": platform.python_version(), "jax": jax.__version__,
                            "x64": bool(jax.config.jax_enable_x64),
                            "matmul_precision": str(jax.config.jax_default_matmul_precision),
                            "devices": [f"{d.platform}:{d.device_kind}" for d in jax.devices()]}
    try:
        info["commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                        text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["commit"] = None
    return info


def require_float64() -> None:
    import jax

    if not jax.config.jax_enable_x64:
        raise SystemExit("phase 7j-a's breakdown is float64: run with JAX_ENABLE_X64=1")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True, default=_plain) + "\n")


def _plain(value: Any) -> Any:
    array = np.asarray(value)
    return array.tolist()
