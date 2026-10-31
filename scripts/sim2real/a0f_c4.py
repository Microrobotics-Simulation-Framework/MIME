"""C4's tilt windows at A (register section 6 C4, with reviews 1 and 2): A0f, the magnet fixed 150 mm above the tube's
axis and turning at 10 Hz, FL-9 floored in the 1/4" tube under it, its moment tilted by ``alpha`` off the body's
transverse plane (towards the body's axis), the closed BEM at 0.16 mm, dt 0.5 ms, float64. One window per ``alpha``
and direction. No measured speed enters. Predictions: ``predictions_c4.json``, committed before any run.

**Checks at every step, each with a stop** (as A0f's pass, section 5): the drag's resistance positive definite (its
smallest non-dimensional eigenvalue above zero), finite, and every wall read accepted (the drag node's
``wall_accepted``); and a stop if the body stalls (its displacement along the pass over the last turn not positive).
The run goes in chunks of ``CHUNK`` steps; every step of a chunk is checked, the window stops at the first step that
fails, and the window's record is saved after every chunk.

**What a complete window reads**: the speed along the pass over its last ``TURNS`` turns (after ``TRANSIENT_S``), and
the body's cone angle (its axis's angle from the tube's) per turn, their mean and scatter: the planted ``alpha = 4``
deg must be recovered within 20% or the scatter, whichever is larger.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import bem  # noqa: E402
import operating_point as op  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
import script_a as A  # noqa: E402

SPACING_MM = 0.16
DT_MS = 0.5
FREQUENCY_HZ = 10.0
MAGNET_MOMENT = 18.89
MAGNET_HEIGHT_M = 0.150
MAGNET_RADIUS, MAGNET_LENGTH = 17.5e-3, 20e-3
TRANSIENT_S = 0.1
TURNS = 5
CHUNK = 20
ALPHAS_DEG = (2.0, 4.0, 6.0)
PLANTED_DEG = 4.0

Save = Callable[[dict[str, Any]], None]


def experiment(alpha_deg: float, sense: int, spacing_mm: float = SPACING_MM, start_s_m: float = 0.0) -> Any:
    """A0f with the moment tilted. ``sense`` +1 swims +x: the field below a dipole turns against the dipole, so the
    motor turns at ``-sense * 2 pi f`` about +x."""
    from mime import Apparatus, Body, Experiment, Medium, State, Structure
    from mime.drives import Constant
    from mime.dynamics import Motor, RigidInertial
    from mime.fields import PointDipole
    from mime.interactions import Contact, Drag, Gravity, Magnetic
    from mime.observations import SwimSpeed
    from mime.scene import Cylinder, Scene, Tube, X, Z
    from mime.things import Part, PermanentMagnet
    from mime.units import A_m2, kg, kg_m2, kg_per_m3, m, m3, m_per_s, m_per_s2, mm, mPa_s, ms, rad, rad_per_s

    reason = A.REASON
    built = Experiment(f"a0f_c4_a{round(alpha_deg * 10)}_{'p' if sense > 0 else 'm'}", scene=Scene.new(up=Z, units="m"),
                       gravity=m_per_s2(9.81),
                       dt=ms(DT_MS))
    tube = built.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                               length=mm(100.0))))
    water = built.add_medium(Medium("water", fills=tube.lumen))
    offset = 1.0002e-3
    # the body's moment (body +y, tilted towards +x) along the field's direction across the tube at t = 0 (it points -y
    # below the magnet, so the body starts turned by pi at s = 0)
    phase = start_phase(start_s_m, offset)
    umr = built.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m((start_s_m, 0.0, -offset)), orientation=(math.cos(phase / 2), math.sin(phase / 2), 0.0, 0.0),
        velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(A.fl9())}))
    built.add_apparatus(Apparatus(
        "magnet",
        chain={"motor": Motor.Kinematic(axis=X, prescribed="velocity", initial_angle=rad(0.0),
                                        at=m((0.0, 0.0, MAGNET_HEIGHT_M)))},
        parts={"rotor_magnet": Part(Cylinder(radius=m(MAGNET_RADIUS), length=m(MAGNET_LENGTH), axis=(1.0, 0.0, 0.0)),
                                    mount="motor.rotor")},
        magnets={"rotor_magnet": PermanentMagnet(direction=(0.0, 1.0, 0.0))},
        fields={"magnetic": PointDipole(part="rotor_magnet")},
        commands={"motor.velocity": Constant(rad_per_s(-sense * 2 * math.pi * FREQUENCY_HZ))}))
    built.declare("magnet.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's 35 x 20 mm N45 rotor magnet")
    moment = float(A.MOMENT.value[1])
    alpha = math.radians(alpha_deg)
    declared = [("umr.inertia", kg_m2(tuple(float(i) for i in A.INERTIA.value))), ("umr.mass", kg(float(A.MASS.value))),
                ("umr.buoyant_volume", m3(float(A.VOLUME.value))),
                ("umr.magnetic_moment", A_m2((moment * math.sin(alpha), moment * math.cos(alpha), 0.0))),
                ("umr.magnet_centre", m((0.0, 0.0, 0.0))), ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))),
                ("water.density", kg_per_m3(float(A.WATER.value))), ("water.viscosity", mPa_s(1.0))]
    for path, value in declared:
        built.declare(path, value, reason=reason if path not in ("umr.centre_of_buoyancy", "umr.magnetic_moment")
                      else ("the reference's body has one density" if path == "umr.centre_of_buoyancy"
                            else f"{reason}; tilted by {alpha_deg:g} deg off the transverse plane (C4)"))
    built.attach(Drag.BEM(spacing=mm(spacing_mm), blob=op.BLOB, patches=None, wall_kind="tube"), body=umr,
                 medium=water)
    built.attach(Magnetic.Torque(), body=umr, field="magnet.magnetic")
    built.attach(Magnetic.Force(), body=umr, field="magnet.magnetic")
    built.attach(Gravity(), body=umr, medium=water)
    built.attach(Contact.Projection(margin=mm(op.MARGIN_MM)), body=umr, surface=tube.inner_surface)
    built.observe(SwimSpeed(umr, along=tube.axis))
    return built


def start_phase(start_s_m: float, offset: float) -> float:
    """The field's direction across the tube at the body's start at t = 0 (the motor at angle 0, the magnet's moment
    along +y), as an angle about +x from +y: the body's initial phase."""
    r = np.asarray([start_s_m, 0.0, -offset - MAGNET_HEIGHT_M])
    moment = np.asarray([0.0, MAGNET_MOMENT, 0.0])
    distance = np.linalg.norm(r)
    field = 3.0 * np.dot(moment, r) * r / distance ** 5 - moment / distance ** 3
    return float(math.atan2(field[2], field[1]))


def plane_tilt_deg(s_m: float, height_m: float = MAGNET_HEIGHT_M + 1.0002e-3) -> float:
    """The tilt of the plane the field turns in at ``s`` (review 1's c1_torque.py): atan(3 h s / (2 h^2 - s^2))."""
    return float(math.degrees(math.atan(3 * height_m * s_m / (2 * height_m ** 2 - s_m ** 2))))


def tilt_of(orientation: np.ndarray) -> np.ndarray:
    """The body's axis's angle from the tube's (rad), from its orientation quaternions (w, x, y, z)."""
    w, x, y, z = (orientation[..., i] for i in range(4))
    axis_x = 1.0 - 2.0 * (y * y + z * z)
    return np.arccos(np.clip(np.abs(axis_x), -1.0, 1.0))


def window(alpha_deg: float, sense: int, save: Save, *, spacing_mm: float = SPACING_MM,
           deadline_unix: float | None = None, start_s_m: float = 0.0, steps: int | None = None) -> dict[str, Any]:
    from mime.units import ms

    dt = DT_MS * 1e-3
    period_steps = round(1.0 / (FREQUENCY_HZ * dt))
    total = round(TRANSIENT_S / dt) + TURNS * period_steps if steps is None else steps
    started = time.perf_counter()
    built = experiment(alpha_deg, sense, spacing_mm, start_s_m).build()
    out: dict[str, Any] = {"alpha_deg": alpha_deg, "sense": sense, "spacing_mm": spacing_mm, "dt_ms": DT_MS,
                           "start_s_m": start_s_m, "plane_tilt_at_start_deg": plane_tilt_deg(start_s_m),
                           "steps_planned": total, "build_seconds": time.perf_counter() - started, "chunks": []}
    poses, least, accepted_all = [], [], []
    step = 0
    while step < total:
        if deadline_unix is not None and time.time() > deadline_unix:
            out["stopped"] = {"reason": "deadline", "step": step}
            break
        begun = time.perf_counter()
        result = built.run(ms(DT_MS * CHUNK), history=True)
        history = result.history
        assert history is not None
        resistance = np.asarray(history[op.DRAG_NODE]["resistance"], dtype=np.float64)
        accepted = np.asarray(history[op.DRAG_NODE]["wall_accepted"]).astype(bool)
        pose = np.asarray(history[op.BODY_NODE]["pose"], dtype=np.float64)
        eigen = np.asarray([bem.smallest_eigenvalue(r) if np.all(np.isfinite(r)) else -math.inf for r in resistance])
        bad = np.nonzero(~(np.isfinite(eigen) & (eigen > 0.0) & accepted & np.all(np.isfinite(pose), axis=1)))[0]
        keep = CHUNK if bad.size == 0 else int(bad[0])
        poses.extend(pose[:keep].tolist())
        least.extend(eigen[:keep].tolist())
        accepted_all.extend(accepted[:keep].tolist())
        out["chunks"].append({"first_step": step, "seconds": time.perf_counter() - begun,
                              "least_eigenvalue_nd": float(np.min(eigen)) if np.all(np.isfinite(eigen)) else None,
                              "tilt_max_deg": float(np.degrees(np.nanmax(tilt_of(pose[:, 3:])))),
                              "all_accepted": bool(accepted.all())})
        if bad.size:
            i = int(bad[0])
            out["stopped"] = {"reason": "check", "step": step + i + 1, "time_s": (step + i + 1) * dt,
                              "s_m": float(pose[i, 0]) if np.all(np.isfinite(pose[i])) else
                              (float(poses[-1][0]) if poses else start_s_m),
                              "wall_accepted": bool(accepted[i]), "least_eigenvalue_nd": float(eigen[i]),
                              "pose_finite": bool(np.all(np.isfinite(pose[i]))),
                              "tilt_deg": float(np.degrees(tilt_of(pose[i, 3:]))) if np.all(np.isfinite(pose[i]))
                              else None,
                              "tilt_before_deg": float(np.degrees(tilt_of(pose[i - 1, 3:]))) if i > 0 else
                              (float(np.degrees(tilt_of(np.asarray(poses[-1])[3:]))) if poses else 0.0)}
            break
        step += CHUNK
        held = np.asarray(poses)
        if len(held) >= period_steps:
            moved = sense * (held[-1, 0] - held[-period_steps, 0])
            if moved <= 0.0:
                out["stopped"] = {"reason": "stall", "step": step, "moved_last_turn_m": float(moved)}
                break
        out["steps_done"] = step
        out["tilt_now_deg"] = float(np.degrees(tilt_of(held[-1, 3:])))
        save(out)
    held = np.asarray(poses)
    out["steps_done"] = int(len(held))
    out["least_eigenvalue_nd_seen"] = float(np.min(least)) if least else None
    out["tilt_max_deg_kept"] = float(np.degrees(tilt_of(held[:, 3:]).max())) if len(held) else None
    out["x_moved_m"] = float(held[-1, 0] - held[0, 0]) if len(held) else None
    later = [c["seconds"] for c in out["chunks"][1:]]
    out["seconds_per_step_after_the_first_chunk"] = float(np.median(later)) / CHUNK if later else None
    if len(held):
        out["tilt_deg_last_turn"] = {"mean": float(np.degrees(tilt_of(held[-period_steps:, 3:]).mean())),
                                     "max": float(np.degrees(tilt_of(held[-period_steps:, 3:]).max()))}
    if len(held) == total and "stopped" not in out:
        window_start = round(TRANSIENT_S / dt)
        tail = held[window_start:]
        out["speed_along_pass_m_s"] = float(sense * (tail[-1, 0] - tail[0, 0]) / ((len(tail) - 1) * dt))
        tilts = tilt_of(tail[:, 3:])
        per_turn = [float(np.degrees(tilts[k * period_steps:(k + 1) * period_steps].mean())) for k in range(TURNS)]
        out["cone_angle_deg_per_turn"] = per_turn
        out["cone_angle_deg"] = float(np.mean(per_turn))
        out["cone_angle_scatter_deg"] = float(np.std(per_turn, ddof=1))
        out["complete"] = True
    out["seconds"] = time.perf_counter() - started
    save(out)
    return out


def summary(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """The two-direction mean speed per alpha, the cone angles, and the planted alpha's control."""
    by = {(w["alpha_deg"], w["sense"]): w for w in windows}
    rows = {}
    for alpha in ALPHAS_DEG:
        pair = [by.get((alpha, s)) for s in (1, -1)]
        complete = all(w is not None and w.get("complete") for w in pair)
        rows[f"{alpha:g}"] = {
            "complete": complete,
            "stops": [None if w is None else w.get("stopped") for w in pair],
            "two_direction_mean_m_s": float(np.mean([w["speed_along_pass_m_s"] for w in pair])) if complete else None,
            "cone_angle_deg": [None if w is None else w.get("cone_angle_deg") for w in pair]}
    planted = [by.get((PLANTED_DEG, s)) for s in (1, -1)]
    control: dict[str, Any] = {"planted_deg": PLANTED_DEG}
    if all(w is not None and w.get("complete") for w in planted):
        cone = float(np.mean([w["cone_angle_deg"] for w in planted]))
        scatter = float(max(w["cone_angle_scatter_deg"] for w in planted))
        allowed = max(0.2 * PLANTED_DEG, scatter)
        control.update(runnable=True, recovered_deg=cone, allowed_deg=allowed,
                       holds=abs(cone - PLANTED_DEG) <= allowed)
    else:
        control.update(runnable=False, holds=None,
                       note="a window of the planted alpha did not complete, so the control could not run: no pass")
    return {"per_alpha": rows, "control_planted_alpha": control}
