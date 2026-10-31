"""Friction 4c of the first-hour walk, measured: the swim speed of the walk's T4 screw (the de Jongh surrogate drag, N45
magnets from the packaged record, water from its record, 5 Hz and 1.2 mT) under the inertia declarations a new user
reaches, and over longer runs.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/inertia_arms.py arms <out.json>
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/inertia_arms.py horizon <out.json>

``arms`` runs 3 s and reads the mean speed over (2 s, 3 s] with the diagonal of the parts' tensor (the products of
inertia dropped, which the refusal prints), the principal moments in its place (the eigenvalues, assigned to the axes
they lie nearest), the isotropic number the walker copied from script A, and the walker's hand cylinder, then mixes the
axial and transverse moments of these to find which moves the answer; with the body started 0.2 mm and 0.6 mm off the
axis under the isotropic inertia; and with the diagonal at half and a quarter of the step. ``horizon`` runs 12 s for
three of them and reads the speed over the last second of 3, 6, 9 and 12 s, with the body's largest offset from the axis
in that second, and the largest angle of the body's axis to the tube's (the surrogate cannot see a tilt, and
``tilt_fired`` reports the step the body leaves the bore). Every arm whose axial moment is its smallest, as a real
screw's is, turns broadside about half a second in and leaves the tube's bore; the numbers it printed are in the pull
request that added it. Not collected by pytest."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from mime import Apparatus, Body, Experiment, Medium, PermanentMagnet, State, Structure
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Drag, Lubrication, Magnetic
from mime.materials import Acceptance, Nominal, Record, packaged_record
from mime.observations import SwimSpeed
from mime.scene import Box, Scene, Screw, Tube, X, Z
from mime.things import Part
from mime.units import (
    Hz,
    T,
    degC,
    dimensionless,
    kg_m2,
    kg_per_m3,
    m_per_s,
    m_per_s2,
    mm,
    mT,
    ms,
    rad_per_s,
    s,
)

#: The diagonal of the parts' tensor, as the refusal prints it; the principal moments; script A's isotropic number; and
#: the walker's solid cylinder of FL-9's radius and length (m R^2 / 2 about the axis, m (3 R^2 + L^2) / 12 across it).
DIAGONAL = (1.3043197461318855e-10, 4.631130043564001e-10, 4.570054968582633e-10)
PRINCIPAL = (1.304320e-10, 4.631533e-10, 4.569652e-10)
ISOTROPIC = (1.0331989974018431e-10,) * 3
_MASS, _R, _L = 1410.0 * 6.022068099805107e-08, 1.56e-3, 7.47e-3
CYLINDER = (_MASS * _R**2 / 2, _MASS * (3 * _R**2 + _L**2) / 12, _MASS * (3 * _R**2 + _L**2) / 12)
ARMS = {"diagonal": DIAGONAL, "principal": PRINCIPAL, "isotropic": ISOTROPIC, "cylinder": CYLINDER,
        "principal, transverse moments swapped (a quarter turn about the axis)": (PRINCIPAL[0], PRINCIPAL[2],
                                                                                   PRINCIPAL[1]),
        "diagonal axial, isotropic transverse": (DIAGONAL[0], ISOTROPIC[1], ISOTROPIC[2]),
        "isotropic axial, diagonal transverse": (ISOTROPIC[0], DIAGONAL[1], DIAGONAL[2]),
        "cylinder axial, diagonal transverse": (CYLINDER[0], DIAGONAL[1], DIAGONAL[2]),
        "diagonal axial, cylinder transverse": (DIAGONAL[0], CYLINDER[1], CYLINDER[2]),
        # The discriminating pair: which moment is the smallest decides, not how large the transverse moments are.
        "transverse moments raised to the diagonal's, axial raised past them": (5.0e-10, DIAGONAL[1], DIAGONAL[2]),
        "transverse moments isotropic, axial the smallest": (0.3e-10, ISOTROPIC[1], ISOTROPIC[2])}


def experiment(inertia: tuple[float, float, float], *, offset_mm: float = 0.0, dt_ms: float = 0.5) -> Experiment:
    exp = Experiment("inertia_arms", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(dt_ms),
                     conditions={"temperature": degC(20.0, plus_minus=0.5)})
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                             length=mm(100.0))))
    water = exp.add_medium(Medium("water", fills=tube.lumen, material=Record(
        "water", packaged_record("openmaterica_example_water"), require={"test_method": ("reference_formulation",)},
        accept={"fluid.dynamic_viscosity": Acceptance(("condition_unstated:shear_rate",), reason="Newtonian")})))
    n45 = Record("n45", packaged_record("openmaterica_example_ndfeb_n45"), accept={"magnetic.remanence": Acceptance((
        "stated_without_origin", "condition_unstated:temperature", "condition_unstated:specimen_shape",
        "condition_unstated:test_method"), reason="a vendor's limits")}, nominal={"magnetic.remanence": Nominal(
            T(1.345), reason="the middle of the grade's limits")})
    umr = exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, offset_mm, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={
        "shell": Part(Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                            modulation=dimensionless(0.33), starts=2)),
        "magnet_a": Part(Box(size=mm((1.0, 1.0, 1.0))), material=n45, at=mm((-0.5, 0.0, 0.0)), inside="shell"),
        "magnet_b": Part(Box(size=mm((1.0, 1.0, 1.0))), material=n45, at=mm((0.5, 0.0, 0.0)), inside="shell")},
        magnets={"magnet_a": PermanentMagnet(direction=(0.0, 1.0, 0.0)),
                 "magnet_b": PermanentMagnet(direction=(0.0, 1.0, 0.0))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="mean density, magnets included")
    exp.declare("umr.inertia", kg_m2(inertia), reason="the arm")
    field = exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(
        rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))}, commands={"frequency": Hz(5.0), "amplitude": mT(1.2)}))
    exp.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15))),
               body=umr, medium=water)
    exp.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    exp.observe(SwimSpeed(umr, along=tube.axis, after=s(2.0)))
    return exp


def tilt_of(pose: Any) -> Any:
    """The angle (degrees) of the body's x axis to the tube's (x), whichever way it points, at each recorded step."""
    along = 1.0 - 2.0 * (pose[:, 5] ** 2 + pose[:, 6] ** 2)
    return np.degrees(np.arccos(np.clip(np.abs(along), 0.0, 1.0)))


def arm(name: str, inertia: tuple[float, float, float], **kwargs: Any) -> dict[str, Any]:
    started = time.perf_counter()
    result = experiment(inertia, **kwargs).build().run(s(3.0), history=True)
    twist, pose = np.asarray(result.history["umr"]["twist"]), np.asarray(result.history["umr"]["pose"])  # type: ignore[index]
    last = slice(-round(1.0 / result.dt), None)
    row = {"arm": name, "inertia": inertia, **kwargs, "mean_speed_mm_per_s": float(
        result["swim_speed:umr"].mean_speed.value) * 1e3, "max_offset_mm": float(np.hypot(pose[last, 1], pose[last, 2]).max()
                                                                                      ) * 1e3,
           "rotation_rate_mean": float(twist[last, 3].mean()), "max_tilt_deg": tilt_of(pose)[last].max(),
           "first_tilt_flag_s": next((f.first for f in result.flags if f.path.endswith("tilt_fired")), None),
           "flags": str(result.flags), "seconds": time.perf_counter() - started}
    print(row, flush=True)
    return row


def arms() -> list[dict[str, Any]]:
    rows = [arm(name, inertia) for name, inertia in ARMS.items()]
    rows += [arm(f"isotropic, started {offset} mm off the axis", ISOTROPIC, offset_mm=offset) for offset in (0.2, 0.6)]
    rows += [arm(f"diagonal, step {dt} ms", DIAGONAL, dt_ms=dt) for dt in (0.25, 0.125)]
    return rows


def horizon() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("isotropic", "cylinder", "diagonal"):
        result = experiment(ARMS[name]).build().run(s(12.0), history=True)
        twist, pose = np.asarray(result.history["umr"]["twist"]), np.asarray(result.history["umr"]["pose"])  # type: ignore[index]
        for end in (3, 6, 9, 12):
            held = slice(round((end - 1) / result.dt), round(end / result.dt))
            rows.append({"arm": name, "end_s": end, "mean_speed_mm_per_s": float(twist[held, 0].mean()) * 1e3,
                         "max_offset_mm": float(np.hypot(pose[held, 1], pose[held, 2]).max()) * 1e3,
                         "max_tilt_deg": float(tilt_of(pose)[held].max())})
            print(rows[-1], flush=True)
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("arms", "horizon"):
        print(__doc__, file=sys.stderr)
        return 2
    Path(argv[1]).write_text(json.dumps(arms() if argv[0] == "arms" else horizon(), indent=1, default=float) + "\n",
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
