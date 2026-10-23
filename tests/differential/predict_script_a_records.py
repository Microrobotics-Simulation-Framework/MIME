"""Predict how each record switch changes script A's steady quantities, before the record arm runs:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/predict_script_a_records.py

writes ``script_a_record_switches.json`` beside this file. The record arm is the reproduction arm with records in
place of the inline values that have records, switched on one at a time: water's density and its viscosity at 20 degC
(OpenMaterica's example record ``example/water``, from the IAPWS formulations), and N45's remanence through two magnet
parts (the example ``example/ndfeb_n45``, a datasheet's interval, read as its bounds with a declared value inside
them). Each switch's value is the one the record gives, or the one the parts derive; nothing here reads MIME's seam.

**How.** The same emulation that predicted phase 4's arm (``predict_script_a.py``: 0.3's update rules in NumPy,
float64, with the shared drag model), which reproduced 0.3's reproduction arm to 7e-15 in speed and 3e-16 rad in lag
(phase 4's gate run), is run once per arm with the switched values, and the change from the emulated reproduction is
the prediction. A first-principles estimate is written beside each, and for the switch whose speed change has no
closed form here, the emulation's own control on the mechanism. Not collected by pytest.
"""

from __future__ import annotations

import json
import math
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
OUTPUT = HERE / "script_a_record_switches.json"
sys.path[:0] = [str(HERE), str(HERE.parent)]

import predict_script_a as emulation  # noqa: E402
from script_a_steady import GATED  # noqa: E402

GRAVITY = 9.81
TRANSIENT = "transient_speed_0_to_0.5_s_m_s"
QUANTITIES = (*GATED, TRANSIENT)

#: The vacuum magnetic permeability, CODATA 2022 (N A^-2), as MIME's ``mime.units.VACUUM_PERMEABILITY``.
MU_0 = 1.25663706127e-6
#: The FL-9 screw as one solid cylinder of its length whose cross-section has the screw's area, pi R^2 (1 + eps^2/2)
#: (the 0.2 code's volume), at the mean density the reference used, magnets included (no record yet).
SHELL_RADIUS = 1.56 * math.sqrt(1 + 0.33 ** 2 / 2) * 1e-3
SHELL_LENGTH = 7.47e-3
SHELL_DENSITY = 1410.0
SHELL_VOLUME = math.pi * SHELL_RADIUS * SHELL_RADIUS * SHELL_LENGTH
#: Two N45 cubes of 1 mm, magnetised across the body's axis, inside the shell, so their mass is the shell's.
MAGNET_EDGE = 1e-3
MAGNET_VOLUME = MAGNET_EDGE * MAGNET_EDGE * MAGNET_EDGE
REMANENCE_BOUNDS = (1.32, 1.37)
REMANENCE = 1.345


@dataclass(frozen=True)
class Values:
    """The values script A's laws read that a record switch changes, in SI."""

    viscosity: float
    fluid_density: float
    mass: float
    buoyant_volume: float
    moment: float


REPRODUCTION = Values(viscosity=1e-3, fluid_density=1000.0, mass=8.491116020725203e-05,
                      buoyant_volume=6.022068099805108e-08, moment=2 * 8.4e-4)
WATER_DENSITY = 998.207       # example/water, bulk.density at 293.15 K
WATER_VISCOSITY = 0.0010016   # example/water, fluid.dynamic_viscosity at 293.15 K
PART_MOMENT = 2 * ((REMANENCE / MU_0) * MAGNET_VOLUME)
PARTS = replace(REPRODUCTION, mass=SHELL_DENSITY * SHELL_VOLUME, buoyant_volume=SHELL_VOLUME, moment=PART_MOMENT)

ARMS: dict[str, Values] = {
    "reproduction": REPRODUCTION,
    "water_density": replace(REPRODUCTION, fluid_density=WATER_DENSITY),
    "water_viscosity": replace(REPRODUCTION, viscosity=WATER_VISCOSITY),
    "n45_through_parts": PARTS,
    "all_records": replace(PARTS, fluid_density=WATER_DENSITY, viscosity=WATER_VISCOSITY),
}


@contextmanager
def emulated_with(values: Values) -> Iterator[None]:
    """The emulation's module constants set to ``values``, with gravity as 0.3's law computes it, ``(m - rho V) g``,
    and restored after."""
    held = (emulation.VISCOSITY, emulation.MASS, emulation.MOMENT, emulation.GRAVITY_FORCE.copy())
    emulation.VISCOSITY, emulation.MASS, emulation.MOMENT = values.viscosity, values.mass, values.moment
    emulation.GRAVITY_FORCE = np.array([0.0, -1.0, 0.0]) * (values.mass * GRAVITY
                                                             - values.fluid_density * values.buoyant_volume * GRAVITY)
    emulation._cache.clear()  # the surrogate's resistance scales with the viscosity
    try:
        yield
    finally:
        emulation.VISCOSITY, emulation.MASS, emulation.MOMENT, emulation.GRAVITY_FORCE = held
        emulation._cache.clear()


#: The scheme the arms are emulated with: phase 4's 0.3 scheme, or (``--corrected``) 4c3's, with its three declared
#: changes made (the increment form, the contact held in the solve, the surrogate about the centre).
SCHEMES = {"phase 4": emulation.SCHEME_0_3, "4c3": emulation.CHANGES["surrogate-about-the-centre"]}
scheme = SCHEMES["phase 4"]


def emulate(values: Values) -> dict[str, float]:
    with emulated_with(values):
        found = emulation.quantities(emulation.emulate(scheme), 10.0)
    return {name: found[name] for name in QUANTITIES}


def change(name: str, value: float, against: float) -> float:
    return emulation.difference(name, value, against)


def axial_torque_only(values: Values) -> dict[str, float]:
    """The emulation with the magnetic torque projected onto the tube's axis: a control on a switch's speed change,
    which then vanishes if it came from the torque's part across the axis."""
    source = Path(emulation.__file__).read_text()
    line = "        torque = np.cross(_rotation(orientation) @ moment, field)\n"
    assert source.count(line) == 1, "the emulation's torque line moved; update this control"
    module = types.ModuleType("axial_torque_emulation")
    module.__file__ = emulation.__file__
    sys.modules[module.__name__] = module  # a dataclass looks its module up while it is defined
    namespace: dict[str, Any] = module.__dict__
    exec(compile(source.replace(line, line + "        torque = (torque @ AXIS) * AXIS\n"), emulation.__file__,  # noqa: S102
                 "exec"), namespace)
    namespace["VISCOSITY"], namespace["MASS"], namespace["MOMENT"] = values.viscosity, values.mass, values.moment
    namespace["GRAVITY_FORCE"] = np.array([0.0, -1.0, 0.0]) * (values.mass * GRAVITY
                                                                - values.fluid_density * values.buoyant_volume
                                                                * GRAVITY)
    held = namespace["SCHEME_0_3"] if scheme is SCHEMES["phase 4"] else namespace["CHANGES"]["surrogate-about-the-centre"]
    found = namespace["quantities"](namespace["emulate"](held), 10.0)
    return {name: found[name] for name in QUANTITIES}


def contact_held(values: Values) -> dict[str, float]:
    """The emulation with the body's contact held inside the implicit solve (phase 4's control on its 'integrator'
    difference): a control on a switch's speed change, which then vanishes if it came from that O(dt) load."""
    with emulated_with(values):
        found = emulation.quantities(emulation.emulate(emulation.CONTACT_HELD), 10.0)
    return {name: found[name] for name in QUANTITIES}


def contact_projected(values: Values) -> dict[str, float]:
    """4c3's scheme with the contact taken back out of the solve (projected after the step, as phase 4 did): the
    control that brings the O(dt) load back."""
    with emulated_with(values):
        found = emulation.quantities(emulation.emulate(replace(SCHEMES["4c3"], integrator="implicit")), 10.0)
    return {name: found[name] for name in QUANTITIES}


def mechanism_controls(values: Values) -> dict[str, float]:
    """The switch's change in the swim speed under each control, beside the change with neither."""
    speed = "swim_speed_axial_m_s"
    return {label: change(speed, control(values)[speed], control(REPRODUCTION)[speed])
            for label, control in (("contact_held", contact_held), ("axial_torque_only", axial_torque_only))}


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    emulated = {name: emulate(values) for name, values in ARMS.items()}
    base = emulated["reproduction"]
    predicted = {name: {q: change(q, found[q], base[q]) for q in QUANTITIES}
                 for name, found in emulated.items() if name != "reproduction"}
    operating = json.loads((HERE / "script_a_expected_differences.json").read_text())["operating_point"]
    lag_term = operating["R_omega_over_m_B"]
    weight_change = (REPRODUCTION.fluid_density - WATER_DENSITY) * REPRODUCTION.buoyant_volume / (
        REPRODUCTION.mass - REPRODUCTION.fluid_density * REPRODUCTION.buoyant_volume)
    integrator_speed = 0.023518259960228577  # phase 4's declared 'integrator' difference in the speed
    moment_ratio = PART_MOMENT / REPRODUCTION.moment
    controls = {name: mechanism_controls(ARMS[name]) for name in ("water_density", "water_viscosity",
                                                                  "n45_through_parts")}
    moment_speed = predicted["n45_through_parts"]["swim_speed_axial_m_s"]
    tolerances = json.loads((HERE / "script_a_expected_differences.json").read_text())["tolerances"]["f64"]

    switches = [
        {
            "id": "water_density",
            "record": "example/water 1.0.0, bulk.density, read at 293.15 K +- 0.5 K (alternative 0, IAPWS R6-95(2018))",
            "value_before": {"water.density": REPRODUCTION.fluid_density},
            "value_after": {"water.density": WATER_DENSITY},
            "reasoning": "The density enters A only through gravity, (m - rho V) g, the body's buoyant weight, which "
                         f"rises by (1000 - 998.207) V / (m - 1000 V) = {weight_change:+.4e}. The weight presses the "
                         "body on the tube's floor. A locked screw's speed does not depend on it at Re -> 0, but 0.3's "
                         "implicit step adds an O(dt) load the body never feels, from the outward velocity of a fall, "
                         "dt F / m, that the projection removes (phase 4's 'integrator' difference, "
                         f"{integrator_speed:+.4e} in the speed), and that load is proportional to the weight. "
                         f"First-principles estimate: the speed changes by {integrator_speed:.4e} x "
                         f"{weight_change:.4e} = {integrator_speed * weight_change:+.3e}, and the rates and the lag "
                         "by about nothing. The mechanism's controls (emulated): with the contact held inside the "
                         "solve, which removes that load, the speed's change falls to "
                         f"{controls['water_density']['contact_held']:+.2e}; with the magnetic torque held to the "
                         f"axis it is {controls['water_density']['axial_torque_only']:+.2e}. The emulation's "
                         "numbers are the prediction.",
            "first_principles": {"swim_speed_axial_m_s": integrator_speed * weight_change},
            "mechanism_controls_speed_change": controls["water_density"],
            "predicted": predicted["water_density"],
        },
        {
            "id": "water_viscosity",
            "record": "example/water 1.0.0, fluid.dynamic_viscosity, read at 293.15 K +- 0.5 K (alternative 0, "
                      "IAPWS R12-08): unknown, its shear rate unstated, accepted by the experiment",
            "value_before": {"water.viscosity": REPRODUCTION.viscosity},
            "value_after": {"water.viscosity": WATER_VISCOSITY},
            "reasoning": "The viscosity scales the surrogate's resistance and the lubrication by 1.0016. The lag "
                         "moves with it: locked, the measured lag is arcsin(R omega / m B) - omega dt (phase 4's lag "
                         f"structure), R proportional to the viscosity, so by about 0.0016 x {lag_term:.5f} / cos = "
                         f"{0.0016 * lag_term / math.sqrt(1 - lag_term ** 2):+.3e} rad. In Stokes flow a screw's "
                         "speed would not move (its thrust and its drag scale alike), but the magnetic torque does "
                         "not scale: what matters is R against m B, so a 0.16% rise in the viscosity acts on the "
                         "torque's part across the axis (see the moment's switch) as a 0.16% fall in the moment "
                         f"would. Estimate from the moment's switch: 0.0016 x {-moment_speed:.4e} / "
                         f"{moment_ratio - 1:.4f} = {0.0016 * -moment_speed / (moment_ratio - 1):+.2e} in the "
                         "speed. The mechanism's controls (emulated): with the magnetic torque held to the axis the "
                         f"speed's change falls to {controls['water_viscosity']['axial_torque_only']:+.2e}; with the "
                         f"contact held it is {controls['water_viscosity']['contact_held']:+.2e}. The "
                         "emulation's numbers are the prediction.",
            "first_principles": {"phase_lag_field_minus_moment_rad": 0.0016 * lag_term / math.sqrt(1 - lag_term ** 2),
                                 "swim_speed_axial_m_s": 0.0016 * -moment_speed / (moment_ratio - 1)},
            "mechanism_controls_speed_change": controls["water_viscosity"],
            "predicted": predicted["water_viscosity"],
        },
        {
            "id": "n45_through_parts",
            "record": "example/ndfeb_n45 0.1.0, magnetic.remanence, read as its bounds [1.32, 1.37] T: unknown "
                      "(no stated origin; temperature, specimen shape and test method unstated), accepted by the "
                      f"experiment, which declares {REMANENCE} T inside them",
            "value_before": {"umr.magnetic_moment": REPRODUCTION.moment, "umr.mass": REPRODUCTION.mass,
                             "umr.buoyant_volume": REPRODUCTION.buoyant_volume},
            "value_after": {"umr.magnetic_moment": PART_MOMENT, "umr.mass": PARTS.mass,
                            "umr.buoyant_volume": PARTS.buoyant_volume},
            "parts": {
                "shell": f"a solid cylinder of FL-9's length, {SHELL_LENGTH} m, and radius {SHELL_RADIUS!r} m, whose "
                         "cross-section has the screw's area, pi R^2 (1 + eps^2/2) (the 0.2 code's volume); density "
                         f"{SHELL_DENSITY} kg/m^3 declared (no record yet: the reference's mean density, magnets "
                         "included)",
                "magnets": f"two cubes of {MAGNET_EDGE} m inside the shell, so their mass is counted in its mean "
                           "density, magnetised along the body's y; moment M V each, M = B_r / mu_0 with "
                           f"mu_0 = {MU_0} N A^-2 (CODATA 2022)",
                "inertia": "declared, the reference's I_eff, in place of the parts' (the derived inertia, a solid "
                           "cylinder's, is a model change, not a record's, and is not switched here)",
            },
            "reasoning": f"The moment rises by {moment_ratio - 1:+.4f} (0.2's 8.4e-4 A m^2 a magnet against M V = "
                         f"{(REMANENCE / MU_0) * MAGNET_VOLUME:.4e}); the mass and buoyant volume move by the parts' "
                         f"rounding only ({PARTS.mass / REPRODUCTION.mass - 1:+.1e}, "
                         f"{PARTS.buoyant_volume / REPRODUCTION.buoyant_volume - 1:+.1e}). Locked, the body turns "
                         "with the field whatever its moment, so the rates do not move; the lag does, arcsin(R "
                         f"omega / m B) falling from {math.asin(lag_term):.6f} to "
                         f"{math.asin(lag_term / moment_ratio):.6f} rad, a change of "
                         f"{math.asin(lag_term / moment_ratio) - math.asin(lag_term):+.4e} rad. The swim speed has "
                         "no closed form here: the torque m x B has a part across the tube's axis once the body's "
                         "axis tilts from it, which a larger moment makes stiffer. The mechanism's controls "
                         "(emulated): with the magnetic torque held to the axis the speed's change falls to "
                         f"{controls['n45_through_parts']['axial_torque_only']:+.2e}; with the contact held it is "
                         f"at {controls['n45_through_parts']['contact_held']:+.2e}. The emulation's numbers are the "
                         "prediction.",
            "first_principles": {"phase_lag_field_minus_moment_rad":
                                 math.asin(lag_term / moment_ratio) - math.asin(lag_term)},
            "mechanism_controls_speed_change": controls["n45_through_parts"],
            "predicted": predicted["n45_through_parts"],
        },
    ]
    summed = {q: sum(s["predicted"][q] for s in switches) for q in QUANTITIES}
    record = {
        "written": "2026-10-23, before MIME's record arm existed (phase 5); by predict_script_a_records.py, from the "
                   "values each record gives and phase 4's emulation",
        "comparison": "each record arm of tests/script_a.py (the reproduction arm with one switch on, then all) "
                      "against the reproduction arm run the same way, by phase 4's window rule (script_a_steady.py), "
                      "float64, 10 Hz, dt 0.5 ms, 5 s",
        "how_predicted": "the emulation of 0.3's scheme (predict_script_a.py) run once per arm with the switched "
                         "values, gravity as 0.3's law computes it, (m - rho V) g; the prediction is the change from "
                         "the emulated reproduction, relative except the lag's (rad)",
        "values": {name: asdict(values) for name, values in ARMS.items()},
        "switches": switches,
        "all_records": {
            "predicted": predicted["all_records"],
            "sum_of_the_switches": summed,
            "reasoning": "the three switches on together; the changes are small enough that they nearly add (the sum "
                         "is beside the prediction, for reading, not gated)",
        },
        "tolerances": {
            "values": {name: tolerances[name] for name in QUANTITIES},
            "units": "on the measured change against the predicted change: relative to the reproduction's value, "
                     "except the lag (rad)",
            "basis": "phase 4's float64 tolerances, unchanged: the emulation reproduced 0.3's reproduction arm to "
                     "7e-15 in speed, 0 in the rates and 3e-16 rad in lag, so they sit far above a second "
                     "implementation's rounding; a predicted change below its tolerance cannot be told from no change, "
                     "and the test reports it as such rather than reading it as agreement",
        },
        "gated": list(QUANTITIES),
        "controls": {
            "reproduction": "the reproduction arm, read through the seam, gives phase 4's derived hash and run "
                            "(references/script_a_phase4_reproduction.json); with one inline value 1% off, it does not",
            "discrimination": "each switch's measured change misses every other switch's prediction on at least one "
                              "gated quantity, and the reproduction misses each switch's prediction on every quantity "
                              "whose predicted change is above ten times its tolerance",
        },
    }
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({name: predicted[name] for name in predicted}, indent=1))
    print(f"wrote {OUTPUT}")


def corrected() -> None:
    """Declare each switch's change on 4c3's physics, appended to the file under ``corrected_physics_4c3``, leaving
    phase 5's declaration as it was; committed before the record arm runs on that physics."""
    global scheme
    import jax

    jax.config.update("jax_enable_x64", True)
    record = json.loads(OUTPUT.read_text())
    if "corrected_physics_4c3" in record:
        raise SystemExit("already declared; a declaration is never rewritten")
    scheme = SCHEMES["4c3"]
    emulated = {name: emulate(values) for name, values in ARMS.items()}
    base = emulated["reproduction"]
    predicted = {name: {q: change(q, found[q], base[q]) for q in QUANTITIES}
                 for name, found in emulated.items() if name != "reproduction"}
    speed = "swim_speed_axial_m_s"
    phase_5 = {entry["id"]: entry["predicted"] for entry in record["switches"]}
    projected = {name: change(speed, contact_projected(ARMS[name])[speed], contact_projected(REPRODUCTION)[speed])
                 for name in ("water_density", "water_viscosity", "n45_through_parts")}
    axial = {name: change(speed, axial_torque_only(ARMS[name])[speed], axial_torque_only(REPRODUCTION)[speed])
             for name in ("water_density", "water_viscosity", "n45_through_parts")}
    density, before = predicted["water_density"][speed], phase_5["water_density"][speed]
    record["corrected_physics_4c3"] = {
        "written": "2026-10-23, by predict_script_a_records.py --corrected, after phase 4c3's three declared changes "
                   "were measured on script A's reproduction arm and before the record arm ran on them",
        "scheme": asdict(SCHEMES["4c3"]),
        "how_predicted": "as phase 5's, with 4c3's scheme: the emulation of 0.3 with the step solved for the change in "
                         "velocity, the contact held inside the implicit solve and the surrogate's resistance about "
                         "the centre (each declared and measured in script_a_expected_differences.json), run once per "
                         "arm with the switched values; the prediction is the change from the emulated reproduction",
        "predicted": predicted,
        "phase_5_predicted": phase_5,
        "controls_speed_change": {"contact_projected_after_the_step": projected, "axial_torque_only": axial},
        "water_density": (
            f"Phase 5 credited the density switch's speed change ({before:+.3e}) to the implicit step's O(dt) load, "
            "the outward velocity of a fall resisted in the solve and then discarded by the projection, proportional "
            "to the buoyant weight; its control, the contact held in the solve, took it to "
            f"{record['switches'][0]['mechanism_controls_speed_change']['contact_held']:+.2e}. Change 2 removes "
            f"that load, so on 4c3's physics the switch's speed change is predicted at {density:+.3e}: what remains "
            "of the weight's effect on a locked screw resting on the wall, through the surrogate about the centre. "
            f"The control: with the contact projected after the step again, it returns to {projected['water_density']:+.3e}."),
        "discrimination": "On 4c3's physics the density switch's predicted change is below ten times its tolerance on "
                          "every gated quantity (the speed -2.6e-8 against 1e-7; the transient -8.7e-6 against 1e-6): "
                          "the record arm cannot tell it from no change, so it is reported as such and not counted as "
                          "agreement; the record's value reaching the gravity law is checked by the derived graph "
                          "(the report's value lines and the parameters they move). Every other switch stays "
                          "resolvable.",
        "tolerances": record["tolerances"],
    }
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["corrected_physics_4c3"], indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--corrected"]:
        corrected()
    else:
        main()
