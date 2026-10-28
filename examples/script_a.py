"""Script A: a de Jongh FL-9 screw on the floor of a 1/4" tube of water, turned by a uniform field rotating about the
tube's axis at 10 Hz and 1.2 mT, swims along the tube.

``script_a()`` is its reproduction arm: every number is the one the 0.2 line's run of A used (its frozen reference,
in ``tests/differential/references/script_a/``), declared in the experiment with where it comes from.
``record_arm()`` is the same script with records in place of the values that have them: water's density and
viscosity at 20 degC from OpenMaterica's example record of water (IAPWS), and the moment from N45's remanence (a
vendor's grade limits) through the body's two magnet parts. The body's resin has no record, so its density stays
declared.

    python examples/script_a.py                     # float32
    JAX_ENABLE_X64=1 python examples/script_a.py    # float64
    python examples/script_a.py --records           # the record arm

It prints the build report, then, over the second half of a 5 s run (the reference's window), the swim speed along
the tube and the body's rotation rate about it. ``tests/differential/test_script_a.py`` holds this experiment to the
0.2 reference, with every expected difference declared before the comparison was first run.
"""

import sys

import numpy as np

from mime import Apparatus, Body, Experiment, Medium, PermanentMagnet, State, Structure
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import UniformRotatingField
from mime.interactions import Contact, Drag, Gravity, Lubrication, Magnetic
from mime.materials import Acceptance, Nominal, Record, packaged_record
from mime.observations import SwimSpeed
from mime.scene import DeJonghHelix, Scene, SolidBox, Tube, X, Z
from mime.things import Part
from mime.units import (
    A_m2,
    Hz,
    T,
    degC,
    dimensionless,
    kg,
    kg_m2,
    kg_per_m3,
    m,
    m3,
    m_per_s,
    m_per_s2,
    mm,
    mPa_s,
    ms,
    mT,
    rad_per_s,
    s,
)

REFERENCE = "the 0.2 line's script A (MIME-port041 at 6485516)"
#: FL-9: R 1.56 mm, L 7.47 mm, nu 2.33, modulation 0.33, two starts (de Jongh et al. 2025, Table 1).
FL9 = DeJonghHelix(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33))


def script_a(frequency=Hz(10.0), dt=ms(0.5)) -> Experiment:
    # The world: up is +z, so gravity points along -z.
    experiment = Experiment("script_a", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=dt)

    # The 1/4" silicone tube, along +x, and the water inside it.
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    experiment.declare("water.density", kg_per_m3(1000.0), reason=REFERENCE)
    experiment.declare("water.viscosity", mPa_s(1.0), reason=REFERENCE)

    # FL-9, resting on the tube's floor 1 mm below the axis, its own axis along the tube's. Its shell is the de Jongh
    # helix (de Jongh et al. 2025, Eq. 1 and Table 1), which gives the drag its shape and the body its extent.
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(FL9)}))
    experiment.declare("umr.mass", kg(8.491116020725203e-05),
                       reason=f"{REFERENCE}: its m_eff, (410 + 1000) kg/m^3 times FL-9's volume, with no added mass")
    experiment.declare("umr.inertia", kg_m2((1.0331989974018431e-10,) * 3),
                       reason=f"{REFERENCE}: its I_eff, m R^2 / 2 for a solid cylinder of FL-9's radius, used about "
                              f"every axis alike")
    experiment.declare("umr.buoyant_volume", m3(6.022068099805108e-08), reason=f"{REFERENCE}: FL-9's volume")
    experiment.declare("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0)),
                       reason="the reference's body has one density, so its centre of buoyancy is its centre of mass")
    experiment.declare("umr.magnetic_moment", A_m2((0.0, 2 * 8.4e-4, 0.0)),
                       reason=f"{REFERENCE}: two magnets of 8.4e-4 A m^2, across the body's axis")

    # The field, starting along the body's moment and turning right-handed about the tube's axis.
    field = experiment.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotatingField(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": frequency, "amplitude": mT(1.2)}))

    experiment.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15))),
                      body=umr, medium=water)
    experiment.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    experiment.observe(SwimSpeed(umr, along=tube.axis))
    return experiment


def record_arm(frequency=Hz(10.0), dt=ms(0.5)) -> Experiment:
    # Every record is read at the experiment's temperature, 20 degC, within half a kelvin.
    experiment = Experiment("script_a", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=dt,
                            conditions={"temperature": degC(20.0, plus_minus=0.5)})
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))

    # Water from OpenMaterica's example record, evaluated from the IAPWS formulations. Its viscosity states no shear
    # rate, so OpenMaterica answers "unknown": the experiment accepts that, with its reason.
    water_record = Record(
        "water", packaged_record("openmaterica_example_water"), require={"test_method": ("reference_formulation",)},
        accept={"fluid.dynamic_viscosity": Acceptance(
            ("condition_unstated:shear_rate",),
            reason="IAPWS R12-08, from which the record's viscosity is evaluated, states no shear rate: A takes "
                   "water as Newtonian at its shear rates, as its reference's constant viscosity did")})
    water = experiment.add_medium(Medium("water", fills=tube.lumen, material=water_record))

    # N45's remanence is a vendor's interval with nothing behind it: the experiment accepts what OpenMaterica cannot
    # check, and says which value inside the bounds it uses.
    n45 = Record(
        "n45", packaged_record("openmaterica_example_ndfeb_n45"),
        accept={"magnetic.remanence": Acceptance(
            ("stated_without_origin", "condition_unstated:temperature", "condition_unstated:specimen_shape",
             "condition_unstated:test_method"),
            reason="a vendor's limits for the grade, the only source at hand for these magnets (the 0.2 line names "
                   "them as that vendor's 1 mm N45 cubes); they hold at 'room temperature' with no number, for no "
                   "stated specimen or test method, and with no stated origin")},
        nominal={"magnetic.remanence": Nominal(
            T(1.345), reason="the middle of the grade's limits: nothing measured places these magnets within them")})

    # FL-9 from its parts: the screw, and two 1 mm N45 cubes inside it, magnetised across its axis.
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={
        "shell": Part(FL9),
        "magnet_a": Part(SolidBox(size=mm((1.0, 1.0, 1.0))), material=n45, at=mm((-0.5, 0.0, 0.0)), inside="shell"),
        "magnet_b": Part(SolidBox(size=mm((1.0, 1.0, 1.0))), material=n45, at=mm((0.5, 0.0, 0.0)), inside="shell"),
    }, magnets={"magnet_a": PermanentMagnet(direction=(0.0, 1.0, 0.0)),
                "magnet_b": PermanentMagnet(direction=(0.0, 1.0, 0.0))}))
    experiment.declare("umr.parts.shell.density", kg_per_m3(1410.0),
                       reason="no record yet: the printed resin has none. The reference's mean density of the whole "
                              "body, magnets included")
    experiment.declare("umr.inertia", kg_m2((1.0331989974018431e-10,) * 3),
                       reason=f"{REFERENCE}: its I_eff, kept in place of the parts' own, which is a model change")

    field = experiment.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotatingField(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": frequency, "amplitude": mT(1.2)}))
    experiment.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=mm(0.15))),
                      body=umr, medium=water)
    experiment.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    experiment.observe(SwimSpeed(umr, along=tube.axis))
    return experiment


def main() -> None:
    built = (record_arm() if "--records" in sys.argv[1:] else script_a()).build()
    print(built.report.text())
    result = built.run(s(5.0), history=True)
    assert result.history is not None
    pose, twist = result.history["umr"]["pose"], result.history["umr"]["twist"]
    second_half = slice(len(pose) // 2 - 1, None)  # from the state after 2.5 s to the last
    along = np.asarray(pose[second_half, 0], dtype=np.float64)
    speed = (along[-1] - along[0]) / 2.5
    rotation = float(np.mean(np.asarray(twist[second_half, 3], dtype=np.float64))) / (2 * np.pi)
    print(f"\nover 2.5 to 5 s: swim speed along the tube {speed * 1e3:.6f} mm/s, rotation about it {rotation:.6f} Hz")


if __name__ == "__main__":
    main()
