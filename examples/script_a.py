"""Script A: a de Jongh FL-9 screw on the floor of a 1/4" tube of water, turned by a uniform field rotating about the
tube's axis at 10 Hz and 1.2 mT, swims along the tube.

This is its reproduction arm: every number is the one the 0.2 line's run of A used (its frozen reference, in
``tests/differential/references/script_a/``), declared in the experiment with where it comes from. Materials will
come from OpenMaterica's records in a later arm; until then each value is written here.

    python examples/script_a.py                     # float32
    JAX_ENABLE_X64=1 python examples/script_a.py    # float64

It prints the build report, then, over the second half of a 5 s run (the reference's window), the swim speed along
the tube and the body's rotation rate about it. ``tests/differential/test_script_a.py`` holds this experiment to the
0.2 reference, with every expected difference declared before the comparison was first run.
"""

import numpy as np

from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import UniformRotatingField
from mime.interactions import Contact, Drag, Gravity, Lubrication, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Scene, Tube, X, Z
from mime.units import (
    A_m2,
    Hz,
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


def script_a(frequency=Hz(10.0), dt=ms(0.5)) -> Experiment:
    # The world: up is +z, so gravity points along -z.
    experiment = Experiment("script_a", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=dt)

    # The 1/4" silicone tube, along +x, and the water inside it.
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    experiment.declare("water.density", kg_per_m3(1000.0), reason=REFERENCE)
    experiment.declare("water.viscosity", mPa_s(1.0), reason=REFERENCE)

    # FL-9, resting on the tube's floor 1 mm below the axis, its own axis along the tube's.
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
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
    experiment.declare("umr.extent", mm(1.33 * 1.56),
                       reason=f"{REFERENCE}: the body's outer radius, (1 + 0.33) times its radius")
    experiment.declare("umr.helix_wavenumber", dimensionless(2.33), reason=f"{REFERENCE}: FL-9's nu")
    experiment.declare("umr.length", mm(7.47), reason=f"{REFERENCE}: FL-9's length")
    experiment.declare("umr.radius", mm(1.56), reason=f"{REFERENCE}: FL-9's cylinder radius")

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


def main() -> None:
    built = script_a().build()
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
