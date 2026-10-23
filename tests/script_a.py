"""Script A's body side, written as a script declares it, with the declared values phase 0's frozen reference ran
with: the 1/4" tube, the FL-9 body's mass, inertia, buoyant volume and two-magnet moment, water at 1000 kg/m^3, a
uniform field turning at 10 Hz and 1.2 mT. Drag arrives in 4c, so here nothing resists the body's motion: the round
trip, the lag list and the golden outputs use it as a graph, not as a physics oracle.

The reference's frame had the tube along z and gravity along -y; here the tube is along x and up is z, so gravity is
along -z. The field starts along the body's moment (body y here, the reference's body x)."""

from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.dynamics import RigidInertial
from mime.fields import UniformRotatingField
from mime.interactions import Contact, Gravity, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Scene, Tube, X, Z
from mime.units import A_m2, Hz, kg, kg_m2, kg_per_m3, m, m3, m_per_s, m_per_s2, mm, ms, mT, rad_per_s

#: The reference's recorded provenance: m_eff, I_eff and the volume; the moment is two magnets of 8.4e-4 A m^2; the
#: extent is 1.33 times the body's 1.56 mm radius.
MASS = kg(8.491116020725203e-05)
INERTIA = kg_m2((1.0331989974018431e-10,) * 3)
VOLUME = m3(6.022068099805108e-08)
MOMENT = A_m2((0.0, 2 * 8.4e-4, 0.0))
EXTENT = mm(1.33 * 1.56)
WATER = kg_per_m3(1000.0)
REASON = "A's reference provenance (phase 0, MIME-port041 at 6485516)"


def script_a(name: str = "script_a_body", *, dt=ms(0.5), velocity=m_per_s((0.0, 0.0, 0.0)),
             frequency=Hz(10.0)) -> Experiment:
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=dt)
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((10.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=velocity,
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    field = experiment.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotatingField(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": frequency, "amplitude": mT(1.2)}))
    for path, value in (("umr.mass", MASS), ("umr.inertia", INERTIA), ("umr.buoyant_volume", VOLUME),
                        ("umr.magnetic_moment", MOMENT), ("umr.extent", EXTENT),
                        ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))), ("water.density", WATER)):
        experiment.declare(path, value, reason=REASON if path != "umr.centre_of_buoyancy" else
                           "the reference's body has one density, so its centre of buoyancy is its centre of mass")
    experiment.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    experiment.observe(SwimSpeed(umr, along=tube.axis))
    return experiment
