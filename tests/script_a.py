"""Script A, written as a script declares it, with the declared values phase 0's frozen reference ran with: the 1/4"
tube, the FL-9 body's mass, inertia, buoyant volume and two-magnet moment, water at 1000 kg/m^3 and 1 mPa s, a uniform
field turning at 10 Hz and 1.2 mT, and the drag surrogate with lubrication (epsilon 0.15 mm) on the FL-9 shape (nu
2.33, L 7.47 mm, R 1.56 mm, modulation 0.33, two starts): the body's shell, a de Jongh helix, whose kind gives the
drag its shape and the body its extent, its radial reach (1.33 R). With ``drag=False`` it is 4b's body side, where
nothing resists the body's motion. The
round trip, the lag list and the golden outputs use it as a graph; :func:`reproduction_arm`, with the body starting
where the reference's did, is what ``differential/test_script_a.py`` runs against the frozen reference.

:func:`record_arm` is the same script with records in place of the values that have them, switched on one at a time
(``records=``): water's density and its viscosity from OpenMaterica's example ``example/water`` at 20 degC, and the
moment from N45's remanence (``example/ndfeb_n45``) through two magnet parts, which brings the body's parts in.

The reference's frame had the tube along z and gravity along -y; here the tube is along x and up is z, so gravity is
along -z. The field starts along the body's moment (body y here, the reference's body x)."""


from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import UniformRotatingField
from mime.interactions import Contact, Drag, Gravity, Lubrication, Magnetic
from mime.materials import Acceptance, Nominal, Record, packaged_record
from mime.observations import SwimSpeed
from mime.registry import Interaction
from mime.scene import DeJonghHelix, Scene, SolidBox, Tube, X, Z
from mime.things import Part, PermanentMagnet
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
)

#: The reference's recorded provenance: m_eff, I_eff and the volume; the moment is two magnets of 8.4e-4 A m^2; the
#: extent is 1.33 times the body's 1.56 mm radius, which the helix's radial reach gives. The drag's: FL-9's nu and
#: length, the body's radius (the helix kind's) and water's viscosity, and the lubrication's epsilon.
MASS = kg(8.491116020725203e-05)
INERTIA = kg_m2((1.0331989974018431e-10,) * 3)
VOLUME = m3(6.022068099805108e-08)
MOMENT = A_m2((0.0, 2 * 8.4e-4, 0.0))
EXTENT = mm(1.33 * 1.56)
WATER = kg_per_m3(1000.0)
VISCOSITY = mPa_s(1.0)
NU, LENGTH, RADIUS, EPSILON = dimensionless(2.33), mm(7.47), mm(1.56), mm(0.15)
MODULATION = dimensionless(0.33)
REASON = "A's reference provenance (phase 0, MIME-port041 at 6485516)"

#: The record arm's switches: water's density, water's viscosity, and N45's remanence through the magnet parts.
DENSITY_RECORD, VISCOSITY_RECORD, MAGNET_PARTS = "water.density", "water.viscosity", "n45"
ALL_RECORDS = frozenset({DENSITY_RECORD, VISCOSITY_RECORD, MAGNET_PARTS})
#: Every record is read at the experiment's temperature, 20 degC, within half a kelvin.
CONDITIONS = {"temperature": degC(20.0, plus_minus=0.5)}
WATER_RECORD = Record(
    "water", packaged_record("openmaterica_example_water"), require={"test_method": ("reference_formulation",)},
    accept={"fluid.dynamic_viscosity": Acceptance(
        ("condition_unstated:shear_rate",),
        reason="IAPWS R12-08, from which the record's viscosity is evaluated, states no shear rate: A takes water as "
               "Newtonian at its shear rates, as its reference's constant viscosity did")})
N45_RECORD = Record(
    "n45", packaged_record("openmaterica_example_ndfeb_n45"),
    accept={"magnetic.remanence": Acceptance(
        ("stated_without_origin", "condition_unstated:temperature", "condition_unstated:specimen_shape",
         "condition_unstated:test_method"),
        reason="a vendor's limits for the grade, the only source at hand for these magnets (the 0.2 line names them "
               "as that vendor's 1 mm N45 cubes); they hold at 'room temperature' with no number, for no stated "
               "specimen or test method, and with no stated origin")},
    nominal={"magnetic.remanence": Nominal(
        T(1.345), reason="the middle of the grade's limits: nothing measured places these magnets within them")})


def fl9(nu=NU, length=LENGTH) -> DeJonghHelix:
    """FL-9 (de Jongh et al. 2025, Table 1), or the same screw at another ``nu`` or ``length``."""
    return DeJonghHelix(radius=RADIUS, length=length, wavenumber=nu, modulation=MODULATION)


#: FL-9's shell, the de Jongh helix, and its two 1 mm N45 cubes inside it, magnetised across its axis.
PARTS = {
    "shell": Part(fl9()),
    "magnet_a": Part(SolidBox(size=mm((1.0, 1.0, 1.0))), material=N45_RECORD, at=mm((-0.5, 0.0, 0.0)),
                     inside="shell"),
    "magnet_b": Part(SolidBox(size=mm((1.0, 1.0, 1.0))), material=N45_RECORD, at=mm((0.5, 0.0, 0.0)),
                     inside="shell"),
}
MAGNETS = {"magnet_a": PermanentMagnet(direction=(0.0, 1.0, 0.0)),
           "magnet_b": PermanentMagnet(direction=(0.0, 1.0, 0.0))}
SHELL_DENSITY = kg_per_m3(1410.0)
SHELL_DENSITY_REASON = ("no record yet: the printed resin has none. The reference's mean density of the whole body, "
                        "(410 + 1000) kg/m^3, magnets included; where in the shell the magnets sit is not recorded "
                        "here, and nothing derived depends on it while their mass is counted in this density")
INERTIA_REASON = (f"{REASON}: its I_eff, m R^2 / 2 for a solid cylinder of FL-9's radius about every axis alike. The "
                  f"parts' own inertia is a model change, not a record's, and is not switched here")


def script_a(name: str = "script_a", *, dt=ms(0.5), velocity=m_per_s((0.0, 0.0, 0.0)), frequency=Hz(10.0),
             drag: bool | Interaction = True, nu=NU, inner_radius=mm(3.175), start=mm((10.0, 0.0, -1.0)),
             records: frozenset[str] = frozenset(), length=LENGTH) -> Experiment:
    """Script A; ``drag`` is the drag law (``True`` for script A's surrogate with lubrication, ``False`` for none).
    ``records`` names the record switches on (``ALL_RECORDS`` for the record arm); none for the reproduction arm.
    ``length`` is the body's, for the drag's training-domain checks."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=dt,
                            conditions=CONDITIONS if records else None)
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=inner_radius, wall=mm(1.6),
                                                                    length=mm(100.0))))
    water_record = WATER_RECORD if records & {DENSITY_RECORD, VISCOSITY_RECORD} else None
    water = experiment.add_medium(Medium("water", fills=tube.lumen, material=water_record))
    parts = MAGNET_PARTS in records
    shell = Part(fl9(nu, length))
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=start, orientation=(1.0, 0.0, 0.0, 0.0), velocity=velocity,
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={**PARTS, "shell": shell} if parts else {"shell": shell},
        magnets=MAGNETS if parts else {}))
    field = experiment.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotatingField(rotation_axis=tube.axis, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": frequency, "amplitude": mT(1.2)}))
    declared = [("umr.inertia", INERTIA)]
    if parts:
        experiment.declare("umr.parts.shell.density", SHELL_DENSITY, reason=SHELL_DENSITY_REASON)
    else:
        declared += [("umr.mass", MASS), ("umr.buoyant_volume", VOLUME), ("umr.magnetic_moment", MOMENT),
                     ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0)))]
    if DENSITY_RECORD not in records:
        declared.append(("water.density", WATER))
    for path, value in declared:
        reason = (INERTIA_REASON if path == "umr.inertia" and parts else REASON if path != "umr.centre_of_buoyancy"
                  else "the reference's body has one density, so its centre of buoyancy is its centre of mass")
        experiment.declare(path, value, reason=reason)
    if drag is not False:
        if VISCOSITY_RECORD not in records:
            experiment.declare("water.viscosity", VISCOSITY, reason=REASON)
        law = Drag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=EPSILON))
        experiment.attach(law if drag is True else drag, body=umr, medium=water)
    experiment.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    experiment.observe(SwimSpeed(umr, along=tube.axis))
    return experiment


#: Where the reference's body started, (0, -1, 0) mm in its frame: on the tube's floor, 1 mm below the axis.
REFERENCE_START = mm((0.0, 0.0, -1.0))


def reproduction_arm(frequency=Hz(10.0), dt=ms(0.5)) -> Experiment:
    """Script A's reproduction arm: every value the frozen reference ran with, the body starting where it did. The
    example ``examples/script_a.py`` declares the same experiment line by line."""
    return script_a("script_a", frequency=frequency, dt=dt, start=REFERENCE_START)


def record_arm(records: frozenset[str] = ALL_RECORDS, frequency=Hz(10.0), dt=ms(0.5)) -> Experiment:
    """Script A's record arm: the reproduction arm with the ``records`` switches on (all, unless fewer are named),
    the body starting where the reference's did. With every switch on, the example's ``record_arm`` declares the same
    experiment."""
    return script_a("script_a", frequency=frequency, dt=dt, start=REFERENCE_START, records=records)
