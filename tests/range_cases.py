"""The rig of the range-warning tests (``tests/build/test_range_warning.py``), shared with the tests of where a warning
is shown (``tests/unit/test_warning_lines.py``): a screw in a medium with no wall, turned by a uniform field faster than
its drag law is declared for, so that its build warns."""

from mime import Apparatus, Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Drag, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Scene, Screw, X, Z
from mime.things import Part
from mime.units import (
    A_m2,
    Hz,
    dimensionless,
    kg_m2,
    kg_per_m3,
    m_per_s,
    m_per_s2,
    mm,
    mPa_s,
    ms,
    mT,
    rad_per_s,
    s,
)

TURNING = "turning Reynolds number"


def rotating_screw(profile: str = "exploratory", *, hz: float = 5.0, viscosity_mpa_s: float = 1.0,
                   density: float = 998.0) -> Experiment:
    """FL-9 in a medium with no wall, a uniform field turning it at ``hz``: the second walk's task in free space (the
    tube's wall table is a download the warning does not need). The inertia is declared, so research builds too."""
    exp = Experiment("rotating", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(5.0), profile=profile)
    medium = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(density), reason="the fluid's density, as stated")
    exp.declare("water.viscosity", mPa_s(viscosity_mpa_s), reason="the fluid's viscosity, as stated")
    umr = exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(Screw(
            radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
            starts=2))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
    exp.declare("umr.inertia", kg_m2((1.3e-10, 4.63e-10, 4.57e-10)), reason="the parts' diagonal, products dropped")
    exp.declare("umr.magnetic_moment", A_m2((0.0, 1.68e-3, 0.0)), reason="two magnets, across the screw's axis")
    field = exp.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotating(rotation_axis=X, initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(hz), "amplitude": mT(5.0)}))
    exp.attach(Drag.MRS(spacing=mm(0.7), blob=0.25, quadrature_refinement=8), body=umr, medium=medium)
    exp.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    exp.observe(SwimSpeed(umr, along=X, after=s(1.0)))
    return exp
