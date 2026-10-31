"""The first-hour walk's T2: a plain screw whose parts give a small product of inertia, with no inertia declared."""
# corpus: codes parts.inertia-axes
from mime import Body, Experiment, Medium, State, Structure
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.scene import Scene, Screw, Tube, X, Z
from mime.things import Part
from mime.units import dimensionless, kg_per_m3, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

exp = Experiment("swim", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))
tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                         length=mm(100.0))))
water = exp.add_medium(Medium("water", fills=tube.lumen))
exp.declare("water.density", kg_per_m3(1000.0), reason="water at 20 degC")
exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
fl9 = Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33))
umr = exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(fl9)}))
exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
exp.attach(Gravity(), body=umr, medium=water)
exp.build()  # corpus: here
