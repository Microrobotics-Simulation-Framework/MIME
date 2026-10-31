"""The first-hour walk's T4: a value declared on a thing that has no place for it."""
# corpus: codes value.unknown
from mime import Apparatus, Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Gravity
from mime.scene import Scene, Sphere, X, Z
from mime.things import Part
from mime.units import Hz, kg_per_m3, m_per_s, m_per_s2, mPa_s, ms, mT, rad_per_s, um

exp = Experiment("settle", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.05))
water = exp.add_medium(Medium("water"))
exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(rotation_axis=X, initial_direction=(0.0, 1.0, 0.0))},
                            commands={"frequency": Hz(5.0), "amplitude": mT(1.2)}))
exp.declare("field.frequency", Hz(5.0), reason="the drive's frequency")
exp.attach(Gravity(), body=ball, medium=water)
exp.build()  # corpus: here
