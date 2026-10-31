"""A length given in the wrong unit: the sphere's radius as a time."""
# corpus: codes setting.units
from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m_per_s, m_per_s2, mPa_s, ms, rad_per_s, um

exp = Experiment("settle", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.05))
water = exp.add_medium(Medium("water"))
exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=ms(50.0)))}))
exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
exp.attach(Gravity(), body=ball, medium=water)
built = exp.build()  # corpus: here
