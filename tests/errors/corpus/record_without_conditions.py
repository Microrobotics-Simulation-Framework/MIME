"""The first-hour walk's T3, first attempt: water from its packaged record, with no conditions to read it at."""
# corpus: codes record.no-pick
from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity
from mime.materials import Record, packaged_record
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m_per_s, m_per_s2, ms, rad_per_s, um

exp = Experiment("settle", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.05))
water = exp.add_medium(Medium("water", material=Record("water", packaged_record("openmaterica_example_water"))))
ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
exp.attach(Drag.MRS(spacing=um(20.0), blob=0.5, wall_kind="free_space"), body=ball, medium=water)
exp.attach(Gravity(), body=ball, medium=water)
exp.build()  # corpus: here
