"""A body declared with an inertia of zero: the step divides by it, and the state turns to NaN at the first step.
(The file keeps its first name, from the base step that made a tumbling body's gyroscopic term grow: the term is
implicit at the midpoint now, MIME-ANO-129, and a step far above a body's period no longer overflows it.)"""
# corpus: codes run.non-finite
from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.observations import SwimSpeed
from mime.scene import Scene, Z
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, ms, rad_per_s, s

exp = Experiment("tumble", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
rod = exp.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((30.0, 20.0, 10.0)))))
exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
exp.declare("rod.inertia", kg_m2((0.0, 0.0, 0.0)), reason="a millimetre rod, its inertia left at zero")
drift = exp.observe(SwimSpeed(rod, along=Z))
built = exp.build()
result = built.run(s(5.0))  # corpus: here
print(result[drift].mean_speed)
