"""A tumbling body stepped with a base step far above its rotation's period: the explicit gyroscopic term grows
without bound, and the state turns to inf and then NaN."""
# corpus: codes run.non-finite
from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.observations import SwimSpeed
from mime.scene import Scene, Z
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, ms, rad_per_s, s

exp = Experiment("tumble", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(100.0))
rod = exp.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((30.0, 20.0, 10.0)))))
exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
exp.declare("rod.inertia", kg_m2((1e-13, 2e-13, 3e-13)), reason="a millimetre rod, three different axes")
drift = exp.observe(SwimSpeed(rod, along=Z))
built = exp.build()
result = built.run(s(5.0))  # corpus: here
print(result[drift].mean_speed)
