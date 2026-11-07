"""A body whose inertia is declared as a full tensor, nine numbers row-major, with one product of inertia written
differently in its two places: refused when the experiment is built, since a body's inertia tensor is symmetric."""
# corpus: codes value.not-positive
from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.observations import SwimSpeed
from mime.scene import Scene, Z
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, ms, rad_per_s

exp = Experiment("tumble", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
rod = exp.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((30.0, 20.0, 10.0)))))
exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
exp.declare("rod.inertia", kg_m2((1e-13, 2e-15, 0.0,
                                  -2e-15, 2e-13, 0.0,
                                  0.0, 0.0, 3e-13)), reason="a millimetre rod, a product's sign mistyped")
drift = exp.observe(SwimSpeed(rod, along=Z))
exp.build()  # corpus: here
