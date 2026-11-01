"""A body whose moments of inertia were left at zero: refused when the experiment is built, where it used to run to a
NaN (or, before the gyroscopic term was implicit, to an overflow)."""
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
exp.declare("rod.inertia", kg_m2((0.0, 0.0, 0.0)), reason="a millimetre rod, its inertia left at zero")
drift = exp.observe(SwimSpeed(rod, along=Z))
exp.build()  # corpus: here
