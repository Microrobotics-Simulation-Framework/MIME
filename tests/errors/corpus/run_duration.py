"""A run whose duration is not a whole number of base steps."""
# corpus: codes run.duration
from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.scene import Scene, Z
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, ms, rad_per_s

exp = Experiment("drift", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
rod = exp.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
exp.declare("rod.inertia", kg_m2((1e-13, 1e-13, 1e-13)), reason="a millimetre rod")
built = exp.build()
built.run(ms(2.5))  # corpus: here
