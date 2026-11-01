"""A spin typed with the wrong power of ten, 1e200 rad/s: the gyroscopic term, quadratic in the rate, overflows float64
as well as float32 at the first step and the state turns to inf and then NaN (a spin of 1e20 rad/s runs finite and
wrong, and one of 1e160 rad/s in float64: the outcome of a rate far past what the step resolves depends on rounding). (This file was `nan_large_dt`, a tumbling body at a step of
100 ms, whose explicit term grew without bound; the term is implicit at the midpoint now, MIME-ANO-129, and a rate
times the step of order one runs finite and wrong, which the run does not refuse; a spin this large it does.)"""
# corpus: codes run.non-finite
from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.observations import SwimSpeed
from mime.scene import Scene, Z
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, ms, rad_per_s, s

exp = Experiment("tumble", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
rod = exp.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((1e200, 7e199, 3e199)))))
exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
exp.declare("rod.inertia", kg_m2((1e-13, 2e-13, 3e-13)), reason="a millimetre rod, three different axes")
drift = exp.observe(SwimSpeed(rod, along=Z))
built = exp.build()
result = built.run(s(5.0))  # corpus: here
print(result[drift].mean_speed)
