"""A controller's law returning a write of the wrong shape, run as a callback."""
# corpus: codes control.law-writes
from dataclasses import dataclass

import numpy as np

from mime import Apparatus, Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.gradients import GradientStatus
from mime.interactions import Gravity, Magnetic
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller
from mime.scene import Scene, Sphere, X, Z
from mime.things import Part
from mime.units import A_m2, kg_per_m3, m_per_s, m_per_s2, mPa_s, ms, mT, rad_per_s, um
from mime.vocabulary import register_namespace

register_namespace("mylab", "0.1.0")


@register_controller
@dataclass(frozen=True, kw_only=True)
class Steady(Controller):
    name = "mylab.steady"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = GradientRecord(where="its step", reason="a switch", route="none")
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"frequency": Signal("Hz", ())}

    def law(self, ctx):
        def step(state, reads):  # corpus: here
            return state, {"frequency": np.full(3, 5.0)}

        return ControlLaw(state={}, held={"frequency": np.float32(0.0)}, step=step)


exp = Experiment("settle", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.05))
water = exp.add_medium(Medium("water"))
exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
exp.declare("ball.parts.sphere.density", kg_per_m3(7850.0), reason="steel")
exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(rotation_axis=X, initial_direction=(0.0, 1.0, 0.0))},
                            commands={"amplitude": mT(1.2)}))
exp.declare("ball.magnetic_moment", A_m2((0.0, 1e-9, 0.0)), reason="a magnetised sphere")
exp.attach(Gravity(), body=ball, medium=water)
exp.attach(Magnetic.Torque(), body=ball, field="field.magnetic")
exp.set_controller(Steady(), reads={"pose": "ball.pose"}, writes={"frequency": "field.frequency"}, mode="callback",
                   rate=ms(1))
built = exp.build()
built.run(ms(2))
