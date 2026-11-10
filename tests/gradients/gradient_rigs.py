"""The rigs of the gradient path's tests: the pulled sphere of the declared-range rigs with its pull's force 1e4 times
smaller, so that it moves inside ``Drag.MRS``'s declared Reynolds number (the range rigs' pull moves it at 0.17 m/s, a
Reynolds number of about 365 by its diameter; this one about 0.04), its radius ranged over [1.0, 1.15] mm; and a
periodic tube of water (R 0.5 mm, D/h 8), IMEX Euler under Stokes, driven by a steady and a pulsatile body force, the
lumen's flux its output, which runs the FVM's conjugate gradient each step."""

from __future__ import annotations

import jax.numpy as jnp
from mrs_cases import free_space, mrs
from rigid_oracles import ConstantWrench

from mime.dynamics import FVM, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce
from mime.scene import PeriodicBox, Scene, Sphere, Tube, Z
from mime.things import Medium, Structure
from mime.units import Hz, kg, kg_m2, kg_per_m3, m_per_s2, mm, mPa_s, ms

SPACING = mm(0.6)
#: The range rigs' pull with its force 1e4 times smaller: (N, N m).
GENTLE = (2e-10, -1e-10, 3e-10, 4e-14, 2e-14, -3e-14)
RADIUS = "ball.parts.shell.radius"
LOW, HIGH = 1.0, 1.15
STEPS = 5
WEIGHTS = (0.3, -0.5, 0.7), (0.2, 0.9, -0.4)


def gentle(radius: float = 1.05, *, ranged: tuple[float, float] | None = (LOW, HIGH), pull=GENTLE,
           name: str = "gentle", **settings: object) -> Experiment:
    """The sphere at ``radius`` mm pulled by ``pull``, its radius's range ``ranged`` (mm) declared."""
    experiment = free_space(name, {"ball": Sphere(radius=mm(radius))}, mrs(SPACING, **settings), mass=kg(2e-8),
                            inertia=kg_m2((2e-17, 2e-17, 2e-17)))
    experiment.attach(ConstantWrench(value=pull), body="ball")
    if ranged is not None:
        experiment.declare_range(RADIUS, mm(ranged[0]), mm(ranged[1]), reason="the design range of the ball's radius")
    return experiment


def pose_loss(final) -> jnp.ndarray:
    """The ball's position (in micrometres) and its quaternion's vector part, against fixed weights."""
    pose = final["ball"]["pose"]
    return 1e6 * jnp.dot(pose[:3], jnp.asarray(WEIGHTS[0])) + jnp.dot(pose[4:], jnp.asarray(WEIGHTS[1]))


TUBE_STEPS = 20


def tube(viscosity: float = 1.0, amplitude: float = 0.01, steady: float = 0.01, *, tolerance: float | None = None,
         report_rows: int = 0, name: str = "womersley_tube", material=None, conditions=None) -> Experiment:
    """The periodic tube: water of ``viscosity`` mPa s, a body force of ``steady`` m/s^2 plus ``amplitude`` m/s^2 at 0.64
    Hz (a Womersley number near one; at 0.01 and 0.01 the lumen's Reynolds number stays under the Stokes line's one), its
    solve at ``tolerance`` (each dtype's own when ``None``), its derivative solves' reports kept a step a row for
    ``report_rows`` steps (summed when 0); its water's density from ``material`` read at ``conditions`` where given,
    declared otherwise."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), dt=ms(20.0), conditions=conditions)
    exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2), length=mm(10.0))))
    settings = {} if tolerance is None else {"tolerance": tolerance}
    exp.add_medium(Medium("water", fills="tube.lumen", material=material, fields={"flow": FVM(
        domain=PeriodicBox(length=mm(0.25)), cells_across=8, stokes=True, scheme=Scheme.IMEXEuler(),
        solve_report_rows=report_rows, **settings)}))
    exp.declare("water.viscosity", mPa_s(viscosity), reason="water near 20 C")
    if material is None:
        exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(steady), amplitude=m_per_s2(amplitude),
                         frequency=Hz(0.64)), medium="water")
    return exp


def flux(final) -> jnp.ndarray:
    """The lumen's axial flux (m^3/s) at the run's end."""
    return final["water.flow"]["lumen_flux"]


__all__ = ["GENTLE", "HIGH", "LOW", "RADIUS", "SPACING", "STEPS", "TUBE_STEPS", "WEIGHTS", "flux", "gentle",
           "pose_loss", "tube"]
