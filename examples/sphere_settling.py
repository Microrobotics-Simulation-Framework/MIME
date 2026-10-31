"""A steel sphere settling in still water, in free space: the smallest experiment with an answer to check.

``sphere_settling(radius)`` is the experiment; ``main()`` runs it and compares the settled speed with Stokes's law,
``2/9 (rho_s - rho_f) g a^2 / mu``, which the boundary-element rung should reproduce for a sphere in free space. A
sphere of 10 um radius settles at a Reynolds number inside the rung's declared range, so the comparison is a fair one;
one of 50 um settles at 3.7, outside it: the run's result flags that, with the size of the inertia the rung leaves out.

    python examples/sphere_settling.py                  # a sphere of 10 um radius
    python examples/sphere_settling.py --radius-um 50   # the same, past the rung's range: the flag is raised

Starting sizes, measured on the CPU of a laptop and on CI's runners (see ``docs/choosing_spacing.md``): ``spacing`` a
quarter of the radius and ``dt`` 0.1 ms, a run of 3 ms for a radius of 10 um (longer for a larger one); the table there says what a finer spacing costs and what it
gains. Nothing here is a record: the steel's and the water's numbers are declared, each with its reason.
"""

import argparse
import math

from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity
from mime.observations import SwimSpeed
from mime.results import RunResult
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import Quantity, kg_per_m3, m, m_per_s, m_per_s2, mm_per_s, mPa_s, ms, rad_per_s, um

STEEL, WATER, VISCOSITY, GRAVITY = 7850.0, 998.0, 1.0e-3, 9.81
#: The run, for the sphere of 10 um: its relaxation time to its settling speed, 2 a^2 rho_s / 9 mu, is 0.17 ms, so 3 ms
#: is many of them; a step of 0.1 ms, and the speed read over the last millisecond. A sphere of another radius takes
#: the same steps over times scaled by its radius squared, which is how its relaxation time scales.
STEP_MS, DURATION_MS, SETTLED_MS = 0.1, 3.0, 2.0


def scale(radius_um: float) -> float:
    return (radius_um / 10.0) ** 2


def duration(radius_um: float = 10.0) -> Quantity:
    """The run's length (30 steps at the default step)."""
    return ms(DURATION_MS * scale(radius_um))


def sphere_settling(radius_um: float = 10.0, spacing_per_radius: float = 0.25, dt_ms: float | None = None) -> Experiment:
    """The sphere of ``radius_um`` released from rest in water, its drag by the boundary-element rung with its samples
    ``spacing_per_radius`` of the radius apart, stepped every ``dt_ms``."""
    exp = Experiment("sphere_settling", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(GRAVITY),
                     dt=ms(STEP_MS * scale(radius_um) if dt_ms is None else dt_ms))
    water = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(WATER), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(VISCOSITY * 1e3), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(radius_um)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(STEEL), reason="steel")
    exp.attach(Drag.BEM(spacing=um(radius_um * spacing_per_radius), blob=0.5), body=ball, medium=water)
    exp.attach(Gravity(), body=ball, medium=water)
    exp.observe(SwimSpeed(ball, along=-Z, after=ms(SETTLED_MS * scale(radius_um))))
    return exp


def stokes_speed(radius_um: float) -> float:
    """Stokes's settling speed (m/s) of the sphere in water: where the drag is ``6 pi mu a U`` and it balances the
    weight less the buoyancy."""
    return 2.0 / 9.0 * (STEEL - WATER) * GRAVITY * (radius_um * 1e-6) ** 2 / VISCOSITY


def run(radius_um: float = 10.0, spacing_per_radius: float = 0.25) -> tuple[RunResult, float]:
    """The run, with the history kept so the watch reads the peak, and the ratio of the settled speed to Stokes's."""
    exp = sphere_settling(radius_um, spacing_per_radius)
    result = exp.build().run(duration(radius_um), history=True)
    speed = float(result["swim_speed:ball"].mean_speed.to(m_per_s))
    return result, speed / stokes_speed(radius_um)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radius-um", type=float, default=10.0)
    parser.add_argument("--spacing", type=float, default=0.25, help="the samples' spacing, as a fraction of the radius")
    arguments = parser.parse_args()
    result, ratio = run(arguments.radius_um, arguments.spacing)
    print(result.text())
    print(f"settled speed {result['swim_speed:ball'].mean_speed.to(mm_per_s):.4g} mm/s; "
          f"Stokes's law {stokes_speed(arguments.radius_um) * 1e3:.4g} mm/s; ratio {ratio:.4f}")
    if math.isnan(ratio):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
