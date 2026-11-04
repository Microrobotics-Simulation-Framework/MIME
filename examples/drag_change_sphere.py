"""How drag 0.4.0 moves the MRS rung's results against drag 0.3.0: a free steel sphere of 100 um settling in water.

The changelog's line on drag 0.4.0 states these numbers. Each run releases the sphere (50 um in radius) from rest in
water, its drag the MRS rung at ``spacing=um(20)`` and ``blob=0.5``, stepped every 2.5 ms for 75 ms, its settling
speed read from 50 ms. It runs four times: the drag of now (0.4.0) and drag 0.3.0, each on the sphere of now (0.2.0)
and on the sphere 0.1.0, both kept registered (``mime.interactions.KEPT`` and ``mime.scene.KEPT``). Drag 0.3.0 on the
sphere 0.1.0 is what drag 0.3.0 gave; the drag of now gives one number on both spheres.

    python examples/drag_change_sphere.py

The sphere settles at a Reynolds number of about 3.7, outside the rung's declared range, so neither speed says which is
nearer the truth: the ratio to Stokes's law is printed beside each, for scale, not as a check.
"""

from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import KEPT as DRAGS
from mime.interactions import Drag, Gravity
from mime.observations import SwimSpeed
from mime.scene import KEPT as SHAPES
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, m_per_s2, mPa_s, ms, rad_per_s, standard_gravity, um

STEEL, WATER, VISCOSITY, RADIUS = 7850.0, 998.0, 1.0e-3, 50e-6
#: The drags and the spheres the four runs pair: the entries of now, and the ones each replaced.
DRAG_030, SPHERE_010 = DRAGS[("drag", "0.3.0")][0], SHAPES[("sphere", "0.1.0")][0]
RUNS = {"drag 0.4.0, sphere 0.2.0": (Drag, Sphere), "drag 0.4.0, sphere 0.1.0": (Drag, SPHERE_010),
        "drag 0.3.0, sphere 0.1.0": (DRAG_030, SPHERE_010), "drag 0.3.0, sphere 0.2.0": (DRAG_030, Sphere)}


def settling(drag: type, sphere: type) -> Experiment:
    """The sphere ``sphere`` makes, released from rest in water, its drag ``drag``'s MRS rung."""
    exp = Experiment("drag_change_sphere", scene=Scene.new(up=Z, units="m"), dt=ms(2.5))
    water = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(WATER), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(VISCOSITY * 1e3), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(sphere(radius=um(RADIUS * 1e6)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(STEEL), reason="steel")
    exp.attach(drag.MRS(spacing=um(20.0), blob=0.5), body=ball, medium=water)
    exp.attach(Gravity(), body=ball, medium=water)
    exp.observe(SwimSpeed(ball, along=-Z, after=ms(50.0)))
    return exp


def speed(drag: type, sphere: type) -> float:
    """The settling speed (m/s) over the last 25 ms of the 75 ms run."""
    result = settling(drag, sphere).build().run(ms(75.0))
    return float(result["swim_speed:ball"].mean_speed.to(m_per_s))


def stokes_speed() -> float:
    """Stokes's settling speed (m/s): where the drag ``6 pi mu a U`` balances the weight less the buoyancy."""
    gravity = float(standard_gravity().to(m_per_s2))
    return 2.0 / 9.0 * (STEEL - WATER) * gravity * RADIUS**2 / VISCOSITY


def main() -> dict[str, float]:
    speeds = {name: speed(*pair) for name, pair in RUNS.items()}
    for name, held in speeds.items():
        print(f"{name}: {held:.6f} m/s, {held / stokes_speed():.3f} of Stokes's law")
    change = speeds["drag 0.4.0, sphere 0.2.0"] / speeds["drag 0.3.0, sphere 0.1.0"] - 1.0
    print(f"drag 0.4.0 against drag 0.3.0 on the sphere it ran on: {100.0 * change:+.1f} %")
    return speeds


if __name__ == "__main__":
    main()
