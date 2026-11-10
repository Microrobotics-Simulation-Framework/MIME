"""The steel sphere of ``sphere_settling.py`` released at the centre of a closed horizontal tube of water: it settles
across the tube onto its floor and comes to rest there, its drag including the tube's wall.

``sphere_settling_tube()`` is the experiment: the same 10 um steel sphere in water, which now fills the lumen of a
straight tube of 60 um radius lying along x, gravity across it. Its drag is the MRS rung with the tube's wall kind,
read from the shipped table of the tube's image system (``unit_tube_R0.997``: ``mime assets fetch unit_tube_R0.997``),
and the contact that keeps a body inside the lumen holds it 1.5 um off the wall. ``main()`` runs it, prints the speed
early in the fall against Stokes's free-space speed, and writes the run's view as a web page.

    mime assets fetch unit_tube_R0.997      # the tube's table, 209 MB, once
    python examples/sphere_settling_tube.py  # the run (under a minute on a laptop's CPU) and sphere_settling_tube.html

**What the drag includes, and where it stops.** The drag includes the tube's wall, from its table, down to a gap of
about one sample spacing (2.5 um here, a quarter of the radius). Closer than that the coarse MRS alone is too fast on
the last approach to the floor, and the contact, not the fluid, holds the ball on the floor (the run flags
``near_wall`` there). Drag 0.5.0 models near-contact lubrication (its hand-over to a thin film), and this demo writes
it off, ``lubrication=False``, for two reasons: at this spacing the hand-over gap (0.75 of 2.5 um) is 0.19 of the
ball's radius, past the film's window (0.15 of it), where the build's rule would leave it out anyway; and at this scale
the default roughness (1 um on each surface, a cutoff of 1.41 um) floors the film across the lift's band (from 0.94 um
up), so the film would be the roughness's, not the gap's. A finer spacing (2 um or less) and a declared roughness well
under a third of the hand-over gap bring it in; its numbers would be measured again. A flat floor's wall (so a dish)
is a later release's.

**The physics is the infinite tube's; the end caps are drawn, not computed.** The table is the image system of a
Stokeslet in an infinitely long tube (Liron and Shahar 1978), so the two caps, short closed tubes 4 radii either side
of the ball (in 0.3 a tube is the one solid placed away from the origin), are structures the drag never reads. Their
absence does not matter at that distance: a disturbance of Stokes flow in a tube decays exponentially along it (Liron
and Shahar 1978), its slowest mode, of azimuthal order one, as ``exp(-2.568 z / R)``: its eigenvalue, 2.568 + 1.123i
per radius, is solved from the no-slip tube's eigenvalue equation in ``tests/examples/test_sphere_settling_tube.py``
(the same equation's axisymmetric root is 4.466 + 1.467i), which also finds MIME's table of the image system decaying
along the axis as that mode, to 0.1% from 1.5 to 4.5 radii. A cap 4 radii away would see about 3.5e-5 of the
disturbance near the ball, and its own reflection back at the ball less again.

**What it measures.** Early in the fall, at the tube's centre, the sphere settles at 0.70 of Stokes's free-space speed:
the wall's drag, far beyond the rung's own error at this spacing (its settled speed 1.75% below Stokes's in free space,
``sphere_settling.py``). No published value for a sphere moving across a tube at its centre is compared here: none was
found to cite.

Sizes: the step is 2 ms, many times the sphere's relaxation time (0.17 ms), which the rigid body's implicit step takes
without loss of stability; the speed it settles at does not depend on it. The run is 80 ms, the fall about 66 ms of it.
"""

import argparse
from pathlib import Path

import numpy as np

from mime import Body, Experiment, Medium, State, Structure
from mime.dynamics import RigidInertial
from mime.experiment import RunWritten
from mime.interactions import Contact, Drag, Gravity
from mime.observations import SwimSpeed
from mime.scene import Scene, Sphere, Tube, X, Z
from mime.things import Part
from mime.units import kg_per_m3, m_per_s, m_per_s2, mm_per_s, mPa_s, ms, rad_per_s, um

STEEL, WATER, VISCOSITY, GRAVITY = 7850.0, 998.0, 1.0e-3, 9.81
#: The sphere's radius, the tube's inner radius and wall, and where the caps stand from the ball (um).
RADIUS_UM, TUBE_UM, WALL_UM, CAPS_UM = 10.0, 60.0, 10.0, 240.0
#: The drag's samples a quarter of the radius apart (sphere_settling's starting spacing), and the contact's margin:
#: 0.6 of a spacing, outside the band where the confined resistance was measured indefinite (CONFINED_GAP, 0.55).
SPACING_UM, MARGIN_UM = 2.5, 1.5
#: The step and the run (ms); the early speed is read over the first steps, the ball near the tube's centre.
STEP_MS, DURATION_MS, EARLY_MS = 2.0, 80.0, (2.0, 6.0)
#: What the run states beside its numbers, and the view's caption.
LIMIT = ("The drag includes the tube's wall, from its table, down to a gap of about one sample spacing (2.5 um). "
         "Closer than that the coarse MRS alone is too fast on the last approach, and the contact, not the fluid, "
         "holds the ball on the floor: drag 0.5.0 models near-contact lubrication, and this demo writes it off, its "
         "hand-over gap at this spacing lying past the film's window. A flat floor's wall (so a dish) is a later "
         "release's. The end caps are drawn, not computed: the physics is the infinite tube's.")


def sphere_settling_tube() -> Experiment:
    """The sphere at rest at the tube's centre, water filling the lumen, gravity across the tube."""
    exp = Experiment("sphere_settling_tube", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(GRAVITY),
                     dt=ms(STEP_MS))
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(TUBE_UM), wall=um(WALL_UM),
                                                             length=um(2.0 * CAPS_UM))))
    for name, side in (("cap_left", -1.0), ("cap_right", 1.0)):
        # A closed end, drawn: a tube one wall thick whose bore is a pinhole, standing past the tube's end.
        exp.add_structure(Structure(name, geometry=Tube(
            axis=+X, inner_radius=um(0.5), wall=um(TUBE_UM + WALL_UM - 0.5), length=um(WALL_UM),
            centre=um((side * (CAPS_UM + WALL_UM / 2.0), 0.0, 0.0)))))
    water = exp.add_medium(Medium("water", fills=tube.lumen))
    exp.declare("water.density", kg_per_m3(WATER), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(VISCOSITY * 1e3), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(RADIUS_UM)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(STEEL), reason="steel")
    # The coarse MRS alone, as the docstring says why: the hand-over gap is past the film's window at this spacing, and
    # the default roughness would floor the film across the lift's band at this scale.
    exp.attach(Drag.MRS(spacing=um(SPACING_UM), blob=0.1, quadrature_refinement=20, wall_kind="tube",
                        lubrication=False), body=ball, medium=water)
    exp.attach(Gravity(), body=ball, medium=water)
    exp.attach(Contact.Projection(margin=um(MARGIN_UM), spacing=um(MARGIN_UM)), body=ball,
               surface=tube.inner_surface)
    exp.observe(SwimSpeed(ball, along=-Z, after=ms(0.0)))
    return exp


def stokes_speed() -> float:
    """Stokes's settling speed (m/s) of the sphere in unbounded water."""
    return 2.0 / 9.0 * (STEEL - WATER) * GRAVITY * (RADIUS_UM * 1e-6) ** 2 / VISCOSITY


def run(folder: str | Path = "sphere_settling_tube_run") -> RunWritten:
    """The run, written to ``folder`` with its recording (a sample every step) and the speed at every step."""
    built = sphere_settling_tube().build()
    swim = "swim_speed:ball"
    return built.run_to(folder, ms(DURATION_MS), record=[f"{swim}.speed"], stage_every=ms(STEP_MS))


def early_speed(written: RunWritten) -> float:
    """The mean settling speed (m/s) over ``EARLY_MS``, the ball near the tube's centre."""
    speed = written.result["swim_speed:ball"].speed
    return float(speed.between(ms(EARLY_MS[0]), ms(EARLY_MS[1])).mean().to(m_per_s))


def heights(written: RunWritten) -> np.ndarray:
    """The ball's centre at every sample of the recording (m)."""
    from pxr import Usd

    assert written.recording is not None
    stage = Usd.Stage.Open(str(written.recording))
    translate = stage.GetPrimAtPath("/World/ball").GetAttribute("xformOp:translate")
    return np.array([list(translate.Get(code)) for code in translate.GetTimeSamples()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="sphere_settling_tube_run", help="the run folder to write")
    parser.add_argument("--page", default="sphere_settling_tube.html", help="the view's web page to write")
    arguments = parser.parse_args()
    written = run(arguments.out)
    print(written.result.text())
    early, stokes = early_speed(written), stokes_speed()
    centre = heights(written)
    floor = TUBE_UM - RADIUS_UM - float(np.hypot(centre[-1, 1], centre[-1, 2])) * 1e6
    print(f"early speed, at the tube's centre: {early * 1e3:.4g} mm/s; Stokes's law in free space "
          f"{stokes * 1e3:.4g} mm/s; ratio {early / stokes:.3f}")
    last = float(written.result["swim_speed:ball"].speed.values.to(mm_per_s)[-1])
    print(f"at rest {floor:.3f} um off the floor (the contact's margin {MARGIN_UM} um), after a fall of "
          f"{-centre[-1, 2] * 1e6:.1f} um; speed at the end {last:.3g} mm/s")
    print(LIMIT)
    from mime import viz

    page, _ = viz.web(arguments.out, arguments.page, caption=LIMIT)
    print(f"the view: {page}")


if __name__ == "__main__":
    main()
