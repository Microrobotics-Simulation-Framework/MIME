"""Camera PR 5a's probe case: an in-loop coil seen at a size, G19's frame cost through the real node
(``node_measure.py --case cases:coil_camera --path scripts/camera_route2``)."""

from __future__ import annotations

import math

from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.observations import Camera, Detector, World
from mime.scene import Helix, Scene, Z
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um

ROOT = math.sqrt(0.5)


def coil_camera(width: int = 160) -> Experiment:
    """A coil (radius 25 um, pitch 30 um, 3 turns, filament 6 um) turning as it moves, seen from 1 mm by a camera
    that renders a frame at every base step: a step's time is its frame's."""
    height = width * 3 // 4
    scale = width / 160.0
    exp = Experiment("coil_camera", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    exp.add_body(Body("coil", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((2e-3, 0.0, 0.0)),
        angular_velocity=rad_per_s((5.0, 0.0, 0.0))),
        parts={"wire": Part(Helix(radius=um(25), pitch=um(30), turns=3.0, filament_radius=um(6)))}))
    exp.declare("coil.parts.wire.density", kg_per_m3(7850.0), reason="steel")
    exp.observe(Camera(detector=Detector(width=width, height=height, fx=750.0 * scale, fy=750.0 * scale,
                                         cx=(width - 1) / 2, cy=(height - 1) / 2),
                       mount=World(position=m((0.0, -1e-3, 0.0)), orientation=(ROOT, -ROOT, 0.0, 0.0)),
                       frame_period=ms(1)))
    return exp
