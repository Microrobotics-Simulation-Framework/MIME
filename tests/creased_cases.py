"""A creased body the in-loop camera images by the sampled route, for the switch scan (``camera/creased``).

No shipped kind takes that route in a node's step yet: the cylinder keeps its closed form there (``route="sampled"`` is
an override of ``render`` alone), and the screw comes with its measure in camera PR 5b-ii. ``oracle.capped_rod`` is
the cylinder registered again under a name of its own and no closed form (registered in this process only), so that the
camera's route by capability takes its sign function and crease measure, the cylinder's own: the scan then reaches the
crease measure's composition, its exact share and its owners from a node's step."""

import math
from dataclasses import dataclass

from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.observations import Camera, Detector, World
from mime.registry import register_geometry
from mime.scene import Cylinder, Scene, Z
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um
from mime.vocabulary import register_namespace

register_namespace("oracle", "0.0.1")


@register_geometry
@dataclass(frozen=True, kw_only=True)
class CappedRod(Cylinder):
    """The cylinder under a name with no closed form registered: imaged by the sampled route and its crease measure."""

    name = "oracle.capped_rod"
    version = "0.0.1"


def creased_camera() -> Experiment:
    """The in-loop camera seeing a capped rod turned towards it (both caps' outlines and the side in view): a small
    detector, as the scan reads the step's structure, not its size."""
    root = math.sqrt(0.5)
    experiment = Experiment("creased_camera", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    rod = CappedRod(radius=um(10.0), length=um(60.0), axis=(1.0, 0.3, 0.1))
    experiment.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(rod)}))
    experiment.declare("rod.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    experiment.observe(Camera(detector=Detector(width=64, height=48, fx=300.0, fy=300.0, cx=31.5, cy=23.5),
                              mount=World(position=m((0.0, -1e-3, 0.0)), orientation=(root, -root, 0.0, 0.0)),
                              frame_period=ms(3)))
    return experiment
