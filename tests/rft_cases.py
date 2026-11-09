"""The experiments the rft rung is tested in, and the coupling spike's R6 helix it is held to.

R6's helix (D = 1 in the spike's units, here a millimetre): radius 0.15 D, pitch 0.6 D, 2.5 turns, filament radius 0.03
D, its centreline read at 70 points 0.04 D apart, in water; in free space and on the axis of a tube of diameter D, the
tube's wall by dev1's table. Its swimming speed, U / Omega for the force-free axial balance about the axis's middle,
and the 6 x 6's eigenvalues in D units are R6's, reproduced to rounding on this tree by phase R's first step (a script
written from the spike's formulas against MIME's tube image)."""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.interactions import Drag
from mime.nodes.hydro import RFTDragNode
from mime.nodes.hydro.mrs import wall_correction_named
from mime.registry import Interaction
from mime.scene import Capsule, Ellipsoid, Helix, Scene, Tube, Z
from mime.scene._helix import HelixTurn
from mime.things import Body, Medium, Part, State, Structure
from mime.units import Quantity, kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

WATER = mPa_s(1.0)
#: R6's helix, a millimetre its diameter's unit, and its settings in SI.
D = 1e-3
R6_SETTINGS = {"radius": 0.15 * D, "pitch": 0.6 * D, "turns": 2.5, "filament_radius": 0.03 * D}
R6_HELIX = Helix(radius=mm(0.15), pitch=mm(0.6), turns=2.5, filament_radius=mm(0.03))
R6_SPACING = mm(0.04)
#: R6's numbers (the coupling spike's results/r6.json and results/r6_resistance.json, reproduced by phase R's first
#: step on this tree to 7e-15 relative): U / Omega in D a radian, and the 6 x 6's eigenvalues in D units, mu = 1.
R6_SPEED_FREE = 0.02802703959464379
R6_SPEED_TUBE = 0.02954203968082897
R6_EIGENVALUES_FREE = (0.22867824063052708, 2.488639170392625, 2.551776869433021, 12.515702530962661,
                       12.69369774437864, 13.258199212864017)
R6_EIGENVALUES_TUBE = (-27.267269520643374, -27.132024598270696, -14.652968493126423, 0.24137120651321764,
                       5.475555443713351, 5.594197972467714)
#: The tube R6's helix swims on the axis of: diameter D, along x, the helix's axis on it.
R6_TUBE = Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(0.5), wall=mm(0.1), length=mm(20.0))


def centroid(settings: Mapping[str, float]) -> np.ndarray:
    """A helix's centroid in its axis frame (the axis's middle at the origin), m."""
    return np.asarray(HelixTurn(settings).centroid(), dtype=np.float64)


def rft(**settings: Any) -> Interaction:
    return Drag.RFT(**settings)


def helix_body(name: str, kind: Any, settings: Mapping[str, float], law: Interaction, *, tube: Tube | None = None,
               velocity: Quantity = m_per_s((0.0, 0.0, 0.0)), spin: Quantity = rad_per_s((0.0, 0.0, 0.0)),
               offset: tuple[float, float, float] = (0.0, 0.0, 0.0), profile: str = "exploratory") -> Experiment:
    """One body, a helix's axis's middle at the world's origin plus ``offset`` (its centre of mass at its centroid; any
    other kind's centre there), unturned, in water filling ``tube``'s lumen (or no region), with ``law``."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1), profile=profile)
    if tube is not None:
        experiment.add_structure(Structure("tube", geometry=tube))
        experiment.add_medium(Medium("water", fills="tube.lumen"))
    else:
        experiment.add_medium(Medium("water"))
    experiment.declare("water.viscosity", WATER, reason="water")
    start = tuple(float(c) + d for c, d in zip(centroid(settings) if isinstance(kind, Helix) else np.zeros(3),
                                                offset, strict=True))
    experiment.add_body(Body("helix", dynamics=RigidInertial(), parts={"shell": Part(kind)}, initial=State(
        position=m(start), orientation=(1.0, 0.0, 0.0, 0.0), velocity=velocity,
        angular_velocity=spin)))
    experiment.declare("helix.mass", kg(1e-9), reason="a sub-millimetre body's order")
    experiment.declare("helix.inertia", kg_m2((1e-16, 1e-16, 1e-16)), reason="a sub-millimetre body's order")
    experiment.attach(law, body="helix", medium="water")
    return experiment


def r6_node(*, tube: bool, coefficients: str = "lighthill", counts: tuple[int, ...] = (70,),
            settings: Mapping[str, float] = R6_SETTINGS, on_indefinite: str = "stop") -> RFTDragNode:
    """The rft node of R6's helix built directly (no plan, so no start-pose check: the tube's arm is refused by the
    build, and gate 1 reads its resistance), with the tube's table as the wall or free space."""
    wall = "tube" if tube else "free_space"
    correction = wall_correction_named(wall, "0.1.0")
    import jax.numpy as jnp

    constants = correction.prepare({"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": 0.5 * D}
                                   if tube else {}, jnp.float64)
    return RFTDragNode("drag:helix", 1e-3, viscosity=1e-3, kind="helix", kind_version="0.1.0", counts=counts,
                       part_centre=(0.0, 0.0, 0.0), part_rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                       coefficients=coefficients, wall_kind=wall, wall_kind_version="0.1.0", spacing=0.04 * D,
                       on_indefinite=on_indefinite, **{f"wall_{k}": v for k, v in constants.items()},
                       **{f"shape_{k}": v for k, v in settings.items()})


def about_the_axis(resistance: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """A 6 x 6 about ``centre`` moved to the origin: ``T^T R T`` with ``T = [[I, [d]x], [0, I]]``, ``d`` from the
    centre to the origin."""
    d = -np.asarray(centre, dtype=np.float64)
    shift = np.eye(6)
    shift[:3, 3:] = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    return shift.T @ resistance @ shift


def dimensionless(resistance: np.ndarray, viscosity: float = 1e-3, length: float = D) -> np.ndarray:
    """A 6 x 6 in SI in units of ``viscosity`` and ``length``: the translations' block over ``mu D``, the couplings'
    over ``mu D^2``, the rotations' over ``mu D^3``."""
    scale = np.concatenate([np.full(3, math.sqrt(viscosity * length)), np.full(3, math.sqrt(viscosity * length ** 3))])
    return resistance / (scale[:, None] * scale[None, :])


def swimming_speed(resistance: np.ndarray) -> float:
    """``U / Omega`` of the force-free balance along the body's x (its axis): ``-R[0, 3] / R[0, 0]``, m a radian."""
    return float(-resistance[0, 3] / resistance[0, 0])


def rft_free_space() -> Experiment:
    """The round-trip case: a helix, a capsule and a prolate ellipsoid in water filling no region, each with the rung
    at its defaults, each moving and turning from the start."""
    experiment = Experiment("rft_free_space", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_medium(Medium("water"))
    experiment.declare("water.viscosity", WATER, reason="water")
    kinds = {"coil": R6_HELIX, "rod": Capsule(radius=mm(0.03), height=mm(1.0), axis=(0.0, 0.6, 0.8)),
             "seed": Ellipsoid(semi_axes=mm((0.08, 0.08, 0.8)))}
    places = {"coil": (0.0, 0.0, 0.0), "rod": (5.0, 0.0, 0.0), "seed": (0.0, 5.0, 0.0)}
    for name, kind in kinds.items():
        half, turn = math.cos(0.25), math.sin(0.25)
        experiment.add_body(Body(name, dynamics=RigidInertial(), parts={"shell": Part(kind)}, initial=State(
            position=mm(places[name]), orientation=(half, turn, 0.0, 0.0), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
            angular_velocity=rad_per_s((3.0, -1.0, 2.0)))))
        experiment.declare(f"{name}.mass", kg(1e-9), reason="a sub-millimetre body's order")
        experiment.declare(f"{name}.inertia", kg_m2((1e-16, 2e-16, 3e-16)), reason="a sub-millimetre body's order")
        experiment.attach(rft(wall_kind="free_space"), body=name, medium="water")
    return experiment


#: A helix small against its tube, on the tube's axis (the validity record's tube_k012): its resistance with the table is
#: definite.
SMALL_HELIX = Helix(radius=mm(0.05), pitch=mm(0.2), turns=2.5, filament_radius=mm(0.01))
SMALL_SETTINGS = {"radius": 0.05 * D, "pitch": 0.2 * D, "turns": 2.5, "filament_radius": 0.01 * D}


def rft_tube() -> Experiment:
    """The round-trip case of the rung with a wall: a helix a tenth of its tube's radius on the tube's axis, moving
    along it and turning, with the tube wall kind (it reads the shipped table, the asset ``unit_tube_R0.997``)."""
    return helix_body("rft_tube", SMALL_HELIX, SMALL_SETTINGS, rft(wall_kind="tube"), tube=R6_TUBE,
                      velocity=m_per_s((1e-3, 0.0, 0.0)), spin=rad_per_s((3.0, -1.0, 2.0)))
