"""The rigs of the declared ranges' tests: the pulled sphere of the MRS gradient oracle (``Drag.MRS`` at a written
spacing of 0.6 mm in free space, its mass and inertia declared, pulled by a constant wrench), whose force points are 66,
72 and 91 at radii of 1.0, 1.1 and 1.15 mm; and a ball of the same kind carrying a planted ranged entry
(``planted.RangedShares``) and nothing else, for the controls."""

from __future__ import annotations

from mrs_cases import free_space, mrs
from planted import RangedShares
from rigid_oracles import ConstantWrench

from mime.experiment import Experiment
from mime.scene import Sphere
from mime.units import kg, kg_m2, m_per_s, mm, rad_per_s

SPACING = mm(0.6)
PULL = (2e-6, -1e-6, 3e-6, 4e-14, 2e-14, -3e-14)
RADIUS = "ball.parts.shell.radius"
LOW, HIGH = 1.0, 1.15
REASON = "the design range of the ball's radius"


def pulled(radius: float = 1.0, *, ranged: tuple[float, float] | None = (LOW, HIGH), name: str = "pulled",
           **settings: object) -> Experiment:
    """The pulled sphere at ``radius`` mm, with the range ``ranged`` (mm) of its radius declared, or none."""
    experiment = free_space(name, {"ball": Sphere(radius=mm(radius))}, mrs(SPACING, **settings), mass=kg(2e-8),
                            inertia=kg_m2((2e-17, 2e-17, 2e-17)))
    experiment.attach(ConstantWrench(value=PULL), body="ball")
    if ranged is not None:
        experiment.declare_range(RADIUS, mm(ranged[0]), mm(ranged[1]), reason=REASON)
    return experiment


def moving(radius: float = 1.0, *, ranged: tuple[float, float] | None = (LOW, HIGH)) -> Experiment:
    """The sphere at ``radius`` mm with ``Drag.MRS`` alone, moving and turning from the start, so that it saves and loads
    with MIME's own entries only; the range ``ranged`` (mm) of its radius declared."""
    experiment = free_space("moving", {"ball": Sphere(radius=mm(radius))}, mrs(SPACING), mass=kg(2e-8),
                            inertia=kg_m2((2e-17, 2e-17, 2e-17)), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
                            spin=rad_per_s((3.0, -1.0, 2.0)))
    if ranged is not None:
        experiment.declare_range(RADIUS, mm(ranged[0]), mm(ranged[1]), reason=REASON)
    return experiment


def planted(mode: str, radius: float = 1.0, *, ranged: tuple[float, float] | None = (LOW, HIGH)) -> Experiment:
    """A ball of ``radius`` mm carrying ``RangedShares(mode=mode)``, with the range ``ranged`` (mm) declared."""
    experiment = free_space(f"planted_{mode.replace('-', '_')}", {"ball": Sphere(radius=mm(radius))},
                            RangedShares(mode=mode), mass=kg(2e-8), inertia=kg_m2((2e-17, 2e-17, 2e-17)))
    if ranged is not None:
        experiment.declare_range(RADIUS, mm(ranged[0]), mm(ranged[1]), reason=REASON)
    return experiment


__all__ = ["HIGH", "LOW", "PULL", "RADIUS", "REASON", "SPACING", "moving", "planted", "pulled"]
