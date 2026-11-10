"""The experiments graded sampling is tested in, and its wrong-way control: a sphere held a gap above a
plane wall with ``Drag.MRS(contact=..., contact_gap=...)`` or a cylinder tilted so its rim is nearest it with
``grading="features"``, and the density that grades the wrong way (coarser where the grading is finer)."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

import mime.interactions.hydrodynamic as hydrodynamic
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.interactions import Drag
from mime.scene import Cylinder, Plane, Scene, Sphere, Z
from mime.scene import _sampling
from mime.scene._grading import GradedDensity
from mime.things import Body, Medium, Part, State, Structure
from mime.units import kg, kg_m2, m, m_per_s, m_per_s2, mPa_s, ms, rad_per_s

#: The bodies' size (m) and the medium's viscosity (Pa s); the force spacing, a third of a radius less a tenth.
RADIUS, VISCOSITY, SPACING = 1e-3, 1e-3, 0.3e-3
#: The rim case's cylinder: its length (m) and its tilt about y (radians), which puts one rim point nearest the plane.
LENGTH, TILT = 2e-3, math.radians(45.0)
TILTED = (math.cos(TILT / 2), 0.0, math.sin(TILT / 2), 0.0)


def rim_drop() -> float:
    """How far below its centre the tilted cylinder's lowest rim point lies (m)."""
    axis_z = math.cos(TILT)
    return RADIUS * math.sqrt(1.0 - axis_z ** 2) + LENGTH / 2 * abs(axis_z)


def above_plane(kind: Any, centre: tuple[float, float, float], orientation: tuple[float, ...], *, blob: float,
                name: str = "graded", **drag: Any) -> Experiment:
    """``kind`` the hull of a body ``ball`` at ``centre`` and ``orientation`` above the plane z = 0, the medium filling
    the half-space above it, with ``Drag.MRS`` in the plane wall kind at :data:`SPACING` and ``drag``'s settings."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    floor = experiment.add_structure(Structure("floor", geometry=Plane(normal=(0.0, 0.0, 1.0), offset=m(0.0))))
    experiment.add_medium(Medium("water", fills=floor.half_space))
    experiment.declare("water.viscosity", mPa_s(VISCOSITY * 1e3), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(kind)},
                             initial=State(position=m(centre), orientation=orientation,
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-5), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    experiment.attach(Drag.MRS(spacing=m(SPACING), blob=blob, wall_kind="plane", **drag), body="ball", medium="water")
    return experiment


def sphere(gap: float, *, blob: float, graded: bool, **drag: Any) -> Experiment:
    """The sphere with its surface ``gap`` force spacings above the plane, its contact graded for half a spacing where
    ``graded``."""
    settings = {"contact": (0.0, 0.0, -1.0), "contact_gap": m(0.5 * SPACING)} if graded else {}
    return above_plane(Sphere(radius=m(RADIUS)), (0.0, 0.0, RADIUS + gap * SPACING), (1.0, 0.0, 0.0, 0.0), blob=blob,
                       **settings, **drag)


def cylinder(gap: float, *, blob: float, graded: bool, **drag: Any) -> Experiment:
    """The tilted cylinder with its lowest rim point ``gap`` force spacings above the plane, its features graded where
    ``graded``."""
    kind = Cylinder(radius=m(RADIUS), length=m(LENGTH), axis=(0.0, 0.0, 1.0))
    return above_plane(kind, (0.0, 0.0, gap * SPACING + rim_drop()), TILTED, blob=blob,
                       **({"grading": "features"} if graded else {}), **drag)


@dataclass(frozen=True)
class WrongWay:
    """The wrong-way control's density: the grading's inverted, coarser where it is finer (at most twice the force
    spacing), scaled to an area mean of one, so that it takes the uniform sampling's count."""

    base: GradedDensity
    mean: float

    @property
    def features(self) -> bool:
        return self.base.features

    @property
    def contact(self) -> Any:
        return self.base.contact

    @property
    def spacing(self) -> float:
        return self.base.spacing

    def raw(self, points: np.ndarray) -> np.ndarray:
        return np.clip(1.0 / self.base(points), 0.25, 1.0)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.raw(np.asarray(points, dtype=np.float64)) / self.mean

    def spacing_at(self, points: np.ndarray) -> np.ndarray:
        return self.base.spacing / np.sqrt(self(points))


@contextmanager
def wrong_way() -> Iterator[None]:
    """Every graded drag built inside it grades the wrong way (:class:`WrongWay`)."""
    held = hydrodynamic.Drag._density

    def inverted(self: Any, settings: Any, hull: Any, spacing: float) -> Any:
        base = held(self, settings, hull, spacing)
        if base is None:
            return None
        patches = tuple(hull.kind.patches)
        points, weights, _ = _sampling.quadrature_arrays(
            hull.kind, hull.settings, patches,
            _sampling.quadrature_counts(hull.kind, hull.settings, spacing / _sampling.REFINEMENT, patches))
        return WrongWay(base, float(np.sum(weights * WrongWay(base, 1.0).raw(points)) / np.sum(weights)))

    hydrodynamic.Drag._density = inverted  # type: ignore[method-assign]
    try:
        yield
    finally:
        hydrodynamic.Drag._density = held  # type: ignore[method-assign]
