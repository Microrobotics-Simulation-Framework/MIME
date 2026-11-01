"""The experiments the wall-kind interface is tested in with a confining wall: a sphere in a straight tube with
``Drag.MRS``, the tube's correction read (``mime.nodes.hydro.mrs.tube_correction``) from a small table this module
builds from the Fourier-Bessel series (``tube_fourier_bessel.image``), never from the network.

The test registers ``oracle.tube_series``, a wall kind that matches the tube's lumen with the tube's correction
(``tube_wall``: its function, its constants, its preparation) on that table. Its table covers the knots of the
2026-05-21 tables nearest the axis, every angle and the smallest axial offsets, in the tube's radius: a sphere of a
tenth of the tube's radius on the axis, sampled at up to eleven Gauss nodes in its polar angle, stays inside it. It
declares no switches: the tube's wall kind, once it is built, declares its read's.

``builder_table`` builds a small table of the unit tube from its knots with the table's own builder
(``scripts/tube_table/build.py``), at the series' truncation (``tube_fourier_bessel.image``'s, at its default counts), in
the form the full table takes: its radii from the axis."""

import functools
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import tube_fourier_bessel as fb

from mime.core.tables import Table
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.interactions import Drag
from mime.nodes.hydro.mrs import register_wall_correction, tube_wall, unit_tube_table, wall_corrections
from mime.registry import Geometry, WallForm, WallKind, register_wall_kind
from mime.scene import KEPT, Scene, Tube, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import Quantity, kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s
from mime.vocabulary import register_namespace

#: The table's knots in the tube's radius: the 2026-05-21 tables' first nine radii (forty, clustered at both ends),
#: the angle on ``[0, pi]`` and the axial offset to a quarter of the radius.
RADII = 0.02 + 0.96 * 0.5 * (1 + np.tanh(1.5 * (2 * np.linspace(0, 1, 40) - 1)) / math.tanh(1.5))[:9]
ANGLES, OFFSETS = np.linspace(0.0, math.pi, 17), np.linspace(0.0, 0.25, 5)
#: The tube the cases use: its radius, and the water's viscosity.
TUBE_RADIUS, WATER = mm(5.0), mPa_s(1.0)


@functools.cache
def series_table():
    """The unit tube's table on these knots, from the series at the 2026-05-21 tables' truncation."""
    angle, offset = (grid.ravel() for grid in np.meshgrid(ANGLES, OFFSETS, indexing="ij"))
    targets = np.concatenate([np.stack([r * np.cos(angle), r * np.sin(angle), offset], axis=1) for r in RADII])
    sources = np.stack([RADII, np.zeros_like(RADII), np.zeros_like(RADII)], axis=1)
    image = fb.image(targets, sources, 1.0, 1.0, modes=15, wavenumbers=80, angles=64)
    turned = np.einsum("pab,psbc->psac", fb.to_cylindrical(np.tile(angle, len(RADII))), image)
    data = turned.reshape(len(RADII), len(ANGLES), len(OFFSETS), len(RADII), 3, 3).transpose(0, 3, 1, 2, 4, 5)
    return unit_tube_table(1.0, RADII, ANGLES, OFFSETS, data)


@functools.cache
def builder() -> ModuleType:
    """``scripts/tube_table/build.py``, the unit tube table's builder, loaded from the repository: scripts are no
    package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "tube_table" / "build.py"
    spec = importlib.util.spec_from_file_location("tube_table_build", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def builder_data(radii: tuple[float, ...], angles: tuple[float, ...], offsets: tuple[float, ...]) -> np.ndarray:
    """The unit tube's image on these knots, ``(rho_t, rho_s, angle, offset, 3, 3)``, by the table's builder at the
    series' truncation (15 modes, 80 wavenumbers, 64 points round the wall; the axial reach of sources to 0.9)."""
    build = builder()
    truncation = build.series_truncation(15, 80, 64, 0.9)
    return np.stack([build.image_grid(r, radii, angles, offsets, truncation, threads=1) for r in radii], axis=1)


def builder_table(radii: tuple[float, ...], angles: tuple[float, ...], offsets: tuple[float, ...]) -> Table:
    """The unit tube's table on these knots, from ``builder_data``."""
    return unit_tube_table(1.0, np.asarray(radii), np.asarray(angles), np.asarray(offsets),
                           builder_data(radii, angles, offsets))


register_namespace("oracle", "0.0.1")


@register_wall_kind
@dataclass(frozen=True, kw_only=True)
class TubeSeries(WallKind):
    """The tube's image system on a small table of the series."""

    name = "oracle.tube_series"
    version = "0.0.1"
    form = WallForm.ANALYTIC
    matches = "cylinder"
    reads = ("centre", "axis", "radius")
    differentiability = GradientStatus.EXACT
    builds = {"": ()}


if f"{TubeSeries.name} {TubeSeries.version}" not in wall_corrections():
    register_wall_correction(tube_wall(TubeSeries.name, TubeSeries.version, series_table))


def grid_sphere(radius: Quantity) -> Geometry:
    """A sphere sampled by its Gauss-Legendre grid, the version kept from before quasi-uniform sampling (the sphere
    0.1.0): Gauss's first polar node keeps every sample more than a fifth of the sphere's radius from its polar axis,
    which the tube oracles' small series table, whose first radius is off the tube's axis, needs of a sphere centred
    on it; the current version's quasi-uniform sampling takes samples at the poles."""
    return KEPT[("sphere", "0.1.0")][0](radius=radius)


def sphere_in_a_tube(name: str, *, radius: float, spacing: Quantity, wall_kind: str = TubeSeries.name,
                     blob: float = 0.5, tube_radius: Quantity = TUBE_RADIUS, hull: Geometry | None = None) -> Experiment:
    """A sphere of ``radius`` (of the tube's) at rest on the axis of a straight tube along z, in water filling its
    lumen, unturned (its sampling's poles along the tube's axis), with ``Drag.MRS`` and nothing else; or ``hull``
    there in its place. The sphere is sampled by its grid (:func:`grid_sphere`), as the series table's oracles need
    (labelled decision D23 of phase 7m-a); the shipped table reads on the axis, and its oracle passes the current
    sphere as ``hull``."""
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=tube_radius,
                                                             wall=mm(0.5), length=mm(200.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen"))
    experiment.declare("water.viscosity", WATER, reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(),
                             parts={"shell": Part(hull if hull is not None else
                                                  grid_sphere(m(radius * float(tube_radius.value))))},
                             initial=State(position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-5), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    experiment.attach(Drag.MRS(spacing=spacing, blob=blob, wall_kind=wall_kind), body="ball", medium="water")
    return experiment


def polar_spacing(radius: float, polar: int, tube_radius: Quantity = TUBE_RADIUS) -> Quantity:
    """The spacing that samples a sphere of ``radius`` (of the tube's) at ``polar`` Gauss nodes in its polar angle
    and twice as many round it."""
    return m(math.pi * radius * float(tube_radius.value) / polar * (1 + 1e-6))


def haberman_sayre(ratio: float) -> float:
    """Haberman & Sayre's (1958) wall correction factor for a sphere translating along the axis of a circular tube,
    ``F = 6 pi mu a U K``, ``ratio`` the sphere's radius over the tube's, in the form commonly quoted from them (the
    paper itself not opened here): ``K = (1 - 0.75857 l^5) / (1 - 2.1050 l + 2.0865 l^3 - 1.7068 l^5 + 0.72603 l^6)``.
    Its first term, ``2.1050``, is Bohlin's and Faxen's first wall coefficient, ``2.10444``, rounded."""
    return (1 - 0.75857 * ratio ** 5) / (1 - 2.1050 * ratio + 2.0865 * ratio ** 3 - 1.7068 * ratio ** 5
                                          + 0.72603 * ratio ** 6)
