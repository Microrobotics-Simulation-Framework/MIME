"""The FVM on a general wall: the vessels its gates run on and the oracles that judge them (F2a's FA2 to FA8). Units: the parent's diameter ``D = 1``, ``nu = rho = 1``, a body force ``G`` along
z on the fluid; every vessel periodic along z, steady Stokes, through the node's own step.

**Vessels.** Tube trees (``mime.scene.TubeTree``) held in a periodic box along z (``periodic_tree_layout``): the
first step's split-and-merge loop (daughters 0.4 and 0.3 D, or both 0.4: the symmetric split), a 75 % stenosis, a
full occlusion and a biconical taper, each turned about z at random (the loop's daughters then off every grid plane)
and moved by a sub-cell offset. Two walls no tree gives exactly, each a solid of the node's wall seam (its
``solids``), the node otherwise a straight tube's (its reports along z): a helical tube (an exact distance by Newton on
its centreline's angle) and an elliptic tube (an exact distance to the ellipse by bisection).

**Oracles.** Dean's first-order profile of Stokes flow in a curved pipe, ``w = (G / 4 nu)(a^2 - r^2)(1 - (3/4) kappa
x)``, ``x`` measured away from the centre of curvature (derived here from the axial momentum equation in toroidal
coordinates at ``O(kappa a)``; a helix with small torsion is the same at that order); the ellipse's Poiseuille
``w = A (1 - x^2/a^2 - y^2/b^2)``; lubrication along a slowly varying radius, ``Q = G L / (8 nu / pi int R^-4 dz)``;
Richardson references at 32 and 64 cells a diameter, stored (``data/general_wall_references.json``)."""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from fvm_cases import Waveform, driven, in_dtype, run
from scipy.optimize import least_squares

from mime.nodes.flow.fvm import ImplicitFlowNode, MACGrid
from mime.nodes.flow.fvm._box import periodic_tree_layout

#: The drive on the fluid, per unit mass (nu = 1).
G = 1.0
#: The data the references are stored in.
REFERENCES = Path(__file__).parent / "data" / "general_wall_references.json"


def draw_seed(fixed: int) -> int:
    """A draw's seed: ``DETECTION_SEED`` where the detection-rate harness sets it, else ``fixed``."""
    held = os.environ.get("DETECTION_SEED")
    return int(held) if held else fixed


def seeds(*fixed: int) -> tuple[int, ...]:
    """A gate's draws: ``DETECTION_SEED`` alone where the detection-rate harness sets it, else ``fixed``."""
    held = os.environ.get("DETECTION_SEED")
    return (int(held),) if held else fixed


def plant(name: str) -> str | None:
    """The plant a control switches on, ``MIME_PLANT_WALL`` (``None`` where it is unset)."""
    held = os.environ.get("MIME_PLANT_WALL")
    return held if held == name or (held is not None and held.startswith(name + ":")) else None


def plant_value(name: str, default: float) -> float:
    held = plant(name)
    if held is None or ":" not in held:
        return default
    return float(held.split(":", 1)[1])


def turned_about_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def draw(seed: int) -> tuple[float, tuple[float, float, float]]:
    """A turn about z, 1 to 89 degrees off the grid's planes in each quarter, and a sub-cell offset (cells)."""
    rng = np.random.default_rng(seed)
    angle = math.radians(rng.uniform(1.0, 89.0)) + rng.integers(4) * math.pi / 2
    offset = rng.uniform(0.0, 1.0, 3)
    return angle, (float(offset[0]), float(offset[1]), float(offset[2]))


# -- tube trees --------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Tree:
    """A tube tree periodic along z with period ``length``, in its own frame (D = 1): ``points``, ``segments`` and their
    ``radii``, and where its reading planes are."""

    points: tuple[tuple[float, float, float], ...]
    segments: tuple[tuple[int, int], ...]
    radii: tuple[tuple[float, float], ...]
    length: float
    middle: float = 0.0

    def turned(self, angle: float) -> Tree:
        turn = turned_about_z(angle)
        moved = np.asarray(self.points) @ turn.T
        return dataclasses.replace(self, points=tuple(tuple(float(c) for c in p) for p in moved))

    def grown(self, segments: tuple[int, ...], by: float) -> Tree:
        """Each of ``segments`` with its radii grown by ``by`` (a wall plant on them)."""
        radii = tuple((a + by, b + by) if k in segments else (a, b) for k, (a, b) in enumerate(self.radii))
        return dataclasses.replace(self, radii=radii)


def loop(daughters: tuple[float, float] = (0.4, 0.3), angle: float = math.radians(30.0)) -> Tree:
    """The first step's split-and-merge loop: a parent of D splits at J1 into daughters of ``daughters`` at +-``angle``,
    straight for 0.8 D, which merge again at J2; the parent's two halves reach a diameter past the period's ends.
    Segments 2 to 4 are the first daughter's (+x), 5 to 7 the second's."""
    slant, straight, lead = 1.0, 0.8, 1.0
    across = slant * math.tan(angle)
    z1, z2 = lead, lead + 2 * slant + straight
    length = z2 + lead
    points = ((0, 0, -lead), (0, 0, z1), (across, 0, z1 + slant), (across, 0, z2 - slant), (0, 0, z2),
              (0, 0, length + lead), (-across, 0, z1 + slant), (-across, 0, z2 - slant))
    segments = ((0, 1), (4, 5), (1, 2), (2, 3), (3, 4), (1, 6), (6, 7), (7, 4))
    a, b = daughters
    radii = ((0.5, 0.5), (0.5, 0.5), (a, a), (a, a), (a, a), (b, b), (b, b), (b, b))
    return Tree(tuple(tuple(float(c) for c in p) for p in points), segments, radii, length, 0.5 * (z1 + z2))


def stenosis(throat: float = 0.25, cone: float = 1.0, lead: float = 1.5) -> Tree:
    """A 75 % stenosis (``throat`` 0.25 against the parent's 0.5): the parent narrows by a cone ``cone`` long to the
    throat and widens by another, the parent reaching a diameter past the period's ends."""
    z1, z2, z3 = lead, lead + cone, lead + 2 * cone
    length = z3 + lead
    points = ((0, 0, -1.0), (0, 0, z1), (0, 0, z2), (0, 0, z3), (0, 0, length + 1.0))
    segments = ((0, 1), (1, 2), (2, 3), (3, 4))
    radii = ((0.5, 0.5), (0.5, throat), (throat, 0.5), (0.5, 0.5))
    return Tree(tuple(tuple(float(c) for c in p) for p in points), segments, radii, length, z2)


def occlusion(plug: float = 1.0, lead: float = 1.5) -> Tree:
    """A full occlusion: the tube closed by ``plug`` diameters of solid between its two ends' balls."""
    z1 = lead
    z2 = z1 + 1.0 + plug
    length = z2 + lead
    points = ((0, 0, -1.0), (0, 0, z1), (0, 0, z2), (0, 0, length + 1.0))
    segments = ((0, 1), (2, 3))
    radii = ((0.5, 0.5), (0.5, 0.5))
    return Tree(tuple(tuple(float(c) for c in p) for p in points), segments, radii, length, 0.5 * (z1 + z2))


def taper(narrow: float = 0.3, half: float = 3.0) -> Tree:
    """A biconical taper: the radius falls linearly from 0.5 to ``narrow`` over ``half`` and rises back, the period
    ``2 half``, the wide ends reaching past it."""
    slope = (0.5 - narrow) / half
    length = 2 * half
    points = ((0, 0, -1.0), (0, 0, 0.0), (0, 0, half), (0, 0, length), (0, 0, length + 1.0))
    segments = ((0, 1), (1, 2), (2, 3), (3, 4))
    radii = ((0.5 + slope, 0.5), (0.5, narrow), (narrow, 0.5), (0.5, 0.5 - slope))
    return Tree(tuple(tuple(float(c) for c in p) for p in points), segments, radii, length, half)


@dataclasses.dataclass(frozen=True)
class Placed:
    """A tree placed in its box: the node, the layout, the tree in its own frame, and its turn about z."""

    node: ImplicitFlowNode
    layout: Any
    tree: Tree
    angle: float


def tree_node(tree: Tree, dh: int, *, angle: float = 0.0, offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
              fillet: float = 0.1, interior: str = "0.1h", scheme: str = "steady", dt: float = 0.5,
              tolerance: float | None = None, solver: Tree | None = None, margin_cells: int = 4) -> Placed:
    """The node on ``tree`` turned by ``angle`` about z, ``dh`` cells a diameter of the parent, its axis offset by
    ``offset`` cells; ``solver`` is the tree the solver is given where a control plants a wall the oracle does not
    know of (the same layout)."""
    turned = tree.turned(angle)
    layout = periodic_tree_layout(turned.points, turned.segments, turned.radii, turned.length, spacing=1.0 / dh,
                                  margin_cells=margin_cells, offset_cells=offset)
    given = turned if solver is None else solver.turned(angle)
    node = ImplicitFlowNode("flow", dt, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                            spacing=layout.spacing, viscosity=1.0, density=1.0, tree_points=np.asarray(layout.points),
                            tree_radii=np.asarray(given.radii), tree_fillet=fillet, tree_segments=turned.segments,
                            convection=False, scheme=scheme, interior=interior, tolerance=tolerance)
    return Placed(driven(node, Waveform(steady=G)), layout, tree, angle)


def solved(placed: Placed, steps: int = 1) -> dict[str, Any]:
    return run(placed.node, steps)


def mid_plane_split(placed: Placed, state: dict[str, Any]) -> tuple[float, float]:
    """The two daughters' axial fluxes through the faces of the plane normal to z nearest the loop's middle, each
    the faces on its side of the plane through the axis normal to the loop's (turned) plane: the first daughter's,
    the second's."""
    layout = placed.layout
    h = layout.spacing
    grid = MACGrid(layout.cells, h)
    k = int(round((placed.tree.middle - layout.origin[2]) / h))
    points = np.asarray(grid.face_points(2, "float64"))[:, :, k] + np.asarray(layout.origin)
    across = np.asarray([math.cos(placed.angle), math.sin(placed.angle)])
    side = points[..., 0] * across[0] + points[..., 1] * across[1]
    w = np.asarray(state["velocity_z"], np.float64)[:, :, k]
    first = float(np.sum(np.where(side > 0.0, w, 0.0))) * h * h
    second = float(np.sum(np.where(side < 0.0, w, 0.0))) * h * h
    return first, second


def box_flux(state: dict[str, Any]) -> float:
    """The box's whole axial flux through its period, ``lumen_flux`` plus the solid's (the bypass) as the node reads
    them."""
    lumen = float(state["lumen_flux"])
    return lumen * (1.0 + float(state["bypass_ratio"]))


def lubrication_sensitivity(tree: Tree, points: int = 20001) -> tuple[float, float]:
    """Along a tree whose radius varies only along z (FA3, FA5): ``int R^-4 dz`` over the period and ``d ln Q / d
    delta = 4 int R^-5 / int R^-4``, a uniform wall shift ``delta``'s effect on the lubrication flux."""
    z = np.linspace(0.0, tree.length, points)
    radius = radius_along(tree, z)
    inverse = np.trapezoid(radius ** -4.0, z)
    return float(inverse), float(4.0 * np.trapezoid(radius ** -5.0, z) / inverse)


def radius_along(tree: Tree, z: np.ndarray) -> np.ndarray:
    """The radius of an axisymmetric tree (its points on the axis) at ``z``, each segment's linear between its ends."""
    radius = np.full_like(z, np.nan)
    for (a, b), (ra, rb) in zip(tree.segments, tree.radii, strict=True):
        za, zb = tree.points[a][2], tree.points[b][2]
        inside = (z >= min(za, zb)) & (z <= max(za, zb))
        radius = np.where(inside & np.isnan(radius), ra + (rb - ra) * (z - za) / (zb - za), radius)
    return radius


def lubrication_flux(tree: Tree, shift: float = 0.0) -> float:
    """Lubrication's flux through the tree with every radius moved by ``shift`` (m): ``G L / (8 int R^-4 dz / pi)``."""
    z = np.linspace(0.0, tree.length, 20001)
    radius = radius_along(tree, z) + shift
    return float(G * tree.length / (8.0 * np.trapezoid(radius ** -4.0, z) / math.pi))


def references() -> dict[str, Any]:
    return json.loads(REFERENCES.read_text(encoding="utf-8")) if REFERENCES.exists() else {}


# -- the helical tube (FA2) ----------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Helix:
    """A tube of radius ``a`` about a helix of radius ``radius`` and ``c`` (its pitch over two pi) about the box's
    z-axis through ``(x0, y0)``, its phase turned by ``phase``: the centreline ``(x0 + R cos t, y0 + R sin t, c (t -
    phase))``. Its curvature ``R / (R^2 + c^2)``, its torsion ``c / (R^2 + c^2)``."""

    x0: float
    y0: float
    radius: float
    c: float
    a: float
    phase: float = 0.0
    region: str = "wall"

    @property
    def curvature(self) -> float:
        return self.radius / (self.radius ** 2 + self.c ** 2)

    def nearest(self, points: Any) -> Any:
        """The centreline's angle nearest each point: from the angle about the axis unwrapped to the point's turn,
        eight Newton steps on ``(C(t) - p) . C'(t) = 0`` (one nearest point within the tube's reach: the turns lie
        ``2 pi c`` apart along z, more than the tube's diameter and the band's reach)."""
        x = points[..., 0] - self.x0
        y = points[..., 1] - self.y0
        z = points[..., 2]
        around = jnp.arctan2(y, x)
        turn = jnp.round((z / self.c + self.phase - around) / (2.0 * jnp.pi))
        t = around + 2.0 * jnp.pi * turn
        for _ in range(8):
            ct, st = jnp.cos(t), jnp.sin(t)
            dx, dy, dz = self.radius * ct - x, self.radius * st - y, self.c * (t - self.phase) - z
            f = dx * (-self.radius * st) + dy * (self.radius * ct) + dz * self.c
            df = self.radius ** 2 + self.c ** 2 + dx * (-self.radius * ct) + dy * (-self.radius * st)
            t = t - f / df
        return t

    def frame(self, t: Any) -> tuple[Any, Any, Any, Any]:
        """The centreline's point, tangent, principal normal (towards the axis) and binormal at ``t``."""
        ct, st = jnp.cos(t), jnp.sin(t)
        w = math.hypot(self.radius, self.c)
        point = jnp.stack([self.x0 + self.radius * ct, self.y0 + self.radius * st, self.c * (t - self.phase)], -1)
        tangent = jnp.stack([-self.radius * st, self.radius * ct, jnp.full_like(t, self.c)], -1) / w
        normal = jnp.stack([-ct, -st, jnp.zeros_like(t)], -1)
        return point, tangent, normal, jnp.cross(tangent, normal)

    def signed_distance(self, points: Any) -> Any:
        point = self.frame(self.nearest(points))[0]
        return jnp.sqrt(jnp.sum((points - point) ** 2, axis=-1)) - self.a


@dataclasses.dataclass(frozen=True)
class EllipticTube:
    """A tube along z of elliptic section, semi-axes ``a`` along its own x and ``b`` along its own y, turned by
    ``angle`` about z, its axis through ``(x0, y0)``."""

    x0: float
    y0: float
    a: float
    b: float
    angle: float
    region: str = "wall"

    def own(self, points: Any) -> tuple[Any, Any]:
        x, y = points[..., 0] - self.x0, points[..., 1] - self.y0
        c, s = math.cos(self.angle), math.sin(self.angle)
        return c * x + s * y, -s * x + c * y

    def signed_distance(self, points: Any) -> Any:
        """The exact distance to the ellipse (Eberly's: the root ``t`` of ``(a u/(t + a^2))^2 + (b v/(t + b^2))^2 = 1``
        above ``-min(a^2, b^2)``, ``u, v`` the point's in the first quadrant, by 100 bisections), its sign the inside
        test's."""
        u, v = self.own(points)
        tiny = jnp.finfo(u.dtype).tiny
        u, v = jnp.abs(u), jnp.maximum(jnp.abs(v), tiny)
        a2, b2 = self.a ** 2, self.b ** 2
        lower = jnp.full_like(u, -min(a2, b2))
        upper = jnp.sqrt(a2 * u * u + b2 * v * v) + max(a2, b2)

        def f(t: Any) -> Any:
            return (self.a * u / (t + a2)) ** 2 + (self.b * v / (t + b2)) ** 2 - 1.0

        for _ in range(100):
            middle = 0.5 * (lower + upper)
            positive = f(middle) > 0.0
            lower, upper = jnp.where(positive, middle, lower), jnp.where(positive, upper, middle)
        t = 0.5 * (lower + upper)
        qu, qv = a2 * u / jnp.maximum(t + a2, tiny), b2 * v / jnp.maximum(t + b2, tiny)
        distance = jnp.sqrt((u - qu) ** 2 + (v - qv) ** 2)
        inside = (u / self.a) ** 2 + (v / self.b) ** 2 < 1.0
        return jnp.where(inside, -distance, distance)


class SolidFlowNode(ImplicitFlowNode):
    """The flow node with a wall its tube parameters do not describe (a test's solid through the wall seam): the
    reports read along z, as a straight tube's along z."""

    held_solid: Any = None

    def solids(self, p: Any, grid: MACGrid) -> list[Any]:
        return [self.held_solid]


def solid_node(solid: Any, cells: tuple[int, int, int], h: float, radius: float, *, scheme: str = "steady",
               interior: str = "0.1h", tolerance: float | None = None) -> SolidFlowNode:
    centre = (cells[0] * h / 2, cells[1] * h / 2, 0.0)
    node = SolidFlowNode("flow", 0.5, cells_x=cells[0], cells_y=cells[1], cells_z=cells[2], spacing=h, viscosity=1.0,
                         density=1.0, tube_centre=centre, tube_axis=(0.0, 0.0, 1.0), tube_radius=radius,
                         convection=False, scheme=scheme, interior=interior, tolerance=tolerance)
    node.held_solid = solid
    return driven(node, Waveform(steady=G))


def cell_velocity(state: dict[str, Any]) -> np.ndarray:
    """The velocity at the cells' centres, each component averaged from its two faces, ``(*cells, 3)``."""
    u = [np.asarray(state[f"velocity_{c}"], np.float64) for c in "xyz"]
    return np.stack([0.5 * (u[k] + np.roll(u[k], -1, k)) for k in range(3)], -1)


def dean_fit(helix: Helix, state: dict[str, Any], cells: tuple[int, int, int], h: float, clean: float
             ) -> dict[str, float]:
    """Dean's first-order Stokes profile fitted to the axial velocity at every cell centre deeper than ``clean`` (m)
    in the tube, each mapped to the canonical section by its nearest centreline point (the flow is invariant under the
    helix's screw motion): ``w = A (a^2 - r^2)(1 + (3/4) kappa n)``, ``r`` from a free centre ``(n0, b0)`` in the
    normal plane, ``n`` its coordinate along the principal normal (towards the axis), ``A`` and ``a`` free. Returns
    the free centre and radius (m), and the amplitude against ``G / (4 nu)``'s."""
    grid = MACGrid(cells, h)
    points = np.asarray(grid.cell_points("float64"), np.float64).reshape(-1, 3)
    held = jnp.asarray(points)
    t = helix.nearest(held)
    centre, tangent, normal, binormal = (np.asarray(v, np.float64) for v in helix.frame(t))
    offset = points - centre
    n, b = np.sum(offset * normal, -1), np.sum(offset * binormal, -1)
    depth = np.hypot(n, b)
    inside = depth < helix.a - clean
    w = np.sum(cell_velocity(state).reshape(-1, 3) * tangent, -1)[inside]
    n, b = n[inside], b[inside]
    kappa = helix.curvature

    def residual(x: np.ndarray) -> np.ndarray:
        amplitude, radius, n0, b0 = x
        r2 = (n - n0) ** 2 + (b - b0) ** 2
        return amplitude * (radius ** 2 - r2) * (1.0 + 0.75 * kappa * (n - n0)) - w

    fitted = least_squares(residual, np.asarray([G / 4.0, helix.a, 0.0, 0.0]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    amplitude, radius, n0, b0 = fitted.x
    return {"radius": float(radius), "normal": float(n0), "binormal": float(b0),
            "amplitude": float(amplitude / (G * helix.c / math.hypot(helix.radius, helix.c) / 4.0)),
            "points": float(inside.sum())}


def ellipse_fit(tube: EllipticTube, state: dict[str, Any], cells: tuple[int, int, int], h: float, clean: float
                ) -> tuple[float, float]:
    """The quadratic ``w = c0 + c1 u + c2 v + c3 u^2 + c4 u v + c5 v^2`` fitted to the axial velocity at the cell
    centres deeper than ``clean`` (m), in the ellipse's own frame: its zero level set's semi-axes (m), the conic's
    principal axes."""
    grid = MACGrid(cells, h)
    points = np.asarray(grid.cell_points("float64"), np.float64).reshape(-1, 3)
    depth = -np.asarray(tube.signed_distance(jnp.asarray(points)), np.float64)
    inside = depth > clean
    u, v = (np.asarray(c, np.float64)[inside] for c in tube.own(jnp.asarray(points)))
    w = cell_velocity(state).reshape(-1, 3)[inside, 2]
    design = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1)
    c0, c1, c2, c3, c4, c5 = np.linalg.lstsq(design, w, rcond=None)[0]
    quadratic = np.asarray([[c3, 0.5 * c4], [0.5 * c4, c5]])
    centre = np.linalg.solve(2.0 * quadratic, -np.asarray([c1, c2]))
    peak = c0 + 0.5 * float(np.dot([c1, c2], centre))
    values, vectors = np.linalg.eigh(quadratic)
    axes = np.sqrt(-peak / values)
    along_u = int(np.argmax(np.abs(vectors[0])))
    return float(axes[along_u]), float(axes[1 - along_u])


__all__ = ["G", "EllipticTube", "Helix", "Placed", "SolidFlowNode", "Tree", "box_flux", "cell_velocity", "dean_fit",
           "draw", "draw_seed", "ellipse_fit", "in_dtype", "loop", "lubrication_flux", "lubrication_sensitivity",
           "mid_plane_split", "occlusion", "plant", "plant_value", "radius_along", "references",
           "seeds", "solid_node", "solved", "stenosis", "taper", "tree_node", "turned_about_z"]
