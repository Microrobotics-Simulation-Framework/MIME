"""G23, the mesh swap: a kind's drawing traced as a mesh images as its closed form does, through the real node (two
builds, the static structure's transmission compared pixel by pixel), within lines derived from the drawing's sagitta
and the sub-rays a pixel.

**The cases.** The tube's drawing (64 segments; ``Tube.drawing``, placed as the stage places it) referenced as a
scenery mesh that attenuates, against the closed-form tube (route 1); and an ellipsoid structure of equal semi-axes, its
drawing traced (the drawn route, at the stage's facets), against a sphere structure (route 1, exact disc coverage),
opaque and attenuating. The camera is the in-loop camera's 160 x 120, 1 mm from the axis.

**Two lines a pixel** (``S`` the sub-rays a pixel each way):

- **against the true surface on the same sub-rays (the tessellation's claim):** the drawing's vertices lie on the surface
  and it is convex where the surface is (a polygon inscribed in each circle, a polyhedron inscribed in the sphere), so
  each sub-ray's chord through it lies between the chords through the surfaces at the inscribed radius ``r cos(pi/n)``
  (the polygon's apothem; for the sphere the least distance from the centre to the drawing's triangles) and at ``r``.
  The traced pixel lies between the means over its sub-rays of the factors at those bounds, plus rounding. The bracket
  is a few thousandths wide at a silhouette and narrower inside: a mesh half a pixel off falls out of it.
- **against route 1 (the design's line):** the closed form's pixel is the box filter's coverage and the centre ray's
  chord; the traced pixel the mean of ``S x S`` sub-rays. They agree within ``(1 - tau_min) (1 / (2S) + sqrt(2) s_px)``
  at a silhouette (the sub-rays' coverage of a straight edge, and the sagitta ``s_px`` in pixels across a pixel's
  diagonal) plus the spread of the true factor over the pixel's crossing rays (the centre ray's chord against the mean of
  the sub-rays'), plus rounding. The run at S = 2, 4 and 8 measures each S's worst ratio to its line, which sets
  ``MESH_SAMPLES`` (the predictions' outcomes).

**Control** (``CAMERA_PLANT=shifted-mesh``): the traced mesh moved half a pixel's footprint across the line of sight
(0.67 um at 1 mm); the first line catches it at the silhouettes on every run (a code patch, no draw)."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest
from in_loop_cases import DETECTOR, LOOK_ACROSS, ballistic, planted
from mesh_cases import VESSEL, with_vessel

from mime import Body, Experiment, State, Structure
from mime.build import kind_prim
from mime.dynamics import RigidInertial
from mime.observations import Camera, World
from mime.observations import _camera as camera_module
from mime.scene import Ellipsoid, Scene, Sphere, Z
from mime.sensing import _scene as scene
from mime.sensing._node import STRUCTURE, unpacked
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, ms, rad_per_s, um

#: Half a pixel's footprint at the tube's axis, 1 mm from the camera at fx = 750 px.
HALF_PIXEL = 0.5e-3 / DETECTOR.fx
#: The tube's inner radius, wall and length (in_loop_cases' tube).
TUBE = (60e-6, 10e-6, 480e-6)
SPHERE = 40e-6
MU = 3000.0

if planted("shifted-mesh"):
    import dataclasses

    original_scenery, original_drawn = scene.scenery_mesh, scene.drawn_mesh
    moved = np.array([0.0, 0.0, HALF_PIXEL])
    scene.scenery_mesh = lambda *a, **k: dataclasses.replace(original_scenery(*a, **k),  # type: ignore[assignment]
                                                             triangles=original_scenery(*a, **k).triangles + moved)
    scene.drawn_mesh = lambda *a, **k: dataclasses.replace(original_drawn(*a, **k),  # type: ignore[assignment]
                                                           triangles=original_drawn(*a, **k).triangles + moved)


def structure_of(exp: Experiment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The static transmission the build's node holds, its rays' directions and derivatives (float64 on the host)."""
    node = exp.build().graph.get_node("camera:")
    rays, static, _ = unpacked(node.params_pytree()[STRUCTURE], *node.image_shape(), node._maker()[1])  # noqa: SLF001
    stack = [np.stack([np.asarray(c, dtype=np.float64) for c in part], axis=-1)
             for part in (rays.direction, rays.along_u, rays.along_v)]
    return np.asarray(static, dtype=np.float64), stack[0], np.stack(stack[1:])


def sub_rays(d: np.ndarray, along: np.ndarray, samples: int) -> np.ndarray:
    """Each pixel's ``samples x samples`` sub-rays (rows, columns, sub-rays, 3), as the tracer moves the pixel's ray."""
    steps = [(i + 0.5) / samples - 0.5 for i in range(samples)]
    moved = np.stack([d + su * along[0] + sv * along[1] for sv in steps for su in steps], axis=2)
    return moved / np.linalg.norm(moved, axis=-1, keepdims=True)


ORIGIN = np.array([0.0, -1e-3, 0.0])


def tube_chord(directions: np.ndarray, radius: float) -> np.ndarray:
    """The chord of each ray from the camera through the infinite cylinder of ``radius`` along x (the view lies well
    inside the tube's length)."""
    across = directions[..., 1:]
    scale = np.linalg.norm(across, axis=-1)
    unit = across / scale[..., None]
    o = ORIGIN[1:]
    rho2 = o @ o - (unit @ -o) ** 2
    return 2.0 * np.sqrt(np.maximum(radius * radius - rho2, 0.0)) / scale


def sphere_chord(directions: np.ndarray, radius: float) -> np.ndarray:
    along = directions @ -ORIGIN
    rho2 = ORIGIN @ ORIGIN - along * along
    return 2.0 * np.sqrt(np.maximum(radius * radius - rho2, 0.0))


def tube_bounds(directions: np.ndarray, segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each ray's chord through the true wall, and the least and greatest through a wall whose outer and inner sides are
    polygons of ``segments`` inscribed in their circles."""
    inner, wall, _ = TUBE
    outer = inner + wall
    shrink = math.cos(math.pi / segments)
    true = tube_chord(directions, outer) - tube_chord(directions, inner)
    low = tube_chord(directions, outer * shrink) - tube_chord(directions, inner)
    high = tube_chord(directions, outer) - tube_chord(directions, inner * shrink)
    return true, low, high


def tube_scene(samples: int, monkeypatch: pytest.MonkeyPatch) -> tuple[Experiment, Experiment, int]:
    """The closed-form tube's scene and the drawing's: the tube's drawing at the stage's segments as a scenery mesh."""
    monkeypatch.setattr(camera_module, "MESH_SAMPLES", samples)
    closed = ballistic(hide=("ball",), absorbers=(("tube", MU),))
    plan = closed.plan().geometries["tube"]
    held = kind_prim(plan.geometry, plan.registered, plan.settings)
    assert held.drawing is not None
    points = np.asarray(held.drawing.points, dtype=np.float64)
    w, x, y, z = held.orientation
    turn = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    world = points @ turn.T + np.asarray(held.translation)
    counts, indices = held.drawing.face_vertex_counts, held.drawing.face_vertex_indices
    faces, start = [], 0
    for count in counts:
        face = indices[start:start + count]
        start += count
        faces += [(face[0], face[k], face[k + 1]) for k in range(1, count - 1)]
    traced = with_vessel(ballistic(hide=("ball", "tube"), absorbers=((VESSEL, MU),)), world, np.asarray(faces))
    return closed, traced, held.segments


def sphere_scene(kind: str, absorbers: bool) -> Experiment:
    """An ellipsoid of equal semi-axes (its drawing traced) at the origin, the camera 1 mm away."""
    assert kind == "ellipsoid"
    exp = Experiment("swap", scene=Scene.new(up=Z, units="m"), dt=ms(1))
    exp.add_structure(Structure("solid", geometry=Ellipsoid(semi_axes=um((SPHERE * 1e6,) * 3))))
    exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 300.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(5)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    exp.observe(Camera(detector=DETECTOR, mount=World(position=m((0.0, -1e-3, 0.0)), orientation=LOOK_ACROSS),
                       frame_period=ms(3), hide=("ball",), absorbers=(("solid", MU),) if absorbers else ()))
    return exp


def inscribed_radius(triangles: np.ndarray) -> float:
    """The least distance from the origin to a closed mesh's triangles (each triangle's nearest point)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    foot = np.einsum("ij,ij->i", a, normal)[:, None] * normal
    inside = np.ones(len(a), dtype=bool)
    for p, q in ((a, b), (b, c), (c, a)):
        inside &= np.einsum("ij,ij->i", np.cross(q - p, foot - p), normal) >= 0.0
    best = np.where(inside, np.abs(np.einsum("ij,ij->i", a, normal)), np.inf)
    for p, q in ((a, b), (b, c), (c, a)):
        span = q - p
        share = np.clip(np.einsum("ij,ij->i", -p, span) / np.einsum("ij,ij->i", span, span), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p + share[:, None] * span, axis=1))
    return float(best.min())


def factor(chord: np.ndarray, mu: float | None) -> np.ndarray:
    """A sub-ray's factor: ``exp(-mu L)`` where it attenuates, else zero where it crosses and one where it misses."""
    return np.exp(-mu * chord) if mu is not None else np.where(chord > 0.0, 0.0, 1.0)


def bracket(true: np.ndarray, low: np.ndarray, high: np.ndarray, mu: float | None) -> tuple[np.ndarray, np.ndarray]:
    """The least and greatest pixel a polygon between the bounds can give, over the pixel's sub-rays."""
    a, b = factor(low, mu), factor(high, mu)
    del true
    return np.minimum(a, b).mean(axis=2), np.maximum(a, b).mean(axis=2)


def dense_rays(d: np.ndarray, along: np.ndarray) -> np.ndarray:
    """Each pixel's rays on a 17 x 17 grid over it, its sides and corners included: every sub-ray of S = 2, 4 and 8
    lies on it."""
    steps = np.linspace(-0.5, 0.5, 17)
    moved = np.stack([d + su * along[0] + sv * along[1] for sv in steps for su in steps], axis=2)
    return moved / np.linalg.norm(moved, axis=-1, keepdims=True)


def design_line(true_dense: np.ndarray, samples: int, sagitta_px: float, mu: float | None) -> np.ndarray:
    """The design's line against route 1: the edge's contrast (the darkest true factor on the pixel or a neighbour, so a
    sliver between the grid's rays is not missed) times ``1/(2S) + sqrt(2) s_px``, plus the spread of the true factor
    over the pixel's crossing rays."""
    tau = factor(true_dense, mu)
    crossing = true_dense > 0.0
    lowest = np.where(crossing, tau, 1.0).min(axis=2)
    padded = np.pad(lowest, 1, constant_values=1.0)
    near = np.min([padded[1 + i:padded.shape[0] - 1 + i, 1 + j:padded.shape[1] - 1 + j]
                   for i in (-1, 0, 1) for j in (-1, 0, 1)], axis=0)
    spread = np.where(crossing.any(axis=2), np.where(crossing, tau, -np.inf).max(axis=2) - lowest, 0.0)
    return (1.0 - near) * (1.0 / (2 * samples) + math.sqrt(2.0) * sagitta_px) + spread


def rounding(dtype: str) -> float:
    return 64.0 * float(np.finfo(np.float32 if dtype == "float32" else np.float64).eps)


@pytest.mark.parametrize("samples", [2, 4, 8])
def test_the_tubes_drawing_images_as_its_closed_form(dtype, samples, monkeypatch):
    closed, traced, segments = tube_scene(samples, monkeypatch)
    route1, d, along = structure_of(closed)
    mesh, _, _ = structure_of(traced)
    rays = sub_rays(d, along, samples)
    true, low, high = tube_bounds(rays, segments)
    least, most = bracket(true, low, high, MU)
    slack = rounding(dtype)
    outside = np.maximum(least - slack - mesh, mesh - most - slack)
    assert outside.max() <= 0.0, f"the traced tube leaves the sagitta's bracket by {outside.max():.3g} at " \
                                 f"{np.unravel_index(np.argmax(outside), outside.shape)}"
    dense = dense_rays(d, along)
    sagitta_px = (TUBE[0] + TUBE[1]) * (1.0 - math.cos(math.pi / segments)) * DETECTOR.fx / (1e-3 - TUBE[0] - TUBE[1])
    line = design_line(tube_chord(dense, TUBE[0] + TUBE[1]) - tube_chord(dense, TUBE[0]), samples, sagitta_px, MU)
    ratio = np.abs(mesh - route1) / (line + slack)
    assert ratio.max() <= 1.0, f"S={samples}: traced against route 1 at {ratio.max():.3g} of the line"


def route1_sphere(absorbers: bool) -> np.ndarray:
    """The sphere's closed form (route 1, exact disc coverage) as the offline renderer images it at the camera's place:
    the in-loop camera holds no static sphere (its disc window is sized from a pose its static part does not trace)."""
    from mime.sensing import Appearance, CameraPose, Subject, render

    solid = Subject("solid", "sphere", {"radius": SPHERE}, appearance=Appearance.attenuating(MU) if absorbers
                    else Appearance.opaque(), window=int(SPHERE * DETECTOR.fx / (1e-3 - SPHERE)) + 4)
    pose = CameraPose(np.array([0.0, -1e-3, 0.0]), np.asarray(LOOK_ACROSS))
    return np.asarray(render([solid], DETECTOR.pinhole(), pose).transmission, dtype=np.float64)


@pytest.mark.parametrize("absorbers", [False, True], ids=["opaque", "attenuating"])
def test_the_spheres_drawing_images_as_its_closed_form(dtype, absorbers):
    drawn = sphere_scene("ellipsoid", absorbers)
    node = drawn.build().graph.get_node("camera:")
    triangles = np.asarray(node.constants()["mesh_triangles"])
    mesh, d, along = structure_of(drawn)
    route1 = route1_sphere(absorbers)
    samples = camera_module.MESH_SAMPLES
    rays = sub_rays(d, along, samples)
    true = sphere_chord(rays, SPHERE)
    low = sphere_chord(rays, inscribed_radius(triangles))
    mu = MU if absorbers else None
    least, most = bracket(true, low, true, mu)
    slack = rounding(dtype)
    outside = np.maximum(least - slack - mesh, mesh - most - slack)
    assert outside.max() <= 0.0, f"the traced sphere leaves the sagitta's bracket by {outside.max():.3g} at " \
                                 f"{np.unravel_index(np.argmax(outside), outside.shape)}"
    sagitta_px = (SPHERE - inscribed_radius(triangles)) * DETECTOR.fx / (1e-3 - SPHERE)
    line = design_line(sphere_chord(dense_rays(d, along), SPHERE), samples, sagitta_px, mu)
    ratio = np.abs(mesh - route1) / (line + slack)
    assert ratio.max() <= 1.0, f"traced against route 1 at {ratio.max():.3g} of the line"


def test_the_brackets_are_narrow_enough_to_see_half_a_pixel(float64, monkeypatch):
    """The first line's resolution: its bracket is at most a few thousandths at any pixel, against the half pixel's
    hundredths that the control plants."""
    closed, _, segments = tube_scene(4, monkeypatch)
    _, d, along = structure_of(closed)
    true, low, high = tube_bounds(sub_rays(d, along, 4), segments)
    least, most = bracket(true, low, high, MU)
    assert float((most - least).max()) < 5e-3
    assert jax.config.jax_enable_x64
