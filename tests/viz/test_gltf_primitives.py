"""USD's analytic shapes as the view tessellates them (``mime.viz._primitives``), at physical sizes and along each
axis: closed, faces counter-clockwise seen from outside (the signed volume positive, every face turned away from the
centre, its vertex normals agreeing), inscribed in the shape (the volume short of the analytic one by no more than
the stated share), the extents the shape's own, and the triangles counted at the stated resolution. The control: the
same shapes with every face turned fail the orientation checks, every draw."""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from mime.viz._primitives import SEGMENTS, Shape, signed_volume, tessellate

#: Lengths in a stage's units, metres or millimetres: a micrometre to a few metres.
lengths = st.floats(min_value=-6.0, max_value=0.5).map(lambda e: 10.0 ** e)
axes = st.sampled_from(["X", "Y", "Z"])
#: How far short of the analytic volume each inscribed shape may fall at SEGMENTS facets a turn (measured: the sphere
#: 0.712%, the cylinder 0.285%; the capsule lies between, by its share of each).
SHORT = {"Sphere": 0.0072, "Cylinder": 0.0029, "Capsule": 0.0072, "Cube": 1e-12}
RINGS = SEGMENTS // 2
TRIANGLES = {"Sphere": 2 * SEGMENTS * (RINGS - 1), "Cylinder": 4 * SEGMENTS, "Capsule": 2 * SEGMENTS * RINGS,
             "Cube": 12}


def problems(shape: Shape, volume: float, kind: str) -> list[str]:
    found: list[str] = []
    a, b, c = (shape.points[shape.triangles[:, k]] for k in range(3))
    faces = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    if not np.all(np.einsum("ij,ij->i", faces, centroid) > 0.0):
        found.append("a face turned towards the centre")
    normals = shape.normals[shape.triangles].mean(axis=1)
    if not np.all(np.einsum("ij,ij->i", faces, normals) > 0.0):
        found.append("a face against its vertex normals")
    if not np.allclose(np.linalg.norm(shape.normals, axis=1), 1.0, atol=1e-12):
        found.append("a normal not of unit length")
    held = signed_volume(shape.points, shape.triangles)
    if not -SHORT[kind] <= held / volume - 1.0 <= 1e-12:
        found.append(f"volume {held / volume - 1.0:+.4%} of the analytic one")
    if len(shape.triangles) != TRIANGLES[kind]:
        found.append(f"{len(shape.triangles)} triangles")
    edges = np.sort(np.concatenate([shape.triangles[:, [0, 1]], shape.triangles[:, [1, 2]],
                                    shape.triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if kind in ("Sphere", "Capsule") and not np.all(counts == 2):
        found.append("an edge not shared by two faces")
    return found


def shapes(radius: float, height: float, axis: str) -> dict[str, tuple[Shape, float]]:
    held: dict[str, tuple[Shape, float]] = {}
    for kind, volume in (("Sphere", 4.0 / 3.0 * math.pi * radius ** 3),
                         ("Cylinder", math.pi * radius ** 2 * height),
                         ("Capsule", math.pi * radius ** 2 * height + 4.0 / 3.0 * math.pi * radius ** 3),
                         ("Cube", (2.0 * radius) ** 3)):
        shape = tessellate(kind, radius=radius, height=height, size=2.0 * radius, axis=axis)
        assert shape is not None
        held[kind] = (shape, volume)
    return held


@given(radius=lengths, height=lengths, axis=axes)
def test_each_shape_is_closed_outward_and_inscribed_in_its_own(radius, height, axis):
    for kind, (shape, volume) in shapes(radius, height, axis).items():
        assert problems(shape, volume, kind) == [], kind


@given(radius=lengths, height=lengths, axis=axes)
def test_each_shape_spans_its_own_extent_along_its_axis(radius, height, axis):
    along = "XYZ".index(axis)
    for kind, (shape, _) in shapes(radius, height, axis).items():
        if kind in ("Sphere", "Cube"):
            continue
        reach = height / 2.0 + (radius if kind == "Capsule" else 0.0)
        extent = shape.points[:, along]
        assert math.isclose(extent.max(), reach, rel_tol=1e-12) and math.isclose(extent.min(), -reach, rel_tol=1e-12)
        across = np.delete(shape.points, along, axis=1)
        assert math.isclose(float(np.linalg.norm(across, axis=1).max()), radius, rel_tol=1e-12), kind


@given(radius=lengths, height=lengths, axis=axes)
def test_control_every_face_turned_fails_the_orientation_checks(radius, height, axis):
    for kind, (shape, volume) in shapes(radius, height, axis).items():
        turned = Shape(shape.points, shape.normals, shape.triangles[:, ::-1])
        found = problems(turned, volume, kind)
        assert "a face turned towards the centre" in found and "a face against its vertex normals" in found, kind
        assert any(line.startswith("volume -") for line in found), kind


def test_a_kind_it_does_not_draw_gives_nothing():
    assert tessellate("Cone") is None and tessellate("Points") is None
