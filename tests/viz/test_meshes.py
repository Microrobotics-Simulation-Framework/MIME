"""The mesh importer on a synthetic mesh (no description's meshes are fetched here, or in CI): an STL file read alike
from its binary form (whose header begins ``solid``, as a CAD tool's may) and its ASCII form; decimated within its
triangle budget and within the distance it states, the most a vertex moved, of the original, measured both ways
on every vertex; written as a USD mesh layer. The control: a decimated mesh with one vertex pushed out by three times
that distance fails the same check."""

import math
import struct

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from mime.viz import Mesh, TriangleBudget, decimate, mesh_distance, read_stl, write_mesh
from mime.viz._meshes import surface_distance
from visual_cases import icosphere


def binary_stl(mesh: Mesh) -> bytes:
    corners = mesh.points[mesh.triangles].astype(np.float32)
    body = b"".join(struct.pack("<3f", 0.0, 0.0, 0.0) + tri.tobytes() + b"\x00\x00" for tri in corners)
    return b"solid synthetic sphere".ljust(80, b" ") + struct.pack("<I", len(corners)) + body


def ascii_stl(mesh: Mesh) -> bytes:
    lines = ["solid sphere"]
    for tri in mesh.points[mesh.triangles].astype(np.float32):
        lines += ["facet normal 0 0 0", "outer loop", *(f"vertex {x!r} {y!r} {z!r}" for x, y, z in tri.tolist()),
                  "endloop", "endfacet"]
    return ("\n".join(lines + ["endsolid sphere"]) + "\n").encode("ascii")


SPHERE = icosphere(4)


def test_an_stl_file_reads_alike_from_its_binary_and_ascii_forms():
    binary, text = read_stl(binary_stl(SPHERE)), read_stl(ascii_stl(SPHERE))
    assert len(binary.triangles) == len(SPHERE.triangles) == 5120 and len(binary.points) == len(SPHERE.points)
    assert np.array_equal(binary.points, text.points) and np.array_equal(binary.triangles, text.triangles)
    with pytest.raises(ValueError, match="whole triangles"):
        read_stl(b"solid x\nvertex 0 0 0\nvertex 1 0 0\nendsolid\n")


def test_decimation_meets_its_budget_and_its_stated_distance():
    original = read_stl(binary_stl(SPHERE))
    for budget in (2000, 500, 120):
        held = decimate(original, budget)
        assert 0 < len(held.mesh.triangles) <= budget and held.moved == pytest.approx(held.cell * math.sqrt(3))
        assert mesh_distance(original, held.mesh) <= held.moved
    assert decimate(original, 10_000).mesh is original


def test_a_vertex_pushed_out_fails_the_distance_check():
    original = read_stl(binary_stl(SPHERE))
    held = decimate(original, 500)
    points = held.mesh.points.copy()
    points[0] *= 1 + 3 * held.moved / np.linalg.norm(points[0])
    planted = Mesh(points, held.mesh.triangles)
    assert mesh_distance(original, planted) > held.moved


def test_the_distance_is_exact_on_points_off_a_triangle():
    triangle = Mesh(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0, 1, 2]]))
    points = np.array([[0.2, 0.2, 0.5], [2.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]])
    assert surface_distance(points, triangle) == pytest.approx([0.5, 1.0, math.sqrt(2), math.sqrt(0.5)])


def test_a_set_shares_its_stage_budget_by_each_meshs_count():
    budget = TriangleBudget(per_mesh=20_000, per_stage=100_000)
    assert budget.shares([1000, 250_000, 750_000]) == [99, 20_000, 20_000]
    assert budget.shares([5000, 5000]) == [5000, 5000]
    assert sum(budget.shares([10_000] * 30)) <= 100_000


def test_a_mesh_layer_is_one_mesh_in_metres(tmp_path):
    held = decimate(SPHERE, 500).mesh
    path = write_mesh(tmp_path / "sphere.usdc", held, "mesh")
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh(stage.GetDefaultPrim())
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and len(mesh.GetFaceVertexCountsAttr().Get()) == \
        len(held.triangles)
    assert np.allclose(np.asarray(mesh.GetPointsAttr().Get()), held.points.astype(np.float32))
