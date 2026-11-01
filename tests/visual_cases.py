"""A stand-in for a visual set's baked meshes, planted in a temporary asset cache: CI never fetches the AR4's meshes,
so the tests bake a synthetic sphere under each of the set's files, each scaled apart so that its bytes are its own."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mime.scene import visual_sets
from mime.viz import Mesh, baked_key, decimate, visuals_dir, write_mesh
from mime.viz._visuals import index_of


def icosphere(subdivisions: int, radius: float = 0.05) -> Mesh:
    t = (1 + math.sqrt(5)) / 2
    points = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
              (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
             (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    held = [np.asarray(p, dtype=np.float64) / np.linalg.norm(p) for p in points]
    for _ in range(subdivisions):
        middle: dict[tuple[int, int], int] = {}

        def between(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in middle:
                point = held[a] + held[b]
                held.append(point / np.linalg.norm(point))
                middle[key] = len(held) - 1
            return middle[key]

        faces = [f for a, b, c in faces for f in ((a, between(a, b), between(c, a)), (b, between(b, c),
                                                  between(a, b)), (c, between(c, a), between(b, c)),
                                                  (between(a, b), between(b, c), between(c, a)))]
    return Mesh(np.asarray(held) * radius, np.asarray(faces, dtype=np.int64))


def planted(name: str = "ar4_mk5") -> dict[str, Path]:
    """Bake a sphere under every file of the set ``name`` into the asset cache (``MIME_ASSETS_DIR``), as ``mime visuals
    fetch`` would bake the set's own meshes; the layers by file."""
    held = visual_sets()[name]
    layers: dict[str, Path] = {}
    files: dict[str, dict[str, str]] = {}
    for index, source in enumerate(held.files):
        sphere = icosphere(2, radius=0.01 + 0.001 * index)
        mesh = decimate(Mesh(sphere.points + np.array([0.0, 0.0, 0.02]), sphere.triangles), 400).mesh
        key = baked_key(source.sha256, 400)
        layers[source.path] = write_mesh(visuals_dir() / f"{key}.usdc", mesh, "mesh")
        files[source.path] = {"sha256": source.sha256, "layer": f"{key}.usdc"}
    index_of(held).parent.mkdir(parents=True, exist_ok=True)
    index_of(held).write_text(json.dumps({"set": name, "files": files}), encoding="utf-8")
    return layers


def mesh_bytes(layers: dict[str, Path]) -> list[bytes]:
    """What would betray a baked mesh in a file: each layer's own bytes, and a window of its points as float32."""
    from pxr import Usd, UsdGeom

    found: list[bytes] = []
    for path in layers.values():
        found.append(path.read_bytes()[64:4096])
        stage = Usd.Stage.Open(str(path))
        points = np.asarray(UsdGeom.Mesh(stage.GetDefaultPrim()).GetPointsAttr().Get(), dtype=np.float32)
        found.append(points[3:13].tobytes())
    return found


def holds_any(data: bytes, needles: list[bytes]) -> bool:
    return any(needle in data for needle in needles)
