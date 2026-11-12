"""Writes ``projection_goldens.json``: OpenCV's ``projectPoints`` on drawn cameras, poses and points, the oracle of the
projection gate. Run once, with OpenCV installed beside numpy (it is not one of MIME's requirements)::

    pip install opencv-python-headless==4.10.0.84
    python tests/sensing/make_projection_goldens.py

The draws are seeded; the file's sha256 is pinned by the gate, so a regenerated file is a deliberate change. Each case
holds the camera (``fx, fy, cx, cy`` and OpenCV's five coefficients in its order ``k1, k2, p1, p2, k3``), the pose in
MIME's convention (the optical frame's origin in the world and the unit quaternion ``(w, x, y, z)`` turning optical-frame
vectors into the world's), the world points and OpenCV's pixels for them."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CASES = 16
POINTS = 24


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main() -> None:
    rng = np.random.default_rng(20261111)
    cases = []
    for index in range(CASES):
        width, height = (640, 480) if index % 2 else (320, 240)
        fx = rng.uniform(0.8, 1.6) * width
        fy = fx * rng.uniform(0.98, 1.02)
        cx, cy = (width - 1) / 2 + rng.uniform(-8, 8), (height - 1) / 2 + rng.uniform(-6, 6)
        strength = 0.0 if index < 2 else 1.0
        distortion = strength * np.array([rng.uniform(-0.3, 0.2), rng.uniform(-0.1, 0.1), rng.uniform(-2e-3, 2e-3),
                                          rng.uniform(-2e-3, 2e-3), rng.uniform(-0.05, 0.05)])
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        position = rng.uniform(-0.05, 0.05, size=3)
        rotation = quaternion_matrix(q)
        # Points in front of the camera, inside its field: drawn in the optical frame, then placed in the world.
        depth = rng.uniform(0.05, 0.2, size=POINTS)
        across = np.stack([rng.uniform(-0.45, 0.45, POINTS) * width / fx, rng.uniform(-0.45, 0.45, POINTS) * height / fy],
                          axis=-1) * depth[:, None]
        local = np.concatenate([across, depth[:, None]], axis=-1)
        world = local @ rotation.T + position
        rvec, _ = cv2.Rodrigues(rotation.T)
        tvec = -rotation.T @ position
        matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        pixels, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec, tvec, matrix, distortion)
        cases.append({"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                      "distortion": distortion.tolist(), "position": position.tolist(), "orientation": q.tolist(),
                      "points": world.tolist(), "pixels": pixels.reshape(-1, 2).tolist()})
    out = Path(__file__).with_name("projection_goldens.json")
    out.write_text(json.dumps({"opencv": cv2.__version__, "cases": cases}) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(cases)} cases, OpenCV {cv2.__version__})")


if __name__ == "__main__":
    main()
