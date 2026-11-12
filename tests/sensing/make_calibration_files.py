"""Writes ``calibration_files/``: calibration files as ``cv2.FileStorage`` writes them (YAML and XML), and
``calibration_goldens.json``, the sha256 of each and OpenCV's ``projectPoints`` for the camera each holds. Run once, with
OpenCV beside numpy (it is not one of MIME's requirements)::

    pip install opencv-python-headless==4.10.0.84
    python tests/sensing/make_calibration_files.py

The draws are seeded; the goldens pin each file's sha256, so a regenerated file is a deliberate change. Every file holds
all five coefficients distinct and non-zero (a swapped p1 and p2, or a dropped k3, then moves the pixels), the camera
matrix and the image size, as OpenCV's calibration sample writes them. Two more files are for the refusals: one at the
rational model's eight coefficients, one whose image size differs from the camera a test builds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
POINTS = 24


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def write(path: Path, matrix: np.ndarray, coefficients: np.ndarray, width: int, height: int) -> None:
    store = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    store.write("image_width", width)
    store.write("image_height", height)
    store.write("camera_matrix", matrix)
    store.write("distortion_coefficients", coefficients.reshape(-1, 1))
    store.release()


def main() -> None:
    rng = np.random.default_rng(20261112)
    cases = []
    sums = {}
    for index, (name, width, height) in enumerate((("camera_a.yml", 640, 480), ("camera_b.xml", 320, 240),
                                                   ("camera_c.yml", 160, 120))):
        fx = rng.uniform(0.9, 1.5) * width
        fy = fx * rng.uniform(0.99, 1.01)
        cx, cy = (width - 1) / 2 + rng.uniform(-6, 6), (height - 1) / 2 + rng.uniform(-5, 5)
        coefficients = np.array([rng.uniform(-0.3, -0.1), rng.uniform(0.02, 0.1), rng.uniform(-2e-3, -5e-4),
                                 rng.uniform(5e-4, 2e-3), rng.uniform(-0.05, -0.01)])
        matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        write(HERE / "calibration_files" / name, matrix, coefficients, width, height)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        position = rng.uniform(-0.05, 0.05, size=3)
        rotation = quaternion_matrix(q)
        depth = rng.uniform(0.05, 0.2, size=POINTS)
        across = np.stack([rng.uniform(-0.45, 0.45, POINTS) * width / fx, rng.uniform(-0.45, 0.45, POINTS) * height / fy],
                          axis=-1) * depth[:, None]
        world = np.concatenate([across, depth[:, None]], axis=-1) @ rotation.T + position
        rvec, _ = cv2.Rodrigues(rotation.T)
        pixels, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec, -rotation.T @ position, matrix, coefficients)
        cases.append({"file": name, "width": width, "height": height, "matrix": [fx, fy, cx, cy],
                      "coefficients": coefficients.tolist(), "position": position.tolist(), "orientation": q.tolist(),
                      "points": world.tolist(), "pixels": pixels.reshape(-1, 2).tolist()})
    rational = np.array([-0.2, 0.05, 1e-3, -1e-3, -0.02, 0.01, 0.002, -0.001])
    write(HERE / "calibration_files" / "rational.yml", np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]]),
          rational, 640, 480)
    for path in sorted((HERE / "calibration_files").iterdir()):
        sums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE / "calibration_goldens.json").write_text(json.dumps({"opencv": cv2.__version__, "sha256": sums, "cases": cases})
                                                   + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
