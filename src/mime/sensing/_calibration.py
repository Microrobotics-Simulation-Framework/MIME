"""A calibration file as OpenCV writes it: ``cv2.FileStorage``'s YAML or XML holding what ``calibrateCamera`` returns, the
camera matrix, the distortion coefficients and the image size (``camera_matrix``, ``distortion_coefficients``,
``image_width`` and ``image_height``, the names OpenCV's own calibration sample writes).

What is read is a fit, not a declaration: it is labelled with the file's name and sha256. A distortion model beyond the five
coefficients ``k1, k2, p1, p2, k3`` (OpenCV's rational, thin-prism and tilted models write 8, 12 and 14) is refused, as is
an image size other than the camera's it is read for: a matrix fitted at one size is not the same camera at another.
OpenCV's fisheye model writes four coefficients under the same name as the pinhole model's four, and the file does not
say which it is: the reader takes four as the pinhole model's ``k1, k2, p1, p2``."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass

#: The names OpenCV's calibration sample writes.
MATRIX, DISTORTION, WIDTH, HEIGHT = "camera_matrix", "distortion_coefficients", "image_width", "image_height"

_MATRIX_YAML = re.compile(r"(?P<name>\w+):\s*!!opencv-matrix\s+rows:\s*(?P<rows>\d+)\s+cols:\s*(?P<cols>\d+)\s+"
                          r"dt:\s*\w+\s+data:\s*\[(?P<data>[^\]]*)\]")
_SCALAR_YAML = re.compile(r"^(?P<name>\w+):\s*(?P<value>[-+0-9.eE]+)\s*$", re.MULTILINE)
_MATRIX_XML = re.compile(r"<(?P<name>\w+)\s+type_id=\"opencv-matrix\">\s*<rows>(?P<rows>\d+)</rows>\s*<cols>(?P<cols>\d+)"
                         r"</cols>\s*<dt>\w+</dt>\s*<data>(?P<data>[^<]*)</data>")
_SCALAR_XML = re.compile(r"<(?P<name>\w+)>\s*(?P<value>[-+0-9.eE]+)\s*</(?P=name)>")


@dataclass(frozen=True)
class Calibration:
    """What a file holds: the camera matrix's four numbers, the five coefficients in OpenCV's order (``k3`` zero where the
    file holds four), the image size (``None`` where the file holds none), and the file's name and sha256."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]
    width: int | None
    height: int | None
    name: str
    sha256: str

    @property
    def source(self) -> str:
        return f"fitted: {self.name}, sha256 {self.sha256}"


def _numbers(text: str) -> list[float]:
    return [float(token) for token in re.split(r"[\s,]+", text.strip()) if token]


def read_calibration(path: str | Path) -> Calibration:
    """The calibration ``path`` holds, refusing (``camera.calibration-file``) a file that is not OpenCV's, or lacks the
    camera matrix or the distortion coefficients."""
    file = Path(path)
    raw = file.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    if "<!DOCTYPE" in text or "<!ENTITY" in text:
        raise ArgumentRefused([Refusal("camera.calibration-file", RefusalClass.USER,
                                       f"{file.name} declares a document type or an entity",
                                       "give the file as `cv2.FileStorage` writes it, which declares neither")])
    yaml_form = "%YAML" in text or "!!opencv-matrix" in text
    matrices = {m["name"]: m for m in (_MATRIX_YAML if yaml_form else _MATRIX_XML).finditer(text)}
    scalars = {m["name"]: m["value"] for m in (_SCALAR_YAML if yaml_form else _SCALAR_XML).finditer(text)}
    for needed in (MATRIX, DISTORTION):
        if needed not in matrices:
            raise ArgumentRefused([Refusal(
                "camera.calibration-file", RefusalClass.USER, f"{file.name} holds no {needed!r} matrix",
                f"give the file `calibrateCamera`'s results were written to with `cv2.FileStorage`: {MATRIX}, "
                f"{DISTORTION}, {WIDTH}, {HEIGHT}")])
    matrix = matrices[MATRIX]
    values = _numbers(matrix["data"])
    if (int(matrix["rows"]), int(matrix["cols"])) != (3, 3) or len(values) != 9:
        raise ArgumentRefused([Refusal(
            "camera.calibration-file", RefusalClass.USER,
            f"{file.name}'s {MATRIX} is {matrix['rows']} by {matrix['cols']}, not 3 by 3",
            "write the camera matrix `calibrateCamera` returned with `cv2.FileStorage`")])
    coefficients = _numbers(matrices[DISTORTION]["data"])
    if len(coefficients) not in (4, 5):
        raise ArgumentRefused([Refusal(
            "camera.calibration-model", RefusalClass.USER,
            f"{file.name} holds {len(coefficients)} distortion coefficients, and the camera's model is the five of k1, "
            f"k2, p1, p2, k3 (four are read as k1, k2, p1, p2)",
            "calibrate with `calibrateCamera` and flags=0 (no `CALIB_RATIONAL_MODEL`, `CALIB_THIN_PRISM_MODEL` or "
            "`CALIB_TILTED_MODEL`), or undistort in OpenCV and render without distortion")])
    k1, k2, p1, p2, *rest = (*coefficients, 0.0)[:5] if len(coefficients) == 4 else coefficients
    k3 = rest[0] if rest else 0.0
    width = int(float(scalars[WIDTH])) if WIDTH in scalars else None
    height = int(float(scalars[HEIGHT])) if HEIGHT in scalars else None
    held = np.asarray(values).reshape(3, 3)
    return Calibration(float(held[0, 0]), float(held[1, 1]), float(held[0, 2]), float(held[1, 2]),
                       (k1, k2, p1, p2, k3), width, height, file.name, digest)


def checked_size(held: Calibration, width: int | None, height: int | None) -> tuple[int, int]:
    """The image size the calibration is read for: the camera's ``width`` by ``height`` where given, which the file's must
    equal (``camera.calibration-size``); the file's where not given, and refused where the file holds none."""
    if width is None or height is None:
        if held.width is None or held.height is None:
            raise ArgumentRefused([Refusal(
                "camera.calibration-size", RefusalClass.USER, f"{held.name} holds no image size, and none was given",
                "give the camera's size: `Pinhole.from_calibration(file, width=640, height=480)`")])
        return held.width, held.height
    if held.width is not None and held.height is not None and (held.width, held.height) != (width, height):
        raise ArgumentRefused([Refusal(
            "camera.calibration-size", RefusalClass.USER,
            f"{held.name} was calibrated at {held.width} by {held.height} pixels, and the camera is {width} by {height}",
            "calibrate at the camera's size, or build the camera at the file's: `Pinhole.from_calibration(file)`; a "
            "fitted matrix is the camera at the size it was fitted, and scaling it is a declared change, not an import")])
    if held.width is None or held.height is None:
        raise ArgumentRefused([Refusal(
            "camera.calibration-size", RefusalClass.USER,
            f"{held.name} holds no image size to check the camera's against",
            "write `image_width` and `image_height` beside the matrix, as OpenCV's calibration sample does")])
    return width, height
