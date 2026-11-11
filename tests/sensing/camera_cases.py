"""Shared pieces of the camera core's tests: the planted errors their controls switch on (``CAMERA_PLANT``, read by
``.github/probe/detection_rate.py --plant-var CAMERA_PLANT``), the scenes, and a camera looking across script A's tube."""

from __future__ import annotations

import math
import os

import jax.numpy as jnp

from mime.sensing import Appearance, BrownConrady, CameraPose, Pinhole, Subject

#: The planted error a control switches on, by name; empty when nothing is planted.
PLANT = os.environ.get("CAMERA_PLANT", "")
#: Its size, where the plant has one (``CAMERA_PLANT_SIZE``).
PLANT_SIZE = float(os.environ.get("CAMERA_PLANT_SIZE", "0") or 0.0)
#: The seed of a control's draw (``DETECTION_SEED``, which detection_rate.py sets for each run).
SEED = int(os.environ.get("DETECTION_SEED", "0") or 0)

TUBE = {"axis": (1.0, 0.0, 0.0), "inner_radius": 3.175e-3, "wall": 1.6e-3, "length": 0.1, "centre": (0.0, 0.0, 0.0)}


def planted(name: str) -> bool:
    return PLANT == name


def across(width: int = 160, distortion: BrownConrady | None = None, field: float = 13.3e-3,
           distance: float = 0.1) -> tuple[Pinhole, CameraPose]:
    """A camera ``distance`` from script A's tube's axis, beside it (on -y), looking across it (+y), the image's
    right along the tube (+x) and its up the world's +z, its field ``field`` wide."""
    height = width * 3 // 4
    focal = distance / (field / width)
    s = math.sqrt(0.5)
    return (Pinhole(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2, distortion or BrownConrady()),
            CameraPose(jnp.array([0.0, -distance, 0.0]), jnp.array([s, -s, 0.0, 0.0])))


def tube_subjects(wall: Appearance, water: Appearance) -> list[Subject]:
    return [Subject("tube", "tube", TUBE, "wall", wall), Subject("water", "tube", TUBE, "lumen", water, parent="tube")]
