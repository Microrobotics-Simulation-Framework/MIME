# Derived from usd2gltf 0.3.5 (https://github.com/mikelyndon/usd2gltf), MIT, Copyright (c) 2020 mikelyndon.
# Modified for MIME: typed; a node's local transform read from its xform ops in place of XformCommonAPI's Euler
# angles: translate, orient and scale ops taken as glTF's translation, rotation and scale exactly, at each time sample
# (the quaternions made continuous in sign, so that glTF's interpolation takes the short way), and any other stack
# taken as its matrix (decomposed where it is animated); the accessor bookkeeping moved to the converter.
# SPDX-License-Identifier: MIT
"""A prim's local transform as glTF holds it: a translation, a rotation (x, y, z, w) and a scale, or a matrix, and the
time samples of those that vary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mime.scene._pxr import Gf, Usd, UsdGeom
from mime.viz._gltf._common import quaternion_xyzw

#: The scale an arrow takes where its vector is zero: glTF has no visibility over time, and a scale of nothing hides it.
HIDDEN_SCALE = 1e-6


@dataclass
class Pose:
    """A node's transform at rest (its first time sample where it moves), and its tracks: the time codes and, for each
    of ``translation``, ``rotation`` and ``scale`` that varies, its value at each."""

    translation: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float, float] | None = None
    scale: tuple[float, float, float] | None = None
    matrix: list[float] | None = None
    codes: np.ndarray | None = None
    tracks: dict[str, np.ndarray] = field(default_factory=dict[str, np.ndarray])


def continuous(rotations: np.ndarray) -> np.ndarray:
    """Unit quaternions (rows of x, y, z, w), each turned to the sign nearer the one before it: ``q`` and ``-q`` are one
    turn, and glTF interpolates between the values it holds, so an antipodal pair would spin a body the long way."""
    held = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    for row in range(1, len(held)):
        if float(np.dot(held[row], held[row - 1])) < 0.0:
            held[row] = -held[row]
    return held


def _matrix(value: Any) -> list[float]:
    return [float(value[row][column]) for row in range(4) for column in range(4)]


def _codes(ops: list[Any]) -> np.ndarray:
    held: set[float] = set()
    for op in ops:
        held.update(float(code) for code in op.GetTimeSamples())
    return np.asarray(sorted(held), dtype=np.float64)


def _trs_ops(ops: list[Any]) -> dict[str, Any] | None:
    """The ops by role where the stack is at most one translate, then one orient, then one scale; else ``None``."""
    roles = {UsdGeom.XformOp.TypeTranslate: "translation", UsdGeom.XformOp.TypeOrient: "rotation",
             UsdGeom.XformOp.TypeScale: "scale"}
    order = ["translation", "rotation", "scale"]
    held: dict[str, Any] = {}
    for op in ops:
        role = roles.get(op.GetOpType())
        if role is None or role in held or op.IsInverseOp() or (held and order.index(role) < order.index(list(held)[-1])):
            return None
        held[role] = op
    return held


def _value(role: str, value: Any) -> np.ndarray:
    if role == "rotation":
        return np.asarray(quaternion_xyzw(value), dtype=np.float64)
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def pose_of(prim: Any) -> Pose:
    """The local transform of the xformable ``prim``, with the time samples its ops hold."""
    xformable = UsdGeom.Xformable(prim)
    ops: list[Any] = list(xformable.GetOrderedXformOps())
    pose = Pose()
    if not ops:
        return pose
    codes = _codes(ops)
    roles = _trs_ops(ops)
    if roles is None:
        if not len(codes):
            matrix = _matrix(xformable.GetLocalTransformation(Usd.TimeCode.Default()))
            if matrix != _matrix(Gf.Matrix4d(1.0)):
                pose.matrix = matrix
            return pose
        translations: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for code in codes:
            transform = Gf.Transform(xformable.GetLocalTransformation(Usd.TimeCode(float(code))))
            translations.append(_value("translation", transform.GetTranslation()))
            rotations.append(_value("rotation", transform.GetRotation().GetQuat()))
            scales.append(_value("scale", transform.GetScale()))
        values = {"translation": np.stack(translations), "rotation": np.stack(rotations), "scale": np.stack(scales)}
    else:
        values: dict[str, np.ndarray] = {}
        for role, op in roles.items():
            if len(codes) and op.GetTimeSamples():
                values[role] = np.stack([_value(role, op.Get(Usd.TimeCode(float(code)))) for code in codes])
            else:
                values[role] = _value(role, op.Get(Usd.TimeCode.Default()))[None, :]
    if "rotation" in values:
        values["rotation"] = continuous(values["rotation"])
    if "scale" in values:
        values["scale"] = np.where(values["scale"] == 0.0, HIDDEN_SCALE, values["scale"])
    for role, rows in values.items():
        first = tuple(float(v) for v in rows[0])
        setattr(pose, role, first)
        if len(rows) > 1 and not np.all(rows == rows[0]):
            pose.tracks[role] = rows
    if pose.tracks:
        pose.codes = codes
    return pose


__all__ = ["HIDDEN_SCALE", "Pose", "continuous", "pose_of"]
