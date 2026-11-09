"""A second body model on the near field's contract, test-only: a regularised-Stokeslet bead line (a rigid body of beads
on a straight segment of its own frame, one blob, a block of three a bead, its kernel at the beads only). It holds the
contract at another block size and another self-block than the MRS's, so that phase R's line and RFT, and a wall patch
later, register against the same seam without changing the joint system."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jax import Array

from mime.core import quaternion
from mime.nodes.hydro.mrs._quadrature import regularised_stokeslet
from mime.nodes.hydro.mrs._walls import WallCorrection
from mime.nodes.hydro.near_field import CROSS_KERNEL, Placed, cross_image, point_block


class BeadLine:
    """``count`` beads along the body's x axis from ``-length/2`` to ``length/2``, off it by ``thickness/2`` to either
    side in turn (a zigzag, so that a turn about the axis meets resistance: beads on the axis itself would give a zero
    diagonal, the semidefinite case a line model's own check must take, phase R's), each standing for a cell of area
    ``pi thickness spacing``, the blob ``blob`` times the spacing."""

    def __init__(self, count: int, length: float, thickness: float, blob: float = 0.5) -> None:
        self._count, self.length, self.thickness, self.blob = count, length, thickness, blob

    @property
    def count(self) -> int:
        return self._count

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({CROSS_KERNEL})

    def placed(self, leaves: Mapping[str, Any], pose: Array) -> Placed:
        spacing = self.length / (self._count - 1)
        along = jnp.linspace(-0.5 * self.length, 0.5 * self.length, self._count, dtype=pose.dtype)
        side = 0.5 * self.thickness * jnp.where(jnp.arange(self._count) % 2 == 0, 1.0, -1.0).astype(pose.dtype)
        beads = jnp.stack([along, side, jnp.zeros_like(along)], axis=1)
        turn = quaternion.rotation_matrix(pose[3:])
        force = pose[:3] + beads @ turn.T
        area = jnp.full(self._count, math.pi * self.thickness * spacing, dtype=pose.dtype)
        eps = jnp.asarray((self.blob * spacing) ** 2, dtype=pose.dtype)
        return Placed(force, force, area, eps, pose[:3])

    def self_block(self, placed: Placed, leaves: Mapping[str, Any], held: Array | None, pose: Array, viscosity: Any,
                   wall: WallCorrection, parameters: Mapping[str, Any]) -> tuple[Array, Array, Array]:
        block = point_block(placed.force, placed) / viscosity
        image = cross_image(wall, placed, placed, parameters, viscosity)
        return block + image.matrix, image.accepted, jnp.asarray(False)

    def cross(self, placed: Placed, targets: Array, reach: Any) -> tuple[Array, Array]:
        del reach  # its kernel is at its beads at every distance: no integrated near part
        blocks, _ = regularised_stokeslet(placed.eps_squared)(targets, placed.force)
        blocks = blocks * placed.area[None, :, None, None]
        return blocks.transpose(0, 2, 1, 3).reshape(3 * targets.shape[0], 3 * placed.count), jnp.asarray(False)
