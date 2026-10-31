"""The magnet's pose through its mount, compared with the reference's recorded pose: the rotor's position, its spin
axis and its phase about that axis, at the reference's sampled steps (every tenth) in the steady window.

A pose is ``[x, y, z, w, qx, qy, qz]``, the rotor frame's origin and orientation (frame to world), as the tag's motor
recorded ``rotor_pose_world`` and 0.3's motor publishes its rotor's row of ``frame_poses``. The spin axis is the rotor
frame's z (the motor's axis), and the phase the turn about that axis of the rotor's x (the magnet's moment) from the
other pose's. Each gate is a largest miss over the window: 0.3's pose less the reference's, against the emulation's
prediction of that difference (0.3's emulated pose less the tag's)."""

from __future__ import annotations

import numpy as np

from a0_runs import window

#: The three misses a pose comparison reads, and their units.
MISSES = {"magnet_position_m": "m", "magnet_spin_axis_rad": "rad", "magnet_phase_rad": "rad"}


def rotation(poses: np.ndarray) -> np.ndarray:
    """The rotation matrices of the poses' unit quaternions, ``(n, 3, 3)``."""
    q = np.asarray(poses, dtype=np.float64)[:, 3:]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)], axis=1)


def parts(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each pose's position, spin axis (its z) and moment's direction (its x)."""
    turned = rotation(poses)
    return np.asarray(poses, dtype=np.float64)[:, :3], turned[:, :, 2], turned[:, :, 0]


def phase(moment: np.ndarray, against: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The signed turn about ``axis`` from ``against`` to ``moment``, each projected onto the plane across the axis,
    wrapped to a half turn either way."""
    def across(v: np.ndarray) -> np.ndarray:
        v = v - np.sum(v * axis, axis=1, keepdims=True) * axis
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    a, b = across(against), across(moment)
    return np.arctan2(np.sum(np.cross(a, b) * axis, axis=1), np.sum(a * b, axis=1))


def tilt(axis: np.ndarray, against: np.ndarray) -> np.ndarray:
    """The angle between two spin axes."""
    return np.arctan2(np.linalg.norm(np.cross(axis, against), axis=1), np.sum(axis * against, axis=1))


def difference(poses: np.ndarray, against: np.ndarray) -> dict[str, np.ndarray]:
    """``poses`` less ``against`` at each sample: the position's difference (``(n, 3)``, m), the spin axis's tilt
    from ``against``'s (rad) and the phase's turn from ``against``'s about ``against``'s axis (rad)."""
    p, z, x = parts(poses)
    q, w, y = parts(against)
    return {"position": p - q, "spin_axis": tilt(z, w), "phase": phase(x, y, w)}


def misses(poses: np.ndarray, reference: np.ndarray, predicted: dict[str, np.ndarray], time: np.ndarray,
           frequency: float) -> dict[str, float]:
    """The largest miss over the steady window of ``poses`` against ``reference`` moved by the ``predicted``
    difference (a :func:`difference` of the emulation's two schemes): the position's distance from where the prediction
    puts it, the spin axis's tilt beyond the predicted tilt, and the phase's turn beyond the predicted turn."""
    found = difference(poses, reference)
    inside = window(np.asarray(time, dtype=np.float64), frequency)
    turn = np.angle(np.exp(1j * (found["phase"] - predicted["phase"])))
    return {"magnet_position_m": float(np.max(np.linalg.norm((found["position"] - predicted["position"])[inside],
                                                             axis=1))),
            "magnet_spin_axis_rad": float(np.max(np.abs(found["spin_axis"] - predicted["spin_axis"])[inside])),
            "magnet_phase_rad": float(np.max(np.abs(turn[inside])))}


__all__ = ["MISSES", "difference", "misses", "parts", "phase", "rotation", "tilt"]
