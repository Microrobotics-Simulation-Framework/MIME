# Derived from usd2gltf 0.3.5 (https://github.com/mikelyndon/usd2gltf), MIT, Copyright (c) 2020 mikelyndon.
# Modified for MIME: typed; the colour clamp and the quaternion order kept, the first-sample read of a static value
# kept and reached through mime.scene._pxr; the Euler-angle helpers, the extras and the logging dropped.
# SPDX-License-Identifier: MIT
"""Small reads shared by the converter: an attribute's static value, a colour clamped to glTF's range, and a USD
quaternion in glTF's order."""

from __future__ import annotations

from typing import Any

from mime.scene._pxr import Usd


def static_value(attribute: Any) -> Any:
    """An attribute's value where glTF cannot change it over time (a mesh's topology): its first time sample, else its
    default."""
    samples: list[float] = list(attribute.GetTimeSamples())
    if samples:
        return attribute.Get(samples[0])
    return attribute.Get(Usd.TimeCode.Default())


def clamped(colour: Any) -> tuple[float, float, float]:
    """A colour's three components, each clamped to [0, 1]."""
    return (max(min(float(colour[0]), 1.0), 0.0), max(min(float(colour[1]), 1.0), 0.0),
            max(min(float(colour[2]), 1.0), 0.0))


def quaternion_xyzw(quat: Any) -> tuple[float, float, float, float]:
    """A ``Gf`` quaternion, (real, imaginary), in glTF's order: its imaginary part, then its real."""
    imaginary = quat.GetImaginary()
    return float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(quat.GetReal())


__all__ = ["clamped", "quaternion_xyzw", "static_value"]
