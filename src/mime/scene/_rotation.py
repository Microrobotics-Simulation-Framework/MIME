"""A rotation as MIME writes it: a unit quaternion ``(w, x, y, z)``, built from the axis and the angle a script means."""

from __future__ import annotations

import math
from typing import Any, cast

from mime.scene._frame import unit_vector
from mime.units import DIMENSIONLESS, Quantity, UnitError, si

type Quaternion = tuple[float, float, float, float]


def _refused(message: str) -> Exception:
    """The refusal of ``rotation``'s arguments (``call.rotation``): the ``ValueError`` a call's argument raises, coded."""
    from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass

    return ArgumentRefused([Refusal("call.rotation", RefusalClass.USER, message,
                                    "write rotation(axis, angle) with an axis of three numbers, not all zero, and an "
                                    "angle with its unit: rotation(Z, deg(30.0)) or rotation((0.0, 1.0, 1.0), "
                                    "rad(0.5))")])


def rotation(axis: Any, angle: Quantity) -> Quaternion:
    """The unit quaternion that turns by ``angle`` about ``axis``, right-handed (counter-clockwise looking down the axis
    towards its origin): ``rotation(Z, deg(30.0))``, for ``State(orientation=...)``, a part's ``orientation=`` or any
    other place MIME takes a rotation.

    The order is **(w, x, y, z)**, the real part first, as everywhere in MIME: ``(1.0, 0.0, 0.0, 0.0)`` is no turn. ROS,
    SciPy and many others write ``(x, y, z, w)``, and ``(0.0, 0.0, 0.0, 1.0)`` in MIME is a half turn about z, not the
    identity. ``axis`` is a direction (``X``, ``-Z``, or three numbers: its length does not matter, zero is refused);
    ``angle`` is a quantity (``deg(30.0)``, ``rad(0.5)``), never a bare number."""
    direction = unit_vector(axis)
    if direction is None:
        raise _refused(f"axis={axis!r} is not a direction: three finite numbers, not all zero")
    if not isinstance(cast(object, angle), Quantity):  # a script is not type-checked
        raise _refused(f"angle={angle!r} is a bare number, and MIME does not assume whether it is degrees or radians")
    try:
        turned = si(angle, DIMENSIONLESS, what="rotation's angle")
    except UnitError as error:
        raise _refused(str(error)) from error
    if not isinstance(turned, float) or not math.isfinite(turned):
        raise _refused(f"angle={angle} is not one finite angle")
    half = 0.5 * turned
    sine = math.sin(half)
    return (math.cos(half) + 0.0, direction[0] * sine + 0.0, direction[1] * sine + 0.0, direction[2] * sine + 0.0)


__all__ = ["rotation"]
