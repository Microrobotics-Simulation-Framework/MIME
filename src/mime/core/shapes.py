"""Shapes for a lab's plugin: the :class:`Shape` protocol a region is evaluated by, MIME's own shapes in closed form,
and the registry nodes reach a shape through by name.

A shape gives one method, its exact signed distance (``signed_distance``), and its ``parameter_names``; it derives the
rest from that distance unless it gives its own: its outward normal (by autodiff, zero on its medial set, the switch
``<name>.kink``), its axis (none) and the camera's sign test (``sign_function``: the distance, slope one), and a
geometry kind whose regions are shapes derives its display mesh (contoured) and its sign function from
them. MIME's closed forms are those overrides: ``SphereShape``, ``CapsuleShape``, ``CylinderShape`` (unbounded),
``SlabShape``, ``HalfSpaceShape`` and ``ScrewShape`` (unbounded along its axis), each registered under its stem
(``"cylinder"``) beside its singleton (``mime.core.CYLINDER``). A script never names these: it declares the scene's
bodies (``mime.scene.Cylinder``, a cylinder cut by a slab), whose regions are these shapes."""

from mime.core._geometry import (
    AXIS_ULPS,
    KINK,
    Parameters,
    Point,
    Shape,
    SignValue,
    register_shape,
    shape_named,
    shape_names,
    shape_switches,
)
from mime.core._shapes import (
    CAPSULE,
    CYLINDER,
    HALF_SPACE,
    SCREW,
    SLAB,
    SPHERE,
    CapsuleShape,
    CylinderShape,
    HalfSpaceShape,
    ScrewShape,
    SlabShape,
    SphereShape,
    nearest_point,
)

__all__ = [
    "AXIS_ULPS", "CAPSULE", "CYLINDER", "HALF_SPACE", "KINK", "SCREW", "SLAB", "SPHERE", "CapsuleShape",
    "CylinderShape", "HalfSpaceShape", "Parameters", "Point", "ScrewShape", "Shape", "SignValue", "SlabShape",
    "SphereShape", "nearest_point", "register_shape", "shape_named", "shape_names", "shape_switches",
]
