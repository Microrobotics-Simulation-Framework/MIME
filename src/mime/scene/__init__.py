"""The USD stage: frames, prim paths, analytic geometry and the stage writer; with viz, the only importer of pxr.

Here now: the analytic geometry kind :class:`Tube`, and the signed axes :data:`X`, :data:`Y` and :data:`Z`. A geometry
kind is a contract a lab implements: a registered ``mime.registry.Geometry`` naming its regions, surfaces and
directions, whose surfaces are evaluated by :class:`Shape` (the protocol: signed distance, outward normal, and the axis
where one exists), registered in ``mime.core``."""

from mime.core import Cylinder, Shape, register_shape
from mime.scene._geometry import Axis, Tube, X, Y, Z

__all__ = ["Axis", "Cylinder", "Shape", "Tube", "X", "Y", "Z", "register_shape"]
