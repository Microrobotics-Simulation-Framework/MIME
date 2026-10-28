"""Field providers: what makes a field available at points. Here now: ``UniformRotatingField``, and
``PointDipoleField``, a magnet part's field."""

from mime.fields._point_dipole import PointDipoleField
from mime.fields._uniform_rotating import UniformRotatingField

__all__ = ["PointDipoleField", "UniformRotatingField"]
