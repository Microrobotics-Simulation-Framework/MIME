"""Field providers: what makes a field available at points. Here now: ``UniformRotating``, and
``PointDipole``, a magnet part's field."""

from mime.fields._point_dipole import PointDipole
from mime.fields._uniform_rotating import UniformRotating

__all__ = ["PointDipole", "UniformRotating"]
