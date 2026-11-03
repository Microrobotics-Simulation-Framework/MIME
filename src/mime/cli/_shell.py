"""``quoted`` is in ``mime/_shell.py``, since the commands ``mime push`` and ``mime pull`` print for pasting are built
inside the library, which does not import ``mime.cli``; this is its name in the command line's package."""

from mime._shell import quoted

__all__ = ["quoted"]
