"""The iterative allocation's run flags, which a law that allocates on that path declares as it is made; the flag
machinery itself (``raise_flag``, how a controller's node holds and the run counts its flags) is
``mime.nodes.control._flags``'s."""

from __future__ import annotations

from mime.nodes.control._flags import declared_flags, raise_flag
from mime.registry import Flag

#: The iterative allocation's flag (``ctx.allocator``), declared by a law that allocates on the iterative path.
UNCONVERGED = Flag(
    "allocation_unconverged",
    "at the controller's last tick an iterative allocation (ctx.allocator) did not converge: its commands are the "
    "last iterate, inside the box, and its derivative the implicit function's there, which solves nothing; the value "
    "is its stationarity over the problem's size",
    warns=True)
#: The iterative allocation's second flag: its derivative not resolved in the run's dtype.
UNRESOLVED = Flag(
    "allocation_derivative_unresolved",
    "at the controller's last tick an iterative allocation's (ctx.allocator) derivative was not resolved in the run's "
    "dtype: its tangent system, the exact Hessian's, is singular at the dtype's rank line, so that a derivative "
    "through the step can be wrong by more than its size (in float32 at a weak effort, the follower's problem, at "
    "every step; float64 resolves it); its commands are not affected; the value is the tangent system's condition "
    "times its order's units in the last place, an indicator of the derivative's error over its size, not a bound on it",
    warns=True)

__all__ = ["UNCONVERGED", "UNRESOLVED", "declared_flags", "raise_flag"]
