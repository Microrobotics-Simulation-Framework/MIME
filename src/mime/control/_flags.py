"""The iterative allocation's run flag, which a law that allocates on that path declares as it is made; the flag
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

__all__ = ["UNCONVERGED", "declared_flags", "raise_flag"]
