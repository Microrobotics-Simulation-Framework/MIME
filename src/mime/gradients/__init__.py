"""Gradient statuses, their composition and the gradient map's checks."""

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum

from mime.core.switch import Declaration


class GradientStatus(Enum):
    """How a derivative passes through an entry; every registered entry declares one.

    ``EXACT``: the derivative of the computed value. ``FIXED_KEY``: exact with every random key held fixed.
    ``PIECEWISE_KINK``: exact away from declared switches, all of them kinks (order 0), where the value is continuous
    and the derivative is the branch taken's. ``PIECEWISE_JUMP``: exact away from declared switches, at least one of
    them a jump (order -1), across which the value itself is discontinuous. ``UNVERIFIED``: computed but not yet
    checked. ``CUT``: no derivative passes, for a stated reason.
    """

    EXACT = "exact"
    FIXED_KEY = "fixed_key"
    PIECEWISE_KINK = "piecewise_kink"
    PIECEWISE_JUMP = "piecewise_jump"
    UNVERIFIED = "unverified"
    CUT = "cut"


#: How much each status promises, least first: a path is as good as the least of its entries'. ``FIXED_KEY`` is
#: ``EXACT`` with its random keys held.
RANK = {GradientStatus.CUT: 0, GradientStatus.UNVERIFIED: 1, GradientStatus.PIECEWISE_JUMP: 2,
        GradientStatus.PIECEWISE_KINK: 3, GradientStatus.EXACT: 4, GradientStatus.FIXED_KEY: 4}


def status_of_switches(switches: Sequence[Declaration]) -> GradientStatus:
    """The status an entry's switches give: ``PIECEWISE_JUMP`` with a switch of order -1 in some output group,
    ``PIECEWISE_KINK`` with one of order 0 and none lower, and ``EXACT`` with none, or none below order 1, across
    which the derivative is continuous."""
    lowest = min((declared.lowest_order for declared in switches), default=1)
    if lowest < 0:
        return GradientStatus.PIECEWISE_JUMP
    return GradientStatus.PIECEWISE_KINK if lowest == 0 else GradientStatus.EXACT


def derived_status(switches: Sequence[Declaration], cap: GradientStatus | None = None) -> GradientStatus:
    """What an entry's declared status must be: its switches' status, or ``cap`` where a listed raw site caps the
    entry lower."""
    found = status_of_switches(switches)
    return cap if cap is not None and RANK[cap] < RANK[found] else found


def status_problem(declared: GradientStatus, switches: Sequence[Declaration],
                   cap: GradientStatus | None = None) -> str | None:
    """Why ``declared`` is not the status its switches and listed sites give, or ``None``. With no switch and no cap
    an entry may declare ``FIXED_KEY``, ``UNVERIFIED`` or ``CUT`` for a reason of its own."""
    derived = derived_status(switches, cap)
    if declared is derived or (derived is GradientStatus.EXACT and declared is GradientStatus.FIXED_KEY):
        return None
    if not switches and cap is None and declared in (GradientStatus.UNVERIFIED, GradientStatus.CUT):
        return None
    if switches:
        lowest = min(declared_.lowest_order for declared_ in switches)
        owners = ", ".join(d.owner for d in switches if d.lowest_order == lowest)
        given = f"its switches' lowest order is {lowest} ({owners})"
    else:
        given = "it declares no switch"
    capped = f", and a listed raw site caps it at {cap.value}" if cap is not None else ""
    return f"it is declared {declared.value}, and {given}{capped}: it is {derived.value}"


def composed(statuses: Iterable[GradientStatus]) -> GradientStatus:
    """A path's status: the least of its entries' (:data:`RANK`), and ``EXACT`` for a path through no entry. A
    ``FIXED_KEY`` path holds its random keys fixed, so it is that where nothing on it promises less."""
    held = list(statuses)
    if not held:
        return GradientStatus.EXACT
    least = min(RANK[status] for status in held)
    found = [status for status in held if RANK[status] == least]
    return GradientStatus.FIXED_KEY if GradientStatus.FIXED_KEY in found else found[0]


#: The statuses a differentiated study refuses on its path unless each entry holding one is acknowledged with a reason:
#: no derivative passes (``CUT``), or one is computed that no check has earned (``UNVERIFIED``).
ACKNOWLEDGED_ONLY = frozenset({GradientStatus.CUT, GradientStatus.UNVERIFIED})


def unacknowledged(path: Mapping[str, GradientStatus], acknowledged: Mapping[str, str]) -> tuple[str, ...]:
    """The entries on ``path`` (by id) whose status is refused unless acknowledged, and that ``acknowledged`` (entry
    id to its reason, a sentence) does not acknowledge."""
    return tuple(sorted(owner for owner, status in path.items()
                        if status in ACKNOWLEDGED_ONLY and not acknowledged.get(owner, "").strip()))


__all__ = ["ACKNOWLEDGED_ONLY", "RANK", "GradientStatus", "composed", "derived_status", "status_of_switches",
           "status_problem", "unacknowledged"]
