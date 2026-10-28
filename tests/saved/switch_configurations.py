"""The configurations every registered node's step is traced in, for the switch scan and the comparison of steps'
jaxprs: script A's four arms (riding the floor, centred, falling onto the floor, starting past the drag's clamp) and
every round-trip case whose build is not refused, which together build every registered entry's nodes."""

from collections.abc import Callable
from functools import partial

from cases import CASES
from script_a import script_a

from mime.experiment import Experiment
from mime.units import mm

#: Where script A's body starts in each arm.
ARMS = {
    "floor": mm((0.0, 0.0, -1.0)),
    "centred": mm((0.0, 0.0, 0.0)),
    "falling": mm((0.0, 0.3, -0.6)),
    "past_clamp": mm((0.0, 0.0, -1.08)),
}


def configurations() -> dict[str, Callable[[], Experiment]]:
    """Each configuration's experiment, by a name: ``script_a/<arm>`` and ``case/<case>``."""
    found: dict[str, Callable[[], Experiment]] = {f"script_a/{arm}": partial(script_a, start=start)
                                                  for arm, start in ARMS.items()}
    found.update({f"case/{case.name}": case.make for case in CASES if not case.refused})
    return found
