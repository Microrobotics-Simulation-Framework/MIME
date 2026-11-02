"""The configurations every registered node's step is traced in, for the switch scan and the comparison of steps'
jaxprs: script A's four arms (riding the floor, centred, falling onto the floor, starting past the drag's clamp),
every round-trip case whose build is not refused, which together build every registered entry's nodes, and A0's
controller inside the compiled step, with its orientation feedback on and a latency, whose step is a node's only
there (one inverse-kinematics iteration: each runs the same functions, so more trace more and scan nothing more); a
dynamic arm with Coulomb friction (``chain_cases.friction_arm``), whose friction and sweeps' start A0's AR4 never
reaches; a rod held by its hull (``hull_cases.rod_contact``), whose cylinder's support no case reaches; and
those
steps traced and scanned, once per process, for the tests that read the scan. ``step_jaxprs.py`` also runs these
configurations on a merge base, whose package has no switch scan, so the scan is imported only where it is used."""

from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

import jax
from cases import CASES
from chain_cases import friction_arm, tracked_rpm, tracking
from hull_cases import rod_contact
from script_a import script_a

import mime
from mime.experiment import Experiment
from mime.units import mm

if TYPE_CHECKING:
    from mime.docs._switches import SwitchScan, TracedStep

#: The package the switch scan reads.
PACKAGE = Path(mime.__file__).parent
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
    found["controller/tracked_rpm_in_graph"] = partial(tracked_rpm, "tracked_rpm_in_graph", mode="graph",
                                                       controller=tracking(orientation_feedback=True, iterations=1))
    found["chain/friction_arm"] = friction_arm
    found["contact/rod_hull"] = rod_contact
    return found


@cache
def traced_steps() -> tuple["TracedStep", ...]:
    """Every node's step in every configuration, traced in float64 with no switch context open."""
    from mime.build._maddening import node_state, parameter_leaves, resolved_inputs, zero_inputs
    from mime.docs._switches import traced_step

    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        traced: list[TracedStep] = []
        for configuration, make in configurations().items():
            graph = make().build().graph
            for name in graph.schedule:
                node = graph.get_node(name)
                given = {**zero_inputs(node), **resolved_inputs(graph, name)}
                traced.append(traced_step(configuration, name, node, node_state(graph, name), given,
                                          parameter_leaves(graph).get(name)))
        return tuple(traced)
    finally:
        jax.config.update("jax_enable_x64", held)


@cache
def tree_scan() -> "SwitchScan":
    """The switch scan of MIME's package over :func:`traced_steps`."""
    from mime.docs._switches import scan_switches

    return scan_switches(traced_steps(), PACKAGE)
