"""The configurations every registered node's step is traced in, for the switch scan and the comparison of steps'
jaxprs: script A's four arms (riding the floor, centred, falling onto the floor, starting past the drag's clamp),
every round-trip case whose build is not refused, which together build every registered entry's nodes, and A0's
controller inside the compiled step, with its orientation feedback on and a latency, whose step is a node's only
there (one inverse-kinematics iteration: each runs the same functions, so more trace more and scan nothing more); a
dynamic arm with Coulomb friction (``chain_cases.friction_arm``), whose friction and sweeps' start A0's AR4 never
reaches; a rod held by its hull (``hull_cases.rod_contact``), whose cylinder's support no case reaches; A0's
apparatus with its masses mounted and that friction arm, each with the energy ledger, whose metered steps (the arm's
and the motor's step identities, and the impulses split by the friction's share) no other configuration takes; a
sphere above a plane within the elliptic form's window, whose drag takes the near-contact hand-over (``lubrication_cases``), which
no round-trip case's body is fine enough for (where the package has the hand-over); and those steps traced and
scanned, once per process, for the tests that read the scan. ``step_jaxprs.py`` also runs these
configurations on a merge base, whose package has no switch scan, so the scan is imported only where it is used."""

from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

import jax
from cases import CASES
from chain_cases import friction_arm, massive_rpm, tracked_rpm, tracking
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
    found["ledger/massive_rpm"] = partial(metered, partial(massive_rpm, "massive_rpm_metered"))
    found["ledger/friction_arm"] = partial(metered, partial(friction_arm, "friction_arm_metered"))
    from mime.interactions import Drag

    if hasattr(Drag, "handover_window"):  # a merge base before the near-contact hand-over builds no such drag
        from lubrication_cases import above_plane

        from mime.scene import Ellipsoid
        from mime.units import mm

        # a coarse blob: the step's structure (its raw sites) is the same at any blob, and each worker that runs the
        # scan holds every configuration's traced step, the quadrature's arrays in it. The sphere takes the
        # near-contact law's sphere form; a mild ellipsoid, within the elliptic form's window, the film's
        found["drag/lubricated_sphere"] = partial(above_plane, 0.2, blob=0.25)
        if hasattr(Ellipsoid, "near_contact"):  # a merge base before the near-contact seam has no forms to tell apart
            found["drag/lubricated_ellipsoid"] = partial(above_plane, 0.16, blob=0.25, name="lubricated_ellipsoid",
                                                         kind=Ellipsoid(semi_axes=mm((1.1, 1.0, 0.95))))
    return found


def metered(make: Callable[[], Experiment]) -> Experiment:
    """``make``'s experiment with the energy ledger attached."""
    from mime.observations import EnergyLedger

    experiment = make()
    experiment.observe(EnergyLedger())
    return experiment


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
