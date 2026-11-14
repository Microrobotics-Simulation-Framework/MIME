"""The configurations every registered node's step is traced in, for the switch scan and the comparison of steps'
jaxprs: script A's four arms (riding the floor, centred, falling onto the floor, starting past the drag's clamp),
every round-trip case whose build is not refused, which together build every registered entry's nodes, and A0's
controller inside the compiled step, with its orientation feedback on and a latency, whose step is a node's only
there (one inverse-kinematics iteration: each runs the same functions, so more trace more and scan nothing more); a
test-only controller inside the compiled step that calls the exact allocation (``allocation_cases``), which no shipped
entry calls; a
dynamic arm with Coulomb friction (``chain_cases.friction_arm``), whose friction and sweeps' start A0's AR4 never
reaches; a rod held by its hull (``hull_cases.rod_contact``), whose cylinder's support no case reaches; A0's
apparatus with its masses mounted and that friction arm, each with the energy ledger, whose metered steps (the arm's
and the motor's step identities, and the impulses split by the friction's share) no other configuration takes; a
sphere above a plane within the film's window, whose drag takes lubrication's hand-over (``lubrication_cases``), which
no round-trip case's body is fine enough for (where the package has the hand-over); the point body with its fitted
self-history (``self_history="modes"``), which no round-trip case writes; and those steps, with each node's
run-constants as a derivative's run makes them, traced and scanned, once per process, for the tests that read the scan.
``step_jaxprs.py`` also runs these configurations on a merge base, whose package has no switch scan, so the scan is
imported only where it is used."""

from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    import mime.control

    if hasattr(mime.control, "exact_allocation"):  # a merge base before the exact allocation has no law to reach
        from allocation_cases import allocating_sphere

        found["controller/allocating_in_graph"] = allocating_sphere
    found["chain/friction_arm"] = friction_arm
    found["contact/rod_hull"] = rod_contact
    found["ledger/massive_rpm"] = partial(metered, partial(massive_rpm, "massive_rpm_metered"))
    found["ledger/friction_arm"] = partial(metered, partial(friction_arm, "friction_arm_metered"))
    from mime.interactions import Drag

    if "self_history" in getattr(Drag, "__dataclass_fields__", {}):  # a merge base before the fitted route has none
        from cases import point_body_flow

        found["drag/point_body_modes"] = partial(point_body_flow, "modes")
    if hasattr(Drag, "handover_window"):  # a merge base before lubrication's hand-over builds no such drag
        from lubrication_cases import above_plane

        from mime.scene import Ellipsoid
        from mime.units import mm

        # a coarse blob: the step's structure (its raw sites) is the same at any blob, and each worker that runs the
        # scan holds every configuration's traced step, the quadrature's arrays in it. The sphere takes the
        # near-contact law's sphere form; a mild ellipsoid, within the film's window, the film's
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
    """Every node's step in every configuration, traced in float64 with no switch context open, on caches emptied once
    (``mime.docs._switches.traced_steps``): what the process traced before does not change the scan's answer. Then each
    node's run-constants as a run a derivative is taken through makes them, once, before its steps
    (``mime.core._node.run_constants_for_derivative``): code a chunk runs that its step may only read (the camera's static
    structure, made there, is read by its step and never made in it)."""
    from mime.build._maddening import make_jaxpr, node_state, parameter_leaves, resolved_inputs, run_constant_nodes, \
        zero_inputs
    from mime.core._node import run_constants_for_derivative
    from mime.docs._switches import TracedStep
    from mime.docs._switches import traced_steps as trace

    making: list[tuple[str, str, Any, object]] = []

    def cases():
        for configuration, make in configurations().items():
            graph = make().build().graph
            leaves = parameter_leaves(graph)
            making.extend((configuration, name, graph.get_node(name), leaves.get(name))
                          for name in run_constant_nodes(graph))
            for name in graph.schedule:
                node = graph.get_node(name)
                given = {**zero_inputs(node), **resolved_inputs(graph, name)}
                yield configuration, name, node, node_state(graph, name), given, leaves.get(name)

    def made(node: Any) -> Callable[[object], object]:
        def run_constants(own: object) -> object:
            with run_constants_for_derivative():
                return node.run_constants(own)

        return run_constants

    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        steps = trace(cases())
        return steps + tuple(TracedStep(configuration, name, type(node), make_jaxpr(made(node), own))
                             for configuration, name, node, own in making)
    finally:
        jax.config.update("jax_enable_x64", held)


@cache
def tree_scan() -> "SwitchScan":
    """The switch scan of MIME's package over :func:`traced_steps`."""
    from mime.docs._switches import scan_switches

    return scan_switches(traced_steps(), PACKAGE)
