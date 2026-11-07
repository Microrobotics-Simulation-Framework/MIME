"""An observation's read of a uniform field through the slot a law reads it by (the planner's ``share_uniform_reads``),
held four ways on script A, each with a control that fails on every run.

1. **The same value.** ``PhaseLag`` receives, at every step, the flux density the magnetic torque receives, bit for bit,
   and it is the provider's state of that step, not the step before: each node's input is recorded inside the compiled
   step, as a field of its own state its step is patched to carry (a host callback is refused in a step), the step
   traced afresh. The control: the observation's reads one step
   stale, as a read of the slot lagged by a step would give them, differ from the torque's at every step, by the field
   turning over the step, ``omega dt`` of its magnitude (0.031 at script A's 10 Hz and 0.5 ms): the error such a lag
   would put into the lag the observation measures.
2. **No perturbation.** Script A's history, every node's state and the laws' outputs, is the same bit for bit with the
   observation attached and without it, and the provider's node is the same: its class, its state's fields and its
   parameters. Measured first over script A's whole 5 s run, in float32 and float64, with 4 usable CPUs and
   ``OPENBLAS_NUM_THREADS=4``: bit for bit, so that is what is asserted (here over 0.25 s, in one process, at whatever
   counts it runs with). The controls: a start one unit in the last place off fails the history's comparison, and a
   second law on the field, which adds a slot, fails the provider's.
3. **The right slot.** With the torque, which reads the flux density, and the force, which reads the gradient, on one
   field, the observation shares the torque's slot, never the force's; the force's id sorts first, so a share blind to
   the interface would take the force's. The control: with the force alone no law reads the flux density, and the
   observation is refused (``observation.field-read``), the route attaching the torque.
4. **The build report** names the law whose slot the observation reads, among the lines the build computed, and never
   the force.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from script_a import script_a

from mime.build import BuildRefused
from mime.build._maddening import parameter_leaves
from mime.core import as_run_array
from mime.experiment import BuiltExperiment, Experiment
from mime.interactions import Magnetic
from mime.nodes.body import PhaseLagNode
from mime.nodes.magnetic import MagneticTorqueNode
from mime.observations import PhaseLag
from mime.scene import X
from mime.units import m, mm, s

LAG, TORQUE, FIELD = "phase_lag:umr", "magnetic.torque:umr", "field.magnetic"
SLOT = "flux_density_magnetic_torque_umr"
#: The state field the recording patch adds to each node: the flux density it received in the step.
RECEIVED = "received_flux_density"
#: Script A's field: 10 Hz, its step 0.5 ms, so the field turns by omega dt over a step.
OMEGA_DT = 2.0 * np.pi * 10.0 * 5e-4


@pytest.fixture(params=["float32", "float64"])
def dtype(request) -> Iterator[str]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    try:
        yield request.param
    finally:
        jax.config.update("jax_enable_x64", held)


def observed(start=mm((10.0, 0.0, -1.0))) -> Experiment:
    experiment = script_a(start=start)
    experiment.observe(PhaseLag("umr", field="field.magnetic", about=X))
    return experiment


def counts() -> str:
    return (f"{len(os.sched_getaffinity(0))} usable CPUs, OPENBLAS_NUM_THREADS="
            f"{os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}")


def test_the_observation_receives_what_the_torque_receives_at_every_step_bit_for_bit(dtype, monkeypatch):
    steps = 200
    traced = Counter[str]()

    def recording(cls: type, name: str) -> None:
        initial, step = cls.initial_state, cls.advance

        def initial_state(self):
            return {**initial(self), RECEIVED: as_run_array(jnp.zeros(3))}

        def advance(self, p, state, boundary_inputs, dt):
            traced[name] += 1
            held = {key: value for key, value in state.items() if key != RECEIVED}
            return {**step(self, p, held, boundary_inputs, dt), RECEIVED: boundary_inputs["flux_density"]}

        monkeypatch.setattr(cls, "initial_state", initial_state)
        monkeypatch.setattr(cls, "advance", advance)

    recording(PhaseLagNode, LAG)
    recording(MagneticTorqueNode, TORQUE)
    jax.clear_caches()
    built = observed().build()
    for name in (LAG, TORQUE):
        built.graph.get_node(name).__dict__.pop("_compiled", None)
    history = built.run(s(steps * 5e-4), history=True).history
    jax.clear_caches()
    assert history is not None
    assert traced[LAG] >= 1 and traced[TORQUE] >= 1, "the recording step was never traced"
    seen, torque = (np.asarray(history[name][RECEIVED])[1:] for name in (LAG, TORQUE))
    provider = np.asarray(history[FIELD][SLOT])
    assert seen.shape == torque.shape == (provider.shape[0] - 1, 3) and len(seen) >= steps - 1
    assert np.array_equal(seen, torque), f"the observation's input is not the torque's ({counts()})"
    assert np.array_equal(seen, provider[1:]), "the input is not the provider's state of the same step"
    stale = seen[:-1]
    magnitude = np.linalg.norm(torque[1:], axis=1)
    off = np.linalg.norm(torque[1:] - stale, axis=1) / magnitude
    print(f"\n{dtype}, {counts()}: {len(seen)} steps bit for bit; one step stale misses by {off.min():.4f} to "
          f"{off.max():.4f} of the field, omega dt {OMEGA_DT:.4f}")
    assert np.all(off > 0.5 * OMEGA_DT), "a read one step stale is not told apart at some step"
    assert np.allclose(off, 2.0 * np.sin(OMEGA_DT / 2.0), rtol=1e-3)


def histories(built: BuiltExperiment) -> dict[str, dict[str, np.ndarray]]:
    history = built.run(s(0.25), history=True).history
    assert history is not None
    return {node: {name: np.asarray(value) for name, value in fields.items()} for node, fields in history.items()}


def moved(base: dict[str, dict[str, np.ndarray]], other: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """The fields of ``base``'s nodes whose arrays ``other`` does not hold bit for bit."""
    return [f"{node}.{name}" for node, fields in base.items() for name, value in fields.items()
            if node not in other or name not in other[node] or not np.array_equal(value, other[node][name])]


def provider_of(built: BuiltExperiment) -> tuple[str, list[str], dict[str, np.ndarray]]:
    node = built.graph.get_node(FIELD)
    leaves = parameter_leaves(built.graph)[FIELD]
    return type(node).__name__, sorted(node.initial_state()), {key: np.asarray(leaves[key]) for key in sorted(leaves)}


def same_provider(first: BuiltExperiment, second: BuiltExperiment) -> bool:
    (cls, fields, leaves), (other_cls, other_fields, other_leaves) = provider_of(first), provider_of(second)
    return cls == other_cls and fields == other_fields and leaves.keys() == other_leaves.keys() and all(
        np.array_equal(leaves[key], other_leaves[key]) for key in leaves)


def test_the_observation_moves_nothing_in_the_run_and_leaves_the_provider_as_it_was(dtype):
    plain = script_a().build()
    with_lag = observed().build()
    base, seen = histories(plain), histories(with_lag)
    assert LAG in seen and LAG not in base
    assert moved(base, seen) == [], f"the observation moved the run ({counts()})"
    assert same_provider(plain, with_lag)
    unit = np.float32 if dtype == "float32" else np.float64
    nudged = float(np.nextafter(unit(0.01), unit(np.inf)))
    assert moved(base, histories(script_a(start=m((nudged, 0.0, -1e-3))).build())), \
        "the control: a start one unit in the last place off is not told apart"
    forced = script_a()
    forced.declare("umr.magnet_centre", m((0.0, 0.0, 0.0)), reason="the control's second law on the field")
    forced.attach(Magnetic.Force(), body="umr", field="field.magnetic")
    assert not same_provider(plain, forced.build()), "the control: a slot added is not told apart"


def torque_and_force(experiment: Experiment) -> Experiment:
    experiment.declare("umr.magnet_centre", m((0.0, 0.0, 0.0)), reason="the force reads the gradient at the magnet")
    experiment.attach(Magnetic.Force(), body="umr", field="field.magnetic")
    return experiment


def test_the_observation_shares_the_slot_of_the_law_that_reads_its_interface_and_the_report_names_it():
    built = torque_and_force(observed()).build()
    reads = {plan.owner: plan.reads for plan in built.plans.entries}
    force, torque, lag = reads["magnetic.force:umr"], reads[TORQUE], reads[LAG]
    assert "magnetic.force:umr" < TORQUE, "the force's id sorts first, so a share blind to the interface takes it"
    assert set(force) == {"pose", "gradient"} and "flux_density" not in force
    assert (lag["flux_density"].node, lag["flux_density"].field) == (FIELD, SLOT)
    assert (lag["flux_density"].node, lag["flux_density"].field) == (torque["flux_density"].node,
                                                                       torque["flux_density"].field)
    assert "magnetic_force" not in lag["flux_density"].field
    named = [line for line in built.report.checked[LAG] if line.startswith("reads ")]
    assert len(named) == 1 and f"where {TORQUE} reads it, {FIELD}.{SLOT}" in named[0]
    assert "magnetic.force" not in named[0]
    (lag_line,) = (line for line in built.report.checked[LAG] if line.startswith("the lag is"))
    assert "magnetic.force" not in lag_line, "a law that reads the gradient is not one that turns the body here"
    assert any(line.startswith(f"validity of {LAG} (computed at build): reads") for line in built.report.summary())


def test_control_with_only_a_law_of_another_interface_the_observation_is_refused_with_the_route():
    experiment = torque_and_force(observed())
    experiment.remove(TORQUE)
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    found = [r for r in refused.value.refusals if r.code == "observation.field-read"]
    assert len(found) == 1
    assert "no law reads field.magnetic.flux_density from it" in found[0].message
    assert "exp.attach(Magnetic.Torque()" in found[0].action
