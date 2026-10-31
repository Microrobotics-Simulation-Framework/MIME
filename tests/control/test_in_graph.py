"""Mode 1, the controller inside the compiled step: its node at its own rate and its latency's delay line, the trace
check that refuses a step that cannot run there, and the order the latency rule needs. Mode 2, the callback, is the
oracle: the same law's step, called on the host between compiled chunks.

Each property is a check that returns what is wrong, so that its control, a planted fault, can be shown to fail it:
(f) the controller's node ordered after the physics, (g) a latency off by one step, (h) a step with a host callback
accepted, (i) the controller's state not carried from one step to the next; and the review's: (j) a step that reads
Python state its calls change accepted, (k) the schedule's phase restarted at each run, (l) the lock recording a
callback with its law's status. Each is measured over 20 runs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import tracked_rpm, tracking
from control_cases import (
    IMPURE_CALLS,
    CalledBack,
    Counter,
    Follower,
    GroupedTorque,
    Growing,
    Impure,
    Ticker,
    expected_angles,
    spun_magnet,
    watched_arm,
)

import mime.build._construct as construct_module
import mime.build._in_graph as in_graph_module
import mime.control._run as run_module
import mime.saved._record as record_module
from mime.build import BuildRefused, Lag
from mime.build._maddening import node_state, step_function
from mime.experiment import Experiment
from mime.nodes.control import ControllerNode, DelayNode
from mime.testing import node_results
from mime.units import ms


def refused_codes(make) -> list[str]:
    try:
        make().build()
    except BuildRefused as caught:
        return caught.codes()
    return []


# -- the build -------------------------------------------------------------------------------------------------------


def test_inside_the_compiled_step_the_controller_runs_first_in_one_scan_and_derives_no_input():
    built = spun_magnet(controller=Counter(), mode="graph", rate=ms(3), latency=ms(2)).build()
    assert built.graph.schedule[:2] == ["controller", "controller.delay"]
    assert built.report.external_inputs == () and built.report.run.chunk is None
    assert built.graph.rate_dividers["controller"] == 3 and built.graph.rate_dividers["controller.delay"] == 1
    lines = {line.node: line for line in built.report.nodes}
    assert (lines["controller"].divider, lines["controller.delay"].divider) == (3, 1)
    line = built.report.controller[0]
    assert line.startswith("controller: oracle.counter 0.1.0, mode 1 (inside the compiled step, as a node at its own "
                           "rate), every 3 steps (0.003 s), latency 2 steps (0.002 s)")
    assert "its step runs inside the compiled step, so a derivative of the run passes through its writes" in \
        built.report.text(details=True)
    lags = {(edge.source[0], edge.target[0]): edge.lag for edge in built.report.lags}
    assert lags[("umr", "controller")] is Lag.PREVIOUS_STEP
    assert lags[("controller", "controller.delay")] is lags[("controller.delay", "spin.motor")] is Lag.THIS_STEP


def test_without_a_latency_the_controller_writes_its_commands_directly():
    built = spun_magnet(controller=Counter(), mode=1, rate=ms(1)).build()
    assert "controller.delay" not in built.graph.node_names
    assert not built.graph.is_multirate
    edges = {(edge.source[0], edge.target[0]): edge.lag for edge in built.report.lags}
    assert edges[("controller", "spin.motor")] is Lag.THIS_STEP


# -- the latency rule, against mode 2 -------------------------------------------------------------------------------


def histories(period: int, latency: int, steps: int, law=Counter, **rig) -> tuple[dict, dict]:
    """The same law's run in mode 1 and mode 2: the histories of every node, by node and field."""
    found = []
    for mode in ("graph", "callback"):
        built = spun_magnet(controller=law(), mode=mode, rate=ms(period), latency=ms(latency) if latency else None,
                            **rig)
        history = built.build().run(ms(steps), history=True).history
        assert history is not None
        found.append(history)
    return found[0], found[1]


#: The single-precision units in the last place per step run that a field of a law that does arithmetic on its reads
#: may differ by between the modes: XLA compiles the same arithmetic into two different steps and may fuse a multiply
#: into an add in one and not the other. The A0 comparison's bound, stated for this grid before it ran.
ULPS_PER_STEP = 8


def latency_failures(period: int, latency: int, law=Counter, **rig) -> list[str]:
    """Every field of every node mode 2 has that mode 1's run does not give as mode 2's: ``Counter`` (integer-valued
    writes, a law that ignores its reads) bit for bit, and its writes in force (the motor's angle) the rule's; any
    other law, such as ``Follower``, which reads the pose and does arithmetic, within ``ULPS_PER_STEP`` units in the
    last place of the field's largest magnitude per step run. The run covers the latency and three more periods, so
    that at least three writes act."""
    steps = period * (latency // period + 4)
    graph, callback = histories(period, latency, steps, law, **rig)
    eps = float(np.finfo(np.float32).eps)
    failures = []
    for node, fields in callback.items():
        for name, oracle in fields.items():
            found = graph[node][name]
            if law is Counter:
                ok = np.array_equal(found, oracle)
            else:
                ok = float(np.max(np.abs(found - oracle))) <= steps * ULPS_PER_STEP * eps * float(np.max(np.abs(oracle)))
            if not ok:
                failures.append(f"period {period}, latency {latency}: {node}.{name} is {found.tolist()} in mode 1 and "
                                f"{oracle.tolist()} in mode 2")
    if law is Counter:
        rule = expected_angles(steps, latency, period)
        if [float(a) for a in graph["spin.motor"]["angle"]] != rule:
            failures.append(f"period {period}, latency {latency}: mode 1 wrote "
                            f"{graph['spin.motor']['angle'].tolist()}, the rule {rule}")
    return failures


#: The 28 cases of 7b1's grid, and the four more the review of 7b2 ran off it (one, (1, 1), is on it).
GRID = [(period, latency) for period in (1, 2, 3, 5) for latency in (0, 1, 2, 4, 5, 9, 11)]
OFF_GRID = [(7, 3), (3, 3), (4, 12), (1, 1)]


@pytest.mark.parametrize(("period", "latency"), [(3, 2), (1, 0), (2, 5)])
def test_a_controller_inside_the_compiled_step_reads_a_coupling_groups_output_as_a_callback_does(period, latency):
    """The rig of the grid with its body's dynamics and its torque solved together in a coupling group, and the
    controller (``Follower``) reading the body's pose, a member's output: mode 1 against mode 2 over every field of
    every node, within the grid's bound. Neither mode's build is refused, and the group is in both graphs."""
    for mode in ("graph", "callback"):
        built = spun_magnet(controller=Follower(), mode=mode, rate=ms(period), latency=ms(latency) if latency else None,
                            torque=GroupedTorque()).build()
        assert [set(group.nodes) for group in built.report.groups] == [{"oracle.grouped_torque:umr", "umr"}]
    assert latency_failures(period, latency, Follower, torque=GroupedTorque()) == []


def test_control_f_on_the_rig_with_a_coupling_group_the_controller_ordered_after_the_physics_is_refused(monkeypatch):
    """Control (f) again, with the group in the graph: the order check still refuses it."""
    monkeypatch.setattr(construct_module, "in_graph_order", lambda mine, physics, edges, groups: [
        *in_graph_module.scheduled(physics, edges, groups), *mine])
    codes = refused_codes(lambda: spun_magnet(controller=Follower(), mode="graph", latency=ms(2),
                                              torque=GroupedTorque()))
    assert "post-build.controller-order" in codes


@pytest.mark.parametrize("law", [Counter, Follower], ids=["counter", "follower"])
@pytest.mark.parametrize(("period", "latency"), [*GRID, *(case for case in OFF_GRID if case not in GRID)])
def test_mode_1_writes_what_mode_2_writes_at_the_same_steps(period, latency, law):
    """7b1's grid, the rule mode 2 is held to there: a period of one step (A0's), latencies longer than the period
    (several writes in flight), equal to it and not a multiple of it, and periods and latencies off the grid. The
    same law runs in both modes. ``Counter`` writes whole numbers and ignores its reads: its whole state is compared
    bit for bit, and its writes against the rule. ``Follower`` reads the pose and filters it, in single precision:
    every field of every node within ``ULPS_PER_STEP``, which pins the reads' timing as well as the writes'."""
    assert latency_failures(period, latency, law) == []


@pytest.mark.parametrize("law", [Counter, Follower], ids=["counter", "follower"])
def test_control_g_a_latency_off_by_one_step_inside_the_compiled_step_is_caught(monkeypatch, law):
    """Control (g): the delay line holds the writes one step longer than declared, caught with a law that ignores its
    reads and with one that reads them (the follower's tolerance is far below one step's shift)."""
    assert latency_off_by_one(monkeypatch, 1, law) != []


def test_control_g_a_latency_one_step_short_is_caught(monkeypatch):
    assert latency_off_by_one(monkeypatch, -1, Follower) != []


def test_control_g_a_latency_off_by_one_step_is_caught_on_the_rig_with_a_coupling_group(monkeypatch):
    assert latency_off_by_one(monkeypatch, 1, Follower, torque=GroupedTorque()) != []


def latency_off_by_one(monkeypatch, shift: int, law, **rig) -> list[str]:
    def off(name, timestep, *, latency, **given):
        return DelayNode(name, timestep, latency=latency + shift, **given)

    monkeypatch.setattr(in_graph_module, "DelayNode", off)
    return latency_failures(3, 2, law, **rig)


def test_control_i_a_state_not_carried_from_step_to_step_is_caught(monkeypatch):
    """Control (i): the controller's node steps its law from its starting state every time, so the counter never
    counts past its first tick."""
    assert state_not_carried(monkeypatch) != []


def state_not_carried(monkeypatch) -> list[str]:
    held = ControllerNode.advance

    def forgetting(self, p, state, boundary_inputs, dt):
        start = {key: value for key, value in self.initial_state().items() if key.startswith("state_")}
        return held(self, p, {**state, **start}, boundary_inputs, dt)

    monkeypatch.setattr(ControllerNode, "advance", forgetting)
    return latency_failures(1, 0)


def test_runs_continue_where_the_last_left_its_controller_inside_the_compiled_step():
    """Two runs of 9 ms with a latency of 4 steps give the 18 steps of one: the graph carries the controller's state
    and its delay line, and the multi-rate schedule its phase."""
    one = spun_magnet(controller=Counter(), mode="graph", latency=ms(4)).build().run(ms(18), history=True).history
    built = spun_magnet(controller=Counter(), mode="graph", latency=ms(4)).build()
    first, second = (built.run(ms(9), history=True).history for _ in range(2))
    assert one is not None and first is not None and second is not None
    joined = np.concatenate([first["spin.motor"]["angle"], second["spin.motor"]["angle"]])
    assert np.array_equal(joined, one["spin.motor"]["angle"])
    assert [float(a) for a in joined] == expected_angles(18, 4, 3)


#: The review's splits of a 15-step run, end to end: 4 + 11, 1 + 14, 7 + 8 and 2 + 2 + 11, none of them on a period
#: of 3 or 5 steps but the ends, run one after another on one graph (60 steps), against one run of 60 steps.
SPLITS = (4, 11, 1, 14, 7, 8, 2, 2, 11)


def joined_run(period: int, latency: int, law) -> tuple[dict, dict]:
    """A run of ``sum(SPLITS)`` steps with the controller inside the compiled step, whole and in runs of ``SPLITS``
    steps joined."""
    def make():
        return spun_magnet(controller=law(), mode="graph", rate=ms(period),
                           latency=ms(latency) if latency else None).build()

    whole = make().run(ms(sum(SPLITS)), history=True).history
    built = make()
    parts = [built.run(ms(steps), history=True).history for steps in SPLITS]
    assert whole is not None and all(part is not None for part in parts)
    return whole, {node: {name: np.concatenate([part[node][name] for part in parts]) for name in fields}
                   for node, fields in whole.items()}


@pytest.mark.parametrize("law", [Counter, Follower], ids=["counter", "follower"])
@pytest.mark.parametrize(("period", "latency"), [(3, 2), (3, 0), (1, 0), (5, 4)])
def test_runs_that_are_not_whole_periods_rejoin_bit_for_bit_with_the_controller_inside(period, latency, law):
    """Mode 1 accepts a run of any number of steps (mode 2 refuses one that is not whole periods, below): the
    multi-rate schedule's phase and the controller's state persist across runs, so runs split off the period (4 + 11
    with a period of 3) rejoin the whole run, every field of every node, bit for bit."""
    whole, joined = joined_run(period, latency, law)
    assert [(n, f) for n in whole for f in whole[n] if not np.array_equal(whole[n][f], joined[n][f])] == []


def split_failures(monkeypatch=None) -> list[str]:
    """The fields of a period-3 run in ``SPLITS`` that do not rejoin the whole run, with the schedule's phase
    restarted at each run if ``monkeypatch`` is given."""
    if monkeypatch is not None:
        held = run_module.scan_chunk_with_history

        def restarting(graph, steps, fed):
            found = held(graph, steps, fed)
            graph._state["_meta"]["step_count"] = jnp.zeros_like(graph._state["_meta"]["step_count"])
            return found

        monkeypatch.setattr(run_module, "scan_chunk_with_history", restarting)
    whole, joined = joined_run(3, 2, Counter)
    return [f"{n}.{f}" for n in whole for f in whole[n] if not np.array_equal(whole[n][f], joined[n][f])]


def test_control_k_a_schedule_phase_restarted_at_each_run_is_caught(monkeypatch):
    """Control (k): the multi-rate schedule's phase restarts at each run, so runs split off the period no longer
    rejoin the whole: the comparison finds it."""
    assert split_failures(monkeypatch) != []


def test_a_run_that_is_not_whole_periods_runs_inside_the_compiled_step_and_is_refused_as_a_callback():
    """The difference N-1 of the review names, for the swap test (7c2) to know: a run of 7 steps with a period of 3
    is the first 7 steps of a longer run in mode 1, and a ``ValueError`` in mode 2."""
    seven = spun_magnet(controller=Counter(), mode="graph", rate=ms(3)).build().run(ms(7), history=True).history
    fifteen = spun_magnet(controller=Counter(), mode="graph", rate=ms(3)).build().run(ms(15), history=True).history
    assert seven is not None and fifteen is not None
    assert np.array_equal(seven["spin.motor"]["angle"], fifteen["spin.motor"]["angle"][:7])
    with pytest.raises(ValueError):
        spun_magnet(controller=Counter(), mode="callback", rate=ms(3)).build().run(ms(7))


def test_a_hook_chunks_a_run_with_the_controller_inside_and_changes_nothing_it_computes():
    whole = spun_magnet(controller=Counter(), mode="graph", latency=ms(2)).build().run(ms(12), history=True).history
    seen = []
    chunked = spun_magnet(controller=Counter(), mode="graph", latency=ms(2)).build().run(
        ms(12), history=True, hook=seen.append, chunk=ms(4)).history
    assert whole is not None and chunked is not None and [chunk.steps for chunk in seen] == [4, 8, 12]
    assert all(np.array_equal(whole[n][f], chunked[n][f]) for n in whole for f in whole[n])


def test_a_controller_inside_the_compiled_step_is_saved_loaded_and_verified(tmp_path):
    """A0's rig with its controller inside the compiled step: the document holds the mode by word; a load rebuilds the
    controller's node and delay line from it, with the same derived hash, and its reference steps rerun bit for bit."""
    exp = tracked_rpm("saved_in_graph", mode="graph")
    assert exp.document()["controller"]["mode"] == "graph"
    built = exp.build()
    built.save(tmp_path / "saved", steps=9)
    rebuilt = Experiment.load(tmp_path / "saved").build()
    assert rebuilt.report.derived_hash == built.report.derived_hash
    assert rebuilt.verify().reproduced


# -- the order -------------------------------------------------------------------------------------------------------


def order_codes(monkeypatch=None, planted=None) -> list[str]:
    if planted is not None and monkeypatch is not None:
        monkeypatch.setattr(construct_module, "in_graph_order", planted)
    return refused_codes(lambda: spun_magnet(controller=Counter(), mode="graph", latency=ms(2)))


def test_every_other_edge_reads_the_step_it_reads_with_the_controller_called_between_chunks():
    """Swapping the controller's mode changes no physics: each edge that is not the controller's reads the same step
    in both modes, on A0's rig and on the motor's."""
    for make in (lambda mode: tracked_rpm(mode=mode, rate=ms(1), latency=None),
                 lambda mode: spun_magnet(controller=Counter() if mode == "graph" else Ticker(), mode=mode,
                                          latency=ms(2))):
        lags = [{(e.source, e.target): e.lag for e in make(mode).build().report.lags} for mode in ("graph",
                                                                                                    "callback")]
        assert {key: lag for key, lag in lags[0].items() if "controller" not in key[0][0] + key[1][0]} == lags[1]


def test_control_f_the_controllers_node_ordered_after_the_physics_is_refused(monkeypatch):
    """Control (f): the controller's nodes added after the rest. Its writes would act a step late and its reads see
    the end of the step; the order check refuses the build."""
    assert "post-build.controller-order" in order_codes(
        monkeypatch, lambda mine, physics, edges, groups: [*in_graph_module.scheduled(physics, edges, groups), *mine])


def test_the_canonical_order_alone_would_read_a_commanded_segment_a_step_late(monkeypatch):
    """With the controller first and the rest in the canonical order, the motor, which the controller commands, runs
    after the field it carries, so the field reads its pose a step late, as it does not with the controller called
    between chunks: refused."""
    codes = order_codes(monkeypatch, lambda mine, physics, edges, groups: [*mine, *physics])
    assert codes == ["post-build.controller-order"]


def test_the_order_holds_and_builds():
    assert order_codes() == []


def test_a_read_of_a_node_the_controllers_writes_do_not_reach_is_refused():
    """The arm is driven, and the controller writes the motor it carries: the arm runs before the controller in every
    step, so the controller would read its state at the end of the step. Refused inside the compiled step; as a
    callback it builds."""
    with pytest.raises(BuildRefused) as caught:
        watched_arm(mode="graph").build()
    assert caught.value.codes() == ["controller.unreached-read"]
    assert "reads rpm.arm.joint_positions, which nothing it writes reaches" in str(caught.value)
    watched_arm(mode="callback").build()


# -- the trace check -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("controller", "said"), [
    (CalledBack(), "its step calls the host (debug_callback)"),
    (Growing(), "its state comes back as 'poses' float32[1, 7], and the state it starts from is float32[0, 7]"),
])
def test_a_step_that_cannot_run_inside_the_compiled_step_is_refused_for_it(controller, said):
    """Python on a traced value, a call to the host, a state that changes shape: each is refused for mode 1, naming
    why, and runs as a callback."""
    assert untraceable_refusal(controller, said) == []
    spun_magnet(controller=controller, mode="callback").build()


def test_a_step_that_asks_a_traced_value_for_a_python_one_is_refused_at_its_line_and_runs_as_a_callback():
    """Python on a traced value (the ticker's ``int()`` of its count): refused for mode 1 as
    ``control.law-traced-branch``, naming the step's line and ``jnp.where``, and run as a callback."""
    with pytest.raises(BuildRefused) as caught:
        spun_magnet(controller=Ticker(), mode="graph").build()
    (held,) = caught.value.refusals
    assert held.code == "control.law-traced-branch" and "TracerIntegerConversionError" in held.message
    assert held.where is not None and "control_cases.py" in held.where and "jnp.where" in held.action
    spun_magnet(controller=Ticker(), mode="callback").build()


def untraceable_refusal(controller, said: str) -> list[str]:
    try:
        spun_magnet(controller=controller, mode="graph").build()
    except BuildRefused as caught:
        if caught.codes() != ["controller.untraceable"]:
            return [f"refused with {caught.codes()}, not controller.untraceable"]
        message = caught.refusals[0].message
        return [] if said in message and "mode='callback'" in caught.refusals[0].action else [message]
    return ["built"]


def test_control_h_a_step_with_a_host_callback_accepted_for_mode_1_is_caught(monkeypatch):
    """Control (h): the trace check finds no host callback; the build is no longer refused as untraceable."""
    assert host_callback_accepted(monkeypatch) != []


def host_callback_accepted(monkeypatch) -> list[str]:
    monkeypatch.setattr(in_graph_module, "host_callbacks", lambda jaxpr: set())
    return untraceable_refusal(CalledBack(), "its step calls the host (debug_callback)")


def impure_refusal() -> list[str]:
    """What is wrong with the build of a step that counts its calls in a global, for mode 1: it must be refused as
    ``controller.impure``, saying why and offering the callback."""
    try:
        spun_magnet(controller=Impure(), mode="graph").build()
    except BuildRefused as caught:
        if caught.codes() != ["controller.impure"]:
            return [f"refused with {caught.codes()}, not controller.impure"]
        refusal = caught.refusals[0]
        said = "reads Python state that its own calls change" in refusal.message
        return [] if said and "mode='callback'" in refusal.action else [refusal.message]
    return ["built"]


def test_a_step_that_reads_state_its_own_calls_change_is_refused_for_mode_1_and_runs_as_a_callback():
    """The review's counter: a global incremented in the step. Traced, it freezes at the number of traces, so mode 1
    would write one number at every tick where mode 2 writes 1, 2, 3, ...: refused for mode 1, with the callback as
    its action. Every run, the refusal holds whatever the global has counted before."""
    for held in (0, 5):
        IMPURE_CALLS[0] = held
        assert impure_refusal() == []
    IMPURE_CALLS[0] = 0
    history = spun_magnet(controller=Impure(), mode="callback", rate=ms(1)).build().run(ms(4), history=True).history
    assert history is not None and [float(a) for a in history["spin.motor"]["angle"]] == [1.0, 2.0, 3.0, 4.0]


def test_control_j_a_step_that_changes_what_it_reads_between_calls_accepted_for_mode_1_is_caught(monkeypatch):
    """Control (j): the purity check finds nothing; the counter's step builds for mode 1, which freezes it."""
    monkeypatch.setattr(in_graph_module, "impure_problems", lambda step, state, reads: [])
    IMPURE_CALLS[0] = 0
    assert impure_refusal() == ["built"]


def test_no_step_of_ours_is_refused_as_impure():
    """The steps the tests run inside the compiled step trace to one program every time (no false refusal); A0's
    ``arm_tracking`` is built for mode 1 by the comparisons below, which would fail if it were refused."""
    for controller in (Counter(), Follower()):
        spun_magnet(controller=controller, mode="graph").build()


# -- the nodes -------------------------------------------------------------------------------------------------------


def test_the_controllers_node_and_delay_line_pass_maddenings_battery():
    built = spun_magnet(controller=Follower(), mode="graph", rate=ms(3), latency=ms(2)).build()
    for name in ("controller", "controller.delay"):
        failed = [str(result) for result in node_results(built.graph.get_node(name)).values() if not result.passed]
        assert failed == [], name


# -- gradients -------------------------------------------------------------------------------------------------------


def angle_after(built, steps: int, omega, fed) -> jax.Array:
    """The motor's angle after ``steps`` of ``built``'s graph from its initial state, the body started turning about
    the tube's axis at ``omega``, traced through the compiled step with ``fed`` as its external inputs."""
    graph = built.graph
    step = step_function(graph)
    state = {name: dict(node_state(graph, name)) for name in graph.node_names}
    state["umr"]["twist"] = state["umr"]["twist"].at[3].set(omega)

    def body(held, _):
        return step(held, fed), None

    final, _ = jax.lax.scan(body, state, None, length=steps)
    return final["spin.motor"]["angle"]


def test_a_derivative_of_the_run_passes_through_the_controller_inside_the_compiled_step_and_not_through_a_callback():
    """Float64. Inside the compiled step, the derivative of the motor's angle after 20 steps with respect to the
    body's starting turn about the tube's axis, through the closed loop, equals a central difference of the same run
    to 1e-6 relative (a step of 1e-6 rad/s: the difference's truncation is of order the step squared, its rounding of
    order 1e-10 of the derivative). As a callback, the compiled chunk takes the controller's writes as fixed inputs,
    and the same derivative is zero: a derivative of the run stops at its writes."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        inside = spun_magnet(controller=Follower(), mode="graph", rate=ms(1)).build()
        omega, h = 2.0, 1e-6
        derivative = float(jax.grad(lambda w: angle_after(inside, 20, w, {}))(jnp.asarray(omega)))
        central = float(angle_after(inside, 20, omega + h, {}) - angle_after(inside, 20, omega - h, {})) / (2 * h)
        assert derivative != 0.0 and abs(derivative - central) <= 1e-6 * abs(central), (derivative, central)
        between = spun_magnet(controller=Follower(), mode="callback", rate=ms(20)).build()
        fed = between.held_inputs()
        assert float(jax.grad(lambda w: angle_after(between, 20, w, fed))(jnp.asarray(omega))) == 0.0
    finally:
        jax.config.update("jax_enable_x64", held)


def test_a0s_controllers_writes_have_a_derivative_through_its_node():
    """A0's controller, unfiltered, inside the compiled step: the derivative of the joint positions it writes with
    respect to the body's distance along the tube, through its node, is finite, and equals a central difference of
    1 mm to 1% (it computes in single precision: the difference's rounding is about 1e-4 of it)."""
    built = tracked_rpm(controller=tracking(smoothing=1.0), mode="graph", rate=ms(1), latency=None).build()
    node = built.graph.get_node("controller")
    state = node_state(built.graph, "controller")
    given = {f"read_{label}": node_state(built.graph, source)[field] for label, source, field in
             (("body_pose", "umr", "pose"), ("body_twist", "umr", "twist"),
              ("joint_positions", "rpm.arm", "joint_positions"))}
    given["read_body_pose"] = given["read_body_pose"].at[0].set(0.02)

    def written(x):
        return node.update(state, {**given, "read_body_pose": given["read_body_pose"].at[0].set(x)},
                           built.report.dt)["write_joint_positions"]

    dtype = given["read_body_pose"].dtype
    # One compiled function gives the writes and their derivative, at each point the difference reads.
    tangent = jax.jit(lambda x: jax.jvp(written, (x,), (jnp.ones((), dtype),)))
    derivative = np.asarray(tangent(jnp.asarray(0.02, dtype))[1])
    central = (np.asarray(tangent(jnp.asarray(0.021, dtype))[0]) -
               np.asarray(tangent(jnp.asarray(0.019, dtype))[0])) / 2e-3
    assert np.all(np.isfinite(derivative)) and np.max(np.abs(derivative)) > 0
    assert np.max(np.abs(derivative - central)) <= 1e-2 * np.max(np.abs(central)), (derivative, central)


# -- the status per mode ---------------------------------------------------------------------------------------------


def recorded_statuses() -> dict[str, str]:
    """The status the lock's resolved record holds for the follower (``exact``) in each mode."""
    return {mode: record_module.resolved_record(spun_magnet("lock_status", controller=Follower(), mode=mode).plan())[
        "entries"]["controller"]["differentiability"] for mode in ("graph", "callback")}


def test_the_lock_records_the_controllers_status_in_the_mode_it_is_placed_in():
    """Mode 1 records the law's own status (the follower's ``exact``; ``arm_tracking``'s ``piecewise_jump`` is in the
    export) and mode 2 the binding's ``cut``, as the lock records each rung's status for an entry with rungs (#176)."""
    assert recorded_statuses() == {"graph": "exact", "callback": "cut"}


def test_control_l_a_lock_that_records_the_laws_status_for_a_callback_is_caught(monkeypatch):
    """Control (l): the lock ignores the mode, so a callback is recorded with its law's status."""
    monkeypatch.setattr(record_module, "_called_back", lambda plans: None)
    assert recorded_statuses()["callback"] != "cut"


# -- A0's rig, mode 1 against mode 2 ----------------------------------------------------------------------------------

#: The steps of the recorded scenario, and the single-precision units in the last place each field may move by a step
#: between the modes: XLA compiles the same arithmetic into two different steps, and may fuse a multiply into an add
#: in one and not the other, which moves a result by a rounding; carried through a stable closed loop, such moves add
#: up at most linearly. A growth faster than that is a difference in what is computed, not in how it rounds.
A0_STEPS = 200
ULPS_PER_STEP = 8


@pytest.mark.parametrize("feedback", [False, True], ids=["feedback-off", "feedback-on"])
def test_a0s_rig_runs_alike_with_its_controller_inside_the_compiled_step_and_as_a_callback(feedback):
    """A0's controller on the kinematic arm, every step (A0's rate) with no latency (the 0.2 runner's), its filter at
    A0's 0.005 a tick, the body started 2 cm along the tube so that the arm follows it: 200 steps in each mode. Every
    field of every node, the controller's writes (the arm's joint positions) among them, agrees between the modes
    within ``ULPS_PER_STEP`` single-precision units in the last place of the field's largest magnitude, per step run.
    The tolerance was stated before the comparison ran. With the feedback on, the controller also reads the body's
    twist, end to end in both modes."""
    found = []
    for mode in ("graph", "callback"):
        exp = tracked_rpm(controller=tracking(orientation_feedback=feedback), mode=mode, rate=ms(1), latency=None)
        for declaration in exp._declarations:
            thing = getattr(declaration, "thing", None)
            if getattr(thing, "name", None) == "umr":
                from mime.things import State
                from mime.units import mm

                object.__setattr__(thing, "initial", State(**{**dict(thing.initial),
                                                              "position": mm((20.0, 0.0, -1.0))}))
        history = exp.build().run(ms(A0_STEPS), history=True).history
        assert history is not None
        found.append(history)
    inside, called = found
    eps = float(np.finfo(np.float32).eps)
    failures = []
    for node, fields in called.items():
        for name, oracle in fields.items():
            moved = float(np.max(np.abs(inside[node][name] - oracle)))
            bound = A0_STEPS * ULPS_PER_STEP * eps * float(np.max(np.abs(oracle)))
            print(f"{node}.{name}: {moved:.3g} (bound {bound:.3g})")
            if not moved <= bound:
                failures.append(f"{node}.{name} moved {moved:.3g}, over {bound:.3g}")
    assert failures == []
