"""Ports through the build: what a declared port builds, where, and what it leaves alone.

- **No port, no change** (design section 9, control 1): an apparatus with no port, with ``ports={}``, and with an
  all-default ``CommandPort()`` on a command and an ideal ``Measurement`` (every value its source makes, at once, no
  noise or quantum, its age not read) builds the same program, its derived hash and canonical text the same, on the
  coil-array sphere and on A0's apparatus. The controls: a stage built for an ideal command port (``stage_for_ideal``),
  and one built for an ideal measurement (``measure_for_ideal``), must each change both.
- **No two ports hold one field**: a port named as another's age (``tr`` and ``tr_age``) is refused; the control
  (``clash_unchecked``, the check removed) builds, and the second overwrites the first's age in silence.
- **One lag, one place**: a ``FirstOrderLag`` on a kinematic arm's position commands is handed to the arm's node (its
  servo, ``(computed by its receiver)``), and builds no stage; the stage computes the lag for a command whose receiver
  does not (a coil array's currents). A lag on some of an arm's position commands and not others, or two lags, is
  refused: its node lags every joint with one time constant. A kinematic arm's port takes its joints' stated limits as
  its bounds, before its slew.
- **The arm's order** (design section 4): a port with a latency, a slew and a lag on a kinematic arm, written by a
  controller inside the compiled step with a drawn square wave that leaves the joints' limits and comes back, delivers
  the design's order: sample, delay, the joints' stated limits as its bounds, slew, then the arm's lag. The controls
  (``PORT_PLANT``): the stage slewing before it bounds (``slew_before_bound``), and the stage leaving the joints' limits
  to the arm's own clip after its slew and lag (``limits_after_slew``), each caught on every run.
- **Controllers through ports**: a controller called on the host and one inside the compiled step, each reading a
  measurement port and writing through a command port, run the same steps to the same numbers; their reads are
  classed ``measurement``, the age is read by its path.
- **Each refusal**, with a near miss that builds."""

from __future__ import annotations

import numpy as np
import pytest
from chain_cases import kinematic_rpm
from port_cases import (C_ARM_LIMITS, SEED, Link, PoseFeedback, coil_sphere, command_oracle, fed_back_sphere, planted,
                        replayed_arm, square_wave)

from mime import CommandPort, Measurement
from mime.build import BuildRefused, _ports
from mime.nodes.control import CommandStageNode
from mime.responses import FirstOrderLag
from mime.testing._graph import graph_snapshot
from mime.things import Gaussian
from mime.units import A, m, mm, ms, rad, s

DT = 1e-3


def sphere(port: object | None = None, *, ports: dict | None = None):
    """The coil-array sphere, its currents through ``port`` (or the apparatus's ``ports``)."""
    return coil_sphere(ports if ports is not None else ({} if port is None else {"currents": port}))


def codes(error: BuildRefused) -> list[str]:
    return [refusal.code for refusal in error.refusals]


@pytest.fixture
def ideal_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control's plant: every command port staged, an ideal one included."""
    if planted("stage_for_ideal"):
        monkeypatch.setattr(_ports.CommandPortPlan, "staged", property(lambda self: True))
    if planted("measure_for_ideal"):
        monkeypatch.setattr(_ports.MeasurementPlan, "ideal", property(lambda self: False))


@pytest.mark.parametrize(("make", "command", "source"), [
    (lambda ports: sphere(ports=ports), "currents", "sphere.pose"),
    (lambda ports: kinematic_rpm("ported_rpm", ports=ports), "arm.joint_positions", "umr.pose"),
], ids=["sphere", "a0_apparatus"])
def test_no_port_an_empty_one_and_an_all_default_one_build_the_program_of_none(make, command, source, ideal_stages):
    ideal = {command: CommandPort(), "tracker": Measurement(of=source)}
    none, empty, default = (make(ports).build() for ports in (None, {}, ideal))
    assert none.report.derived_hash == empty.report.derived_hash == default.report.derived_hash
    assert graph_snapshot(none) == graph_snapshot(empty) == graph_snapshot(default)
    assert not [name for name in default.graph.node_names if name.startswith(("ports:", "measure:"))]


@pytest.fixture
def unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    if planted("clash_unchecked"):
        monkeypatch.setattr("mime.build._stages.clashes", lambda node: {})


def test_a_port_named_as_another_ports_age_is_refused(unchecked):
    clashing = {"tr": Measurement(of="sphere.pose", latency=ms(1), reason="a tracker"),
                "tr_age": Measurement(of="sphere.pose", latency=ms(2), reason="another tracker")}
    with pytest.raises(BuildRefused) as refused:
        sphere(ports=clashing).build()
    assert codes(refused.value) == ["port.declaration"] and "'tr_age'" in str(refused.value)
    sphere(ports={"tr": clashing["tr"], "second": clashing["tr_age"]}).build()


def test_a_rest_nothing_delivers_or_outside_its_bounds_is_refused():
    with pytest.raises(BuildRefused) as refused:
        sphere(CommandPort(rest=A(1.0), reason="r")).build()
    assert codes(refused.value) == ["port.declaration"] and "which nothing it declares delivers" in str(refused.value)
    with pytest.raises(BuildRefused) as refused:
        sphere(CommandPort(bounds=A(1.0), on_bound="trip", rest=A(5.0), reason="r")).build()
    assert codes(refused.value) == ["port.declaration"] and "outside its bounds" in str(refused.value)
    sphere(CommandPort(bounds=A(5.0), on_bound="trip", rest=A(1.0), reason="r")).build()
    sphere(CommandPort(latency=ms(1), rest=A(1.0), reason="r")).build()


def test_a_measurement_ports_noise_seed_is_in_the_reports_seeds():
    built = sphere(ports={"tracker": Measurement(of="sphere.pose", latency=ms(1), reason="r"),
                          }).build()
    assert "measure:coils" not in built.report.seeds
    from mime.observations import SwimSpeed
    from mime.scene import X
    from mime.things import Gaussian as Noise
    from mime.units import m_per_s

    exp = sphere(ports={"speed": Measurement(of="swim_speed:sphere.speed", noise=Noise(sigma=m_per_s(1e-6)),
                                             reason="r")})
    exp.observe(SwimSpeed("sphere", along=X))
    seeds = exp.build().report.seeds
    assert set(seeds["measure:coils"]) == {"speed"} and seeds["measure:coils"]["speed"] > 0


def lag(tau_ms: float = 20.0) -> CommandPort:
    return CommandPort(response=FirstOrderLag(tau=ms(tau_ms)), reason="the arm's servo, its time constant")


def test_a_lag_on_a_kinematic_arms_position_commands_is_handed_to_its_node():
    built = replayed_arm(np.zeros((4, 2)), lag(), slide=lag()).build()
    assert not [name for name in built.graph.node_names if name.startswith("ports:")]
    assert all(line.endswith("(computed by its receiver)") for line in built.report.ports)
    assert built.graph.get_node("carm.arm").constants()["servo_time_constant"] == pytest.approx(0.02)


def test_a_lag_on_some_of_an_arms_commands_or_two_lags_are_refused():
    with pytest.raises(BuildRefused) as refused:
        replayed_arm(np.zeros((4, 2)), lag()).build()
    assert codes(refused.value) == ["port.declaration"] and "none on joint_displacements" in str(refused.value)
    with pytest.raises(BuildRefused) as refused:
        replayed_arm(np.zeros((4, 2)), lag(20.0), slide=lag(10.0)).build()
    assert codes(refused.value) == ["port.declaration"] and "different responses" in str(refused.value)


def test_a_lag_on_coil_currents_is_computed_by_the_stage():
    built = sphere(CommandPort(response=FirstOrderLag(tau=ms(7.5)), reason="the coils' L/R, measured")).build()
    node = built.graph.get_node("ports:coils")
    assert isinstance(node, CommandStageNode)
    assert node.constants()["responses"]["currents"] == ("response.first_order_lag", "0.1.0")
    assert "response.first_order_lag 0.1.0" in "\n".join(built.report.ports)


def drawn_arm_port() -> tuple[np.ndarray, Link, CommandPort, CommandPort]:
    """A drawn square wave for the arm's two revolute joints, 2 to 3 times past their limits and back on half periods of
    6 to 10 steps, and a port with a latency of 1 to 3 steps, a slew moving 30 % to 80 % of the smaller limit a step,
    and a lag of 1 to 8 ms (the slide's port lags it alike): within each half period the slewed command passes the
    limits, so bounding after the slew, or leaving the limits to the arm, delivers another value at every draw."""
    rng = np.random.default_rng(SEED)
    table = square_wave(rng, 40, 2, beyond=1.0, halves=(6, 11), past=(2.0, 3.0)) * \
        np.array([limit[1] for limit in C_ARM_LIMITS])
    latency, slew, tau = int(rng.integers(1, 4)), float(rng.uniform(0.3, 0.8)) * 2.0 / DT, \
        float(rng.uniform(1e-3, 8e-3))
    link = Link(latency=latency, slew=slew, tau=tau, rest=0.0)
    port = CommandPort(latency=ms(latency), slew=rad(slew) / s(1.0), response=FirstOrderLag(tau=s(tau)),
                       reason="a drawn link, slew and servo")
    return table, link, port, CommandPort(response=FirstOrderLag(tau=s(tau)), reason="the slide's servo")


def test_an_arm_port_applies_its_latency_bounds_slew_and_lag_in_the_designs_order(monkeypatch):
    table, link, port, slide = drawn_arm_port()
    if planted("slew_before_bound"):
        monkeypatch.setattr(CommandStageNode, "RUNGS", ("delay", "slew", "bound", "respond"))
    if planted("limits_after_slew"):
        monkeypatch.setattr("mime.dynamics.apparatus._arm._stated_limits", lambda articulation, joints: None)
    built = replayed_arm(table, port, slide=slide).build()
    history = built.run(ms(len(table)), history=True).history
    revolute = np.asarray(history["carm.arm"]["joint_positions"], dtype=np.float64)
    low = np.array([limit[0] for limit in C_ARM_LIMITS])
    high = np.array([limit[1] for limit in C_ARM_LIMITS])
    # The controller's tick at a step reads the start of it and writes within it (mode 1): the row the arm takes at
    # step n is the table's n-th. The rest is the arm's start, its revolute joints at zero.
    delivered = command_oracle(table, Link(latency=link.latency), DT, order=("sample", "delay"))
    expected = command_oracle(np.clip(delivered, low, high), Link(slew=link.slew, tau=link.tau), DT,
                              order=("slew", "response"))
    scale = float(np.max(np.abs(table)))
    worst = float(np.max(np.abs(revolute[:, -2:] - expected)))
    assert worst <= 64 * float(np.finfo(revolute.dtype if revolute.dtype != np.float64 else np.float32).eps) * \
        scale * len(table), worst


def test_a_staged_port_on_a_kinematic_arm_takes_its_joints_stated_limits_as_its_bounds():
    built = kinematic_rpm("bounded_rpm", ports={"arm.joint_positions": CommandPort(
        latency=ms(1), reason="a test link, one step")}).build()
    (plan,) = built.plans.command_ports
    tree_limits = built.plans.stages[0].commands[0]
    assert plan.bounded == "clip" and not plan.bounds_declared and tree_limits is plan
    assert "(its receiver's stated bounds)" in "\n".join(built.report.ports)
    node = built.graph.get_node("ports:rpm")
    assert len(node.constants()["low_arm_joint_positions"]) == 6


#: The loop's ports: the coils' currents one step late, the sphere's pose two steps late.
LOOP_PORTS = {"currents": CommandPort(latency=ms(1), reason="a test link, one step"),
              "tracker": Measurement(of="sphere.pose", latency=ms(2), reason="a test tracker, two steps")}


@pytest.fixture(scope="module")
def ported_loops() -> dict[str, tuple[object, dict]]:
    """The coil sphere's loop closed through a measurement port on the sphere's pose and a command port on the
    coils' currents, its controller called on the host and inside the compiled step: each build and its run."""
    found = {}
    for mode in ("callback", "graph"):
        built = fed_back_sphere(LOOP_PORTS, mode=mode, name=f"ported_{mode}").build()
        found[mode] = (built, built.run(ms(8), history=True).history)
    return found


@pytest.mark.parametrize("mode", ["callback", "graph"])
def test_a_controller_reads_a_measurement_port_and_writes_through_a_command_port(ported_loops, mode):
    built, history = ported_loops[mode]
    assert {"ports:coils", "measure:coils"} <= set(built.graph.node_names)
    classed = [line.strip() for line in built.report.controller if line.strip().startswith(("pose", "age"))]
    assert len(classed) == 2 and all("measurement, a declared measurement port" in line for line in classed)
    assert list(np.asarray(history["measure:coils"]["tracker_age"])) == [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    truth = np.asarray(history["sphere"]["pose"])
    measured = np.asarray(history["measure:coils"]["tracker"])
    np.testing.assert_array_equal(measured[2:], truth[:-2])
    written = np.asarray(history["ports:coils"]["written_currents"])
    np.testing.assert_array_equal(np.asarray(history["ports:coils"]["currents"])[1:], written[:-1])


def test_the_two_modes_run_a_ported_loop_to_the_same_numbers(ported_loops):
    """Swapping a controller's mode changes no physics, with ports as without: the steps are the same steps, each
    mode its own compiled program, so they agree to the run's rounding on the inputs' scale."""
    callback, graph = (np.asarray(ported_loops[mode][1]["sphere"]["pose"]) for mode in ("callback", "graph"))
    assert float(np.max(np.abs(callback - graph))) <= 64 * float(np.finfo(callback.dtype).eps)


def test_a_ports_age_bound_where_its_value_is_read_is_refused():
    exp = coil_sphere(LOOP_PORTS, name="aged_wrong", commands={})
    exp.set_controller(PoseFeedback(), reads={"pose": "coils.tracker.age", "age": "coils.tracker.age"},
                       writes={"currents": "coils.currents"}, mode="callback", rate=ms(1))
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    assert codes(refused.value) == ["controller.signal"] and "'1' with shape ()" in str(refused.value)


REFUSED = {
    "port.command-unknown": lambda: sphere(ports={"voltages": CommandPort(latency=ms(1), reason="r")}),
    "port.clock": lambda: sphere(CommandPort(latency=ms(1.5), reason="half a step")),
    "port.seam": lambda: sphere(CommandPort(drift=1.0, reason="r")),
    "port.declaration": lambda: sphere(CommandPort(latency=ms(1))),
    "port.source-unknown": lambda: sphere(ports={"pose": Measurement(of="sphere.wobble", reason="r")}),
}
NEAR_MISSES = {
    "port.command-unknown": lambda: sphere(ports={"currents": CommandPort(latency=ms(1), reason="r")}),
    "port.clock": lambda: sphere(CommandPort(latency=ms(2), reason="two steps")),
    "port.seam": lambda: sphere(CommandPort(reason="r")),
    "port.declaration": lambda: sphere(CommandPort(latency=ms(1), reason="the link's datasheet")),
    "port.source-unknown": lambda: sphere(ports={"pose": Measurement(of="sphere.pose", reason="r")}),
}


@pytest.mark.parametrize("code", sorted(REFUSED))
def test_each_port_refusal_and_its_near_miss(code):
    with pytest.raises(BuildRefused) as refused:
        REFUSED[code]().build()
    assert codes(refused.value) == [code]
    NEAR_MISSES[code]().build()


def test_bounds_with_no_reason_name_the_reason_and_unit_restated_wrong_is_refused():
    with pytest.raises(BuildRefused) as refused:
        sphere(CommandPort(unit=mm, reason="r")).build()
    assert codes(refused.value) == ["port.declaration"] and "restates its unit" in str(refused.value)
    sphere(CommandPort(unit=A, shape=(3,), bounds=A(25.0), reason="the amplifier's datasheet")).build()


def test_a_trip_with_no_bounds_and_a_bound_beyond_the_rest_of_the_shape_are_refused():
    with pytest.raises(BuildRefused) as refused:
        sphere(CommandPort(on_bound="trip", reason="r")).build()
    assert "no bounds to trip at" in str(refused.value)
    with pytest.raises(BuildRefused) as refused:
        sphere(CommandPort(bounds=(A((-1.0, -2.0)), A((1.0, 2.0))), reason="r")).build()
    assert codes(refused.value) == ["port.declaration"] and "takes shape (3,)" in str(refused.value)


def test_a_measurement_named_as_a_command_or_reserved_is_refused():
    with pytest.raises(BuildRefused) as refused:
        sphere(ports={"currents": Measurement(of="sphere.pose", reason="r")}).build()
    assert "is a command of the apparatus" in str(refused.value)
    with pytest.raises(BuildRefused) as refused:
        sphere(ports={"steps": Measurement(of="sphere.pose", reason="r")}).build()
    assert codes(refused.value) == ["port.declaration"]


def test_noise_on_a_pose_is_refused_naming_its_composite_units():
    with pytest.raises(BuildRefused) as refused:
        sphere(ports={"pose": Measurement(of="sphere.pose", noise=Gaussian(sigma=m(1e-6)), reason="r")}).build()
    assert codes(refused.value) == ["port.declaration"] and "composite" in str(refused.value)


def test_the_port_stage_is_on_a_derivatives_path_with_its_status():
    from mime.gradients import GradientStatus

    built = sphere(CommandPort(bounds=A(25.0), slew=A(4000.0) / s(1.0), reason="an amplifier")).build()
    (stage,) = built.plans.stages
    assert stage.status("ports:coils") is GradientStatus.PIECEWISE_KINK
    tripped = sphere(CommandPort(bounds=A(25.0), on_bound="trip", reason="an amplifier")).build()
    assert tripped.plans.stages[0].status("ports:coils") is GradientStatus.PIECEWISE_JUMP


def test_a_ports_traffic_is_recorded_through_record_and_its_loop_chain_reported(ported_loops):
    """A port's traffic is its stage's outputs: ``record=`` takes them as any per-step output (the written and the
    delivered command, a measured value's age); the report's controller block gives the loop's latency chain through
    the ports: a measurement two steps late, the controller's none (a tick reads the state as it stands and its writes
    act from that step) and the command one step late, three steps from a state to the step its command acts in."""
    built, history = ported_loops["graph"]
    result = built.run(ms(6), record=["ports:coils.written_currents", "ports:coils.currents",
                                      "measure:coils.tracker_age"])
    stage = result["ports:coils"]
    np.testing.assert_array_equal(stage["currents"].raw[1:], stage["written_currents"].raw[:-1])
    assert list(result["measure:coils"]["tracker_age"].raw) == [2.0] * 6
    (chain,) = [line.strip() for line in built.report.controller if line.strip().startswith("coils.tracker ->")]
    assert chain.startswith("coils.tracker -> coils.currents: 3 base steps (0.003 to 0.003 s)"), chain


def test_a_camera_ports_image_is_refused_in_record_naming_the_re_render():
    """Design R4: a port's image is the camera's frames by a second path, refused in ``record=`` as the camera's own
    image is (``camera.recorded-frames``); its age records."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "sensing"))
    from test_camera_port import camera_with

    from mime.build import ArgumentRefused

    built = camera_with(Measurement(of="camera:.image", rate=ms(6), reason="a test frame grabber")).build()
    with pytest.raises(ArgumentRefused) as refused:
        built.run(ms(6), record=["measure:imaging.frames"])
    assert codes(refused.value) == ["camera.recorded-frames"] and "mime.sensing.render" in str(refused.value)
    built.run(ms(6), record=["measure:imaging.frames_age"])
