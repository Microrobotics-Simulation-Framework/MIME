"""Ports through the build: what a declared port builds, where, and what it leaves alone.

- **No port, no change** (design section 9, control 1): an apparatus with no port, with ``ports={}``, and with an
  all-default ``CommandPort()`` on a command and an ideal ``Measurement`` (every value its source makes, at once, no
  noise or quantum, its age not read) builds the same program, its derived hash and canonical text the same, on the
  coil-array sphere and on A0's apparatus. The controls: a stage built for an ideal command port (``stage_for_ideal``),
  and one built for an ideal measurement (``measure_for_ideal``), must each change both.
- **No two ports hold one field**: a port named as another's age (``tr`` and ``tr_age``) is refused; the control
  (``clash_unchecked``, the check removed) builds, and the second overwrites the first's age in silence.
- **A response is refused by name** until it is built, so none is ignored; a kinematic arm's port takes its joints'
  stated limits as its bounds, before its slew.
- **Controllers through ports**: a controller called on the host and one inside the compiled step, each reading a
  measurement port and writing through a command port, run the same steps to the same numbers; their reads are
  classed ``measurement``, the age is read by its path.
- **Each refusal**, with a near miss that builds."""

from __future__ import annotations

import numpy as np
import pytest
from chain_cases import kinematic_rpm
from port_cases import PoseFeedback, coil_sphere, fed_back_sphere, planted

from mime import CommandPort, Measurement
from mime.build import BuildRefused, _ports
from mime.testing._graph import graph_snapshot
from mime.things import Gaussian
from mime.units import A, m, mm, ms, s


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


def test_a_response_is_refused_by_name_until_it_is_built():
    with pytest.raises(BuildRefused) as refused:
        sphere(CommandPort(response=object(), reason="the coils' L/R, measured")).build()
    assert codes(refused.value) == ["port.seam"] and "declares response=" in str(refused.value)
    assert "MIME 0.3.0.dev2" in str(refused.value)


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


def test_a_declared_port_is_unsavable_until_its_saved_form(tmp_path):
    built = sphere(CommandPort(latency=ms(1), reason="a link")).build()
    with pytest.raises(BuildRefused) as refused:
        built.save(tmp_path / "ported")
    assert codes(refused.value) == ["save.unsavable"] and "declares ports" in str(refused.value)
    sphere().build().save(tmp_path / "unported")


def test_the_port_stage_is_on_a_derivatives_path_with_its_status():
    from mime.gradients import GradientStatus

    built = sphere(CommandPort(bounds=A(25.0), slew=A(4000.0) / s(1.0), reason="an amplifier")).build()
    (stage,) = built.plans.stages
    assert stage.status("ports:coils") is GradientStatus.PIECEWISE_KINK
    tripped = sphere(CommandPort(bounds=A(25.0), on_bound="trip", reason="an amplifier")).build()
    assert tripped.plans.stages[0].status("ports:coils") is GradientStatus.PIECEWISE_JUMP
