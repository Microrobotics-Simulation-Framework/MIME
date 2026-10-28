"""The controller's declaration: every refusal it can meet, each raised here; the external inputs the build derives
from its writes, and nothing else; the double declaration of 0.2's A0, which cannot be written; its reads classed in
the build report and the saved form; the seam where a profile will judge them; and its round trip through the saved
form.

The motor rig of ``control_cases`` is used wherever the arm is not the point: it builds in about a second."""

import dataclasses

import numpy as np
import pytest
from chain_cases import TRACKING_READS, tracked_rpm
from control_cases import Ticker, TwoWrites, Unregistered, Watcher, spun_magnet

import mime.build._construct as construct_module
import mime.build._plan as plan_module
from mime.build import BuildRefused, ReadClass, _testing
from mime.build._maddening import external_inputs, run_scan
from mime.experiment import Experiment
from mime.registry import Controller
from mime.units import ms, rad


def refused(make) -> BuildRefused:
    with pytest.raises(BuildRefused) as caught:
        make().build()
    return caught.value


def codes(make) -> list[str]:
    """The codes the build refuses with, less the command a refused controller leaves without a writer, which every
    such refusal brings (``command.undriven``, a true consequence)."""
    found = refused(make).codes()
    return [code for code in found if code != "command.undriven"] or found


# -- the refusals ---------------------------------------------------------------------------------------------------


def test_a_second_controller_is_refused():
    def twice():
        exp = spun_magnet()
        exp.set_controller(Ticker(), reads={"pose": "umr.pose"}, writes={"angle": "spin.motor.angle"},
                           mode="callback")
        return exp

    assert codes(twice) == ["controller.more-than-one"]


@pytest.mark.parametrize(("mode", "lands"), [("graph", "phase 7b2"), (1, "phase 7b2"), ("client", "0.3.0a2"),
                                             (3, "0.3.0a2"), ("fmu", "0.3.0a2"), (4, "0.3.0a2")])
def test_a_mode_not_built_yet_is_refused_naming_where_it_lands(mode, lands):
    caught = refused(lambda: spun_magnet(mode=mode))
    assert caught.codes() == ["controller.mode-unbuilt"]
    assert lands in caught.refusals[0].message and "mode=" in caught.refusals[0].action


@pytest.mark.parametrize("mode", ["fast", 0, 5, None, True])
def test_a_mode_that_names_none_is_refused(mode):
    caught = refused(lambda: spun_magnet(mode=mode))
    assert caught.codes() == ["controller.mode-unknown"]
    assert "'callback' or 2" in caught.refusals[0].action


def test_the_callback_mode_by_word_and_by_number_builds_alike():
    assert spun_magnet(mode=2).build().report.derived_hash == spun_magnet(mode="callback").build().report.derived_hash


def test_a_rate_not_a_whole_multiple_of_dt_is_refused_naming_the_nearest():
    caught = refused(lambda: spun_magnet(rate=ms(2.5)))
    assert caught.codes() == ["run.rate-not-multiple"]
    assert "0.002 s and 0.003 s" in caught.refusals[0].action


@pytest.mark.parametrize("latency", [ms(0.5), ms(-1.0), rad(1.0)])
def test_a_latency_not_a_whole_number_of_steps_from_zero_is_refused(latency):
    assert codes(lambda: spun_magnet(latency=latency)) == ["controller.latency"]


@pytest.mark.parametrize(("reads", "writes", "said"), [
    ({}, {"angle": "spin.motor.angle"}, "'pose', and nothing is bound to it"),
    ({"pose": "umr.pose", "speed": "umr.twist"}, {"angle": "spin.motor.angle"}, "reads no label 'speed'"),
    ({"pose": 3.0}, {"angle": "spin.motor.angle"}, "not a path"),
    (["umr.pose"], {"angle": "spin.motor.angle"}, "is not a mapping"),
    ({"pose": "umr.pose"}, {}, "'angle', and nothing is bound to it"),
])
def test_each_label_is_bound_once_to_a_path(reads, writes, said):
    caught = refused(lambda: spun_magnet(reads=reads, writes=writes))
    assert codes(lambda: spun_magnet(reads=reads, writes=writes)) == ["controller.labels"]
    assert said in caught.refusals[0].message


@pytest.mark.parametrize(("path", "said"), [
    ("spin.motor.torque", "apparatus 'spin' takes no command 'motor.torque'"),
    ("spin.motor.mount_poses", "apparatus 'spin' takes no command 'motor.mount_poses'"),
    ("umr.pose", "is no apparatus's command"),
    ("spin", "is no apparatus's command"),
])
def test_a_write_no_command_has_is_refused(path, said):
    caught = refused(lambda: spun_magnet(writes={"angle": path}))
    assert caught.codes() == ["controller.write-unknown", "command.undriven"]
    assert said in caught.refusals[0].message and "spin.motor.angle" in caught.refusals[0].action


def test_a_command_with_a_drive_that_the_controller_writes_is_refused():
    caught = refused(lambda: spun_magnet(drive_angle=True))
    assert caught.codes() == ["controller.written-twice"]
    assert "has a drive, and the controller's write 'angle' writes it too" in caught.refusals[0].message


def test_a_command_two_write_labels_name_is_refused():
    caught = refused(lambda: spun_magnet(controller=TwoWrites(), reads={},
                                         writes={"first": "spin.motor.angle", "second": "spin.motor.angle"}))
    assert caught.codes() == ["controller.written-twice"]


@pytest.mark.parametrize(("path", "code"), [("umr.position", "controller.read-unknown"),
                                            ("spin.motor.joint_positions", "controller.read-unknown"),
                                            ("swim_speed:nobody.speed", "controller.read-unknown"),
                                            ("umr.twist", "controller.signal")])
def test_a_read_of_nothing_published_or_of_another_signal_is_refused(path, code):
    assert codes(lambda: spun_magnet(reads={"pose": path})) == [code]


def test_a_read_of_a_field_its_observation_does_not_hold_is_refused_once_its_node_is_built():
    caught = refused(lambda: spun_magnet(controller=Watcher(), observe=True,
                                         reads={"pose": "umr.pose", "speed": "swim_speed:umr.velocity"}))
    assert caught.codes() == ["controller.read-unknown"]
    assert "holds no 'velocity'" in caught.refusals[0].message and "speed" in caught.refusals[0].action


def test_a_controller_no_one_registered_is_refused():
    assert codes(lambda: spun_magnet(controller=Unregistered())) == ["entry.unregistered"]


def test_set_controller_takes_a_controller():
    with pytest.raises(TypeError, match="takes a Controller"):
        spun_magnet().set_controller("ticker", reads={}, writes={}, mode="callback")  # type: ignore[arg-type]


def test_a_controller_entry_without_writes_is_not_registered():
    from mime.registry import register_controller

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Silent(Controller):
        name = "oracle.silent"
        version = "0.1.0"
        differentiability = Ticker.differentiability

    with pytest.raises(TypeError, match="writes: a non-empty mapping"):
        register_controller(Silent)


# -- external inputs: derived from the writes, and nothing else -----------------------------------------------------


def test_every_external_input_is_derived_from_a_write_and_the_report_lists_it():
    built = tracked_rpm().build()
    assert external_inputs(built.graph) == [("rpm.arm", "joint_positions")]
    assert built.report.external_inputs == ("rpm.arm.joint_positions",)
    assert "joint_positions -> rpm.arm.joint_positions (an external input of rpm.arm.joint_positions" in \
        built.report.text()
    assert "drive:rpm.arm.joint_positions" not in [line.owner for line in built.report.nodes]


def test_without_a_controller_the_graph_has_no_external_input():
    from chain_cases import kinematic_rpm

    assert external_inputs(kinematic_rpm().build().graph) == []


def test_the_0_2_double_declaration_cannot_be_written():
    """A0's 0.2 graph declared the motor's mount pose an external input and also wired it from the arm, and MADDENING
    0.4.0 fed it zeros. In 0.3 an external input comes only from a controller's write, a write names a command, and a
    mount is derived by the chain, not commanded: naming it is refused, and so is a command both driven and
    written."""
    caught = refused(lambda: tracked_rpm(writes={"joint_positions": "rpm.motor.mount_poses"}))
    assert "controller.write-unknown" in caught.codes()
    assert "takes no command 'motor.mount_poses'" in str(caught)
    assert codes(lambda: tracked_rpm(controlled=())) == ["controller.written-twice"]


def test_control_a_write_dropped_from_the_derived_inputs_is_refused(monkeypatch):
    """Control (a): the build derives the external input of every write but one. The command it drops has no writer,
    and the post-build check refuses it, naming the write; the graph never runs on MADDENING's zeros."""
    found = check_dropped_write(monkeypatch)
    assert found == ["post-build.command-writers"]


def check_dropped_write(monkeypatch) -> list[str]:
    derive = construct_module.derived_inputs
    monkeypatch.setattr(construct_module, "derived_inputs", lambda plans: derive(plans)[:-1])
    try:
        tracked_rpm().build()
    except BuildRefused as caught:
        assert "the controller's write 'joint_positions' (rpm.arm.joint_positions)" in str(caught)
        return caught.codes()
    return []


def test_a_command_whose_drive_edge_is_removed_after_construction_is_refused():
    def unplug(graph, plans):
        graph.remove_edge("drive:rpm.motor.angle", "rpm.motor", "value", "angle")

    with _testing.after_construction(unplug), pytest.raises(BuildRefused) as caught:
        from chain_cases import kinematic_rpm

        kinematic_rpm(prescribed="angle").build()
    assert "post-build.command-writers" in caught.value.codes()


def test_a_scan_that_does_not_feed_every_external_input_is_refused():
    """MIME's own scans feed every input: one that leaves the controller's write out raises, where MADDENING would
    run it on zeros; fed the controller's held writes it runs."""
    built = spun_magnet().build()
    with pytest.raises(ValueError, match="not fed: spin.motor.angle"):
        run_scan(built.graph, 2)
    state = run_scan(built.graph, 2, fed=built.held_inputs())
    assert float(state["spin.motor"]["angle"]) == -1.0


def test_a_read_published_under_a_field_the_node_does_not_hold_is_refused_after_construction():
    def misname(graph, plans):
        held = plans.controller
        object.__setattr__(plans, "controller", dataclasses.replace(
            held, reads=tuple(dataclasses.replace(r, field="nowhere") for r in held.reads)))

    with _testing.after_construction(misname), pytest.raises(BuildRefused) as caught:
        spun_magnet().build()
    assert caught.value.codes() == ["post-build.read-field"]


# -- classed reads ---------------------------------------------------------------------------------------------------


def read_classes(built) -> dict[str, str]:
    """Each read's class as the build report states it and as the lock records it, which must agree."""
    lines = {line.split(" <- ")[0].strip(): line for line in built.report.controller if " <- " in line}
    stated = {label: ("state" if ": state, ground truth, not deployable" in line else
                      "observation" if ": observation, an observation's output" in line else "unclassed")
              for label, line in lines.items()}
    held = built.record.resolved["controller"]
    recorded = {label: read["read_class"] for label, read in held["reads"].items()} if held else {}
    return stated if stated == recorded else {"report": str(stated), "lock": str(recorded)}


def classes_failures(built, expected: dict[str, str]) -> list[str]:
    found = read_classes(built)
    return [] if found == expected else [f"classes {found}, expected {expected}"]


def watched():
    return spun_magnet(controller=Watcher(), observe=True,
                       reads={"pose": "umr.pose", "speed": "swim_speed:umr.speed"})


def test_each_read_is_classed_state_or_observation_in_the_report_and_the_lock():
    built = watched().build()
    assert classes_failures(built, {"pose": "state", "speed": "observation"}) == []
    report = built.report.text()
    assert "pose <- umr.pose (body.pose of umr): state, ground truth, not deployable" in report
    assert "speed <- swim_speed:umr.speed (observation swim_speed:umr's speed): observation" in report
    held = built.record.resolved["controller"]
    assert held["reads"]["speed"] == {"path": "swim_speed:umr.speed", "read_class": "observation",
                                      "meaning": ReadClass.OBSERVATION.meaning, "node": "swim_speed:umr",
                                      "field": "speed"}


def test_a0s_reads_are_all_ground_truth():
    assert classes_failures(tracked_rpm().build(), {label: "state" for label in TRACKING_READS}) == []


def test_control_a_truth_read_classed_as_an_observation_is_caught(monkeypatch):
    """Control (d): the planner classes a state read as an observation. The report and the lock then call the body's
    pose deployable, and the classes check fails."""
    assert classed_wrongly(monkeypatch) != []


def classed_wrongly(monkeypatch) -> list[str]:
    held = plan_module._Planner.controller_read

    def mislabelled(self, *args):
        found = held(self, *args)
        return dataclasses.replace(found, read_class=ReadClass.OBSERVATION) if found is not None else None

    monkeypatch.setattr(plan_module._Planner, "controller_read", mislabelled)
    return classes_failures(watched().build(), {"pose": "state", "speed": "observation"})


def test_the_profile_seam_can_refuse_a_state_read(monkeypatch):
    """The seam a1's profiles will use: a judgement that refuses state reads refuses the build, naming the read; the
    seam as it stands refuses none."""
    assert plan_module.refused_reads(None, watched().plan().controller.reads) == []
    monkeypatch.setattr(plan_module, "refused_reads", lambda profile, reads: [
        (read, "a profile that holds a controller to what an instrument gives") for read in reads
        if read.read_class is ReadClass.STATE])
    caught = refused(watched)
    assert caught.codes() == ["controller.read-refused"]
    assert "'pose' (umr.pose, state) is refused" in caught.refusals[0].message


def test_the_gradients_through_a_callback_are_declared_absent():
    built = tracked_rpm().build()
    assert "gradients: none pass through it in mode 2" in built.report.text()
    from mime.control import ArmTracking
    from mime.gradients import GradientStatus

    assert ArmTracking.differentiability is GradientStatus.CUT and ArmTracking.gradient is not None
    assert ArmTracking.gradient.route is not None and "inside the compiled step" in ArmTracking.gradient.route


# -- the saved form ---------------------------------------------------------------------------------------------------


def test_the_document_holds_the_controller_by_name_with_its_bindings_and_reads_it_back(tmp_path):
    exp = tracked_rpm()
    document = exp.document()
    held = document["controller"]
    assert held is not None
    assert (held["name"], held["version"], held["mode"]) == ("arm_tracking", "0.1.0", "callback")
    assert held["reads"] == dict(sorted(TRACKING_READS.items()))
    assert held["writes"] == {"joint_positions": "rpm.arm.joint_positions"}
    assert held["rate"] == {"value": 0.002, "unit": "s"} and held["latency"] == {"value": 0.001, "unit": "s"}
    built = exp.build()
    built.save(tmp_path)
    loaded = Experiment.load(tmp_path)
    assert loaded.document() == document
    again = loaded.build()
    assert again.report.controller == built.report.controller
    assert again.verify().moved == ()


def test_a_saved_controller_with_an_unknown_label_is_refused_when_its_document_is_built(tmp_path):
    from mime.saved import DOCUMENT_FILE, dump_document

    document = tracked_rpm().document()
    held = dict(document["controller"])
    held["reads"] = {**held["reads"], "elbow": "rpm.arm.frame_poses"}
    (tmp_path / DOCUMENT_FILE).write_text(dump_document({**document, "controller": held}), encoding="utf-8")
    import shutil

    from chain_cases import AR4
    from mime.saved._document import description_path
    from mime.scene import sha256_hex

    target = tmp_path / description_path(AR4.name, sha256_hex(AR4.read_bytes()))
    target.parent.mkdir(parents=True)
    shutil.copy(AR4, target)
    with pytest.raises(BuildRefused) as caught:
        Experiment.from_document(tmp_path / DOCUMENT_FILE).build()
    assert caught.value.codes() == ["controller.labels"]


def test_the_report_states_the_mode_period_and_latency():
    line = tracked_rpm().build().report.controller[0]
    assert line.startswith("controller: arm_tracking 0.1.0, mode 2 (a callback on the host, between compiled chunks), "
                           "every 2 steps (0.002 s), latency 1 step (0.001 s)")
    assert np.isclose(float(line.split("latency 1 step (")[1].split(" s)")[0]), 0.001)
