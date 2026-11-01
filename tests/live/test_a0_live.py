"""A0 live (design #197): its saved folder served by ``mime serve`` at real time, every frame delivered and each the
batch run's state at its steps bit for bit, the session keeping pace; its live session's recording over its run
folder's stage; no byte of an AR4 mesh in what a pack of A0's saved folder or of a live session's run folder carries,
nor anywhere in the run folder, with the visual set's stand-in baked in the cache and drawn; and A0's controller, called back, steered by its live input. The controls:
a mesh planted in a copy of the run folder is found by the same scan, and a session that sets nothing leaves the magnet
where it was."""

import json
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest
from script_a0 import a0
from visual_cases import holds_any, mesh_bytes, planted

from mime.experiment import Experiment
from mime.live import FrameOrder, LiveClient
from mime.build import BuildRefused
from mime.saved import pack, read_folder
from mime.saved._folder import named_files
from mime.scene import check_stage
from mime.units import ms, s
from mime.viz import RECORDING_FILE


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    held = tmp_path_factory.mktemp("a0") / "script_a0"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a0().save(held)
    return held


def test_a0_through_mime_serve_keeps_real_time_and_each_frame_is_the_batch_runs_state(folder, tmp_path):
    ready = tmp_path / "ready.json"
    process = subprocess.Popen([sys.executable, "-m", "mime.cli", "serve", str(folder), "--commands",
                                "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0", "--ready-file", str(ready),
                                "--out", str(tmp_path / "out")], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while not ready.exists():
        assert process.poll() is None
        time.sleep(0.05)
    endpoints = json.loads(ready.read_text())
    live = LiveClient(endpoints["commands"], endpoints["frames"])
    live.request("start")
    frames, arrived = [], []
    while not frames or frames[-1].time < 1.0:
        frames.append(live.frame())
        arrived.append(time.monotonic())
    live.request("stop")
    assert process.wait(timeout=120) == 0
    order = FrameOrder()
    assert [order.see(f) for f in frames] == [None] * len(frames)
    # Paced at real time: from the first chunks after the compile to the last, the wall clock kept with simulated time,
    # neither behind it nor running ahead to make up the compile.
    simulated = frames[-1].time - frames[3].time
    assert 0.8 * simulated <= arrived[-1] - arrived[3] <= 1.25 * simulated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = a0().build().run(s(frames[-1].step * 0.0005), history=True).history
    for held in frames[1:]:
        for path, value in held.values.items():
            node, _, field = path.rpartition(".")
            assert value.tobytes() == np.asarray(history[node][field][held.step - 1]).tobytes(), (path, held.step)


def files_hold(folder, needles) -> list[str]:
    return [str(p.relative_to(folder)) for p in sorted(folder.rglob("*")) if p.is_file() and
            holds_any(p.read_bytes(), needles)]


def test_no_byte_of_an_ar4_mesh_in_a0s_packed_folder_or_a_live_run_folder(folder, tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    layers = planted()
    needles = mesh_bytes(layers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = Experiment.load(folder).build().live(pace=None, out=tmp_path / "run")
    hello = session.hello()
    assert hello["visuals"]["drawn"] == 18 and hello["visuals"]["overlay"].startswith(str(tmp_path / "cache"))
    session.record(True)
    frames = session.step(4)
    run = session.stop()
    assert files_hold(run, needles) == [] and files_hold(folder, needles) == []
    # What a pack carries is every file the saved form names, and none holds a byte of a mesh. A0's folder itself is
    # refused by pack, for one reason only: its URDF's xacro header names the machine it was generated on, a path pack
    # never carries (the file is pinned by A0's frozen provenance, so it is not edited here).
    for saved in (folder, run / "experiment"):
        with pytest.raises(BuildRefused) as refused:
            pack(saved, out=tmp_path / f"{saved.name}.mimebundle")
        assert refused.value.codes() == ["bundle.machine-path"] and "ar4.urdf" in str(refused.value)
        form = read_folder(saved)
        assert [name for name in sorted(named_files(form)) if holds_any((saved / name).read_bytes(), needles)] == []
    # The recording over the run folder's stage: no finding, and its arm frames the frames' poses exactly.
    recording = run / RECORDING_FILE
    assert check_stage(recording) == []
    from pxr import Usd

    stage = Usd.Stage.Open(str(recording))
    flange = stage.GetPrimAtPath("/World/rpm/arm/link_6")
    last = np.asarray(frames[-1].values["rpm.arm.frame_poses"], dtype=np.float64)
    link_6 = next(e.index for e in session.built.binding().entries if e.name == "rpm.arm.link_6")
    assert np.array_equal(np.asarray(flange.GetAttribute("xformOp:translate").Get(4)), last[link_6, :3])
    # The control: a baked mesh planted in a copy of the run folder is found by the same scan.
    copy = tmp_path / "planted"
    shutil.copytree(run, copy)
    shutil.copyfile(next(iter(layers.values())), copy / "experiment" / "scene" / "planted.usdc")
    assert files_hold(copy, needles) == ["experiment/scene/planted.usdc"]


def rotor_ahead(frames) -> float:
    """The rotor's lead on the body along the tube (x), m, over the frames."""
    return float(np.mean([np.asarray(f.values["rpm.motor.frame_poses"], dtype=np.float64)[1, 0] -
                          float(f.values["umr.pose"][0]) for f in frames]))


def test_a0s_controller_called_back_is_steered_by_its_live_input():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steered = a0(mode="callback", rate=ms(10)).build().live(pace=None, chunk=ms(10))
        plain = a0(mode="callback", rate=ms(10)).build().live(pace=None, chunk=ms(10))
    for session in (steered, plain):
        session.step(50)
    steered.set("offset", 0.002)
    after_steered, after_plain = steered.step(50)[30:], plain.step(50)[30:]
    # The magnet moves towards a point 2 mm ahead, through the controller's smoothing (each tick a fraction 0.005 of the
    # way), so that over the ticks 31 to 50 after the set its lead on the body grows by 2 mm times the mean of
    # 1 - 0.995**n: the prediction, held within 30% (the body, pulled by the magnet, moves too).
    predicted = 0.002 * float(np.mean([1 - 0.995 ** n for n in range(31, 51)]))
    assert rotor_ahead(after_steered) - rotor_ahead(after_plain) == pytest.approx(predicted, rel=0.3)
    # The control: the session that set nothing is where a batch run of the same steps is.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = a0(mode="callback", rate=ms(10)).build().run(s(1.0))
    assert np.asarray(batch.final["umr"]["pose"]).tobytes() == np.asarray(after_plain[-1].values["umr.pose"]).tobytes()
