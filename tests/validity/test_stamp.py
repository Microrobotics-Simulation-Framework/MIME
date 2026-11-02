"""The stamp on every output (design #190 item 4): the profile a build was made under and each item it leaves pending,
on the build report, ``RunResult``'s printout, ``outputs.json`` (and the result read back from it), the ``.mat``'s
``run`` and the run's own saved form; a sealed build's says nothing is pending. Its controls: a manifest written before
profiles reads with no stamp and says so, and an unsealed and a sealed build differ in every stamp.

``mime pending <folder>`` prints what ``built.report.pending()`` prints, and exits 1 when something is pending, 0 when
nothing is, 2 when the load is refused."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import scipy.io
from profile_cases import BARE_PENDING, Paperwork, settling_ball

from mime.cli import main as mime_main
from mime.results import RUN_MANIFEST, RunResult
from mime.saved import LOCK_FILE
from mime.units import mm, ms
from mime.validity import Stamp

SEALED = Paperwork(profile="research", acceptance="water is Newtonian at a settling ball's shear rates",
                   density="a steel ball's density", spacing=mm(0.2, reason="converged at half this spacing"),
                   blob=0.5, quadrature_refinement=4,
                   sources={"drag:ball.blob": "the regularised method's usual blob",
                            "drag:ball.quadrature_refinement": "the quadrature's spacing half the blob"})


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory):
    """The bare ball and the sealed one, each built, run three steps into a folder, and saved."""
    held = {}
    for name, paperwork in (("bare", Paperwork()), ("sealed", SEALED)):
        folder = tmp_path_factory.mktemp(name)
        built = settling_ball(paperwork).build()
        written = built.run_to(folder / "run", ms(3.0), record=["swim_speed:ball.speed"])
        built.save(folder / "saved")
        held[name] = (built, written, folder)
    return held


def stamps(built, written, folder: Path) -> tuple[dict[str, str], dict]:
    """The stamp as each output carries it, in one text each, and the run's lock's record of the profile (its items by
    target and codes, never their words)."""
    manifest = json.loads((folder / "run" / RUN_MANIFEST).read_text())
    written.result.to_mat(folder / "run.mat")
    mat = scipy.io.loadmat(folder / "run.mat", simplify_cells=True)["run"]
    lock = json.loads((folder / "run" / "experiment" / LOCK_FILE).read_text())
    return {"report": next(line for line in built.report.text().splitlines() if line.startswith("profile ")),
            "result": written.result.text().splitlines()[-1],
            "outputs.json": Stamp.from_record(manifest["stamp"]).text(),
            "read back": RunResult.read(folder / "run").text().splitlines()[-1],
            "mat": str(mat["stamp"])}, lock["resolved"]["profile"]


def test_every_output_carries_the_stamp_of_what_is_pending(runs):
    built, written, folder = runs["bare"]
    found, locked = stamps(built, written, folder)
    expected = built.report.profile.stamp().text()
    assert expected.startswith("profile exploratory: 5 items pending, not sealed")
    assert all(target in expected for target in BARE_PENDING)
    assert ". Measured: " in expected and "drag:ball.spacing: on the walk's T1 sphere (338 force points)" in expected
    assert found == dict.fromkeys(found, expected)
    assert locked["profile"] == "exploratory"
    assert [item["target"] for item in locked["pending"]] == list(BARE_PENDING)
    mat = scipy.io.loadmat(folder / "run.mat", simplify_cells=True)["run"]
    assert mat["profile"] == "exploratory" and "drag:ball.spacing" in mat["stamp"]


def test_the_control_a_sealed_build_s_stamp_says_nothing_is_pending(runs):
    built, written, folder = runs["sealed"]
    found, locked = stamps(built, written, folder)
    assert found == dict.fromkeys(found, "profile research: nothing pending")
    assert (locked["profile"], locked["pending"]) == ("research", [])
    assert found != stamps(*runs["bare"])[0]


def test_the_control_a_manifest_written_before_profiles_reads_with_no_stamp(runs, tmp_path: Path):
    _, written, folder = runs["bare"]
    old = tmp_path / "old"
    old.mkdir()
    manifest = json.loads((folder / "run" / RUN_MANIFEST).read_text())
    del manifest["stamp"]
    (old / RUN_MANIFEST).write_text(json.dumps({**manifest, "version": 1}))
    (old / "outputs.npz").write_bytes((folder / "run" / "outputs.npz").read_bytes())
    read = RunResult.read(old)
    assert read.layout.stamp is None
    assert read.text().splitlines()[-1] == "profile: not recorded (a run written before profiles)"


def test_mime_pending_prints_the_items_and_exits_with_what_it_found(runs, capsys: pytest.CaptureFixture[str]):
    built, _, folder = runs["bare"]
    assert mime_main(["pending", str(folder / "saved")]) == 1
    assert capsys.readouterr().out == built.report.pending()
    sealed, _, sealed_folder = runs["sealed"]
    assert mime_main(["pending", str(sealed_folder / "saved")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("profile research: nothing pending.") and "listed by research" in out
    assert mime_main(["pending", str(folder / "no such folder")]) == 2


def test_the_stamp_reads_back_from_its_record():
    stamp = Stamp("exploratory", ("drag:ball.spacing (a numerical default)",), ("drag:ball.spacing: measured",))
    assert Stamp.from_record(json.loads(json.dumps(stamp.record()))) == stamp
    assert replace(stamp, pending=()).text() == "profile exploratory: nothing pending"
