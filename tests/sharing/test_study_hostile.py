"""A study container, or a study folder, written by someone who means harm (lane P, P2's fix round): a group's NPZ that
names gigabytes in a few kilobytes, a manifest or a recipe of the wrong shapes, a name that is a shell command, a
destination that is a link. Each is refused with a code of its own, before the memory, the exception or the command
that it aims at.

The memory a read allocates is measured with ``tracemalloc`` (NumPy reports its arrays to it), against a control that
reads the same entry with no bound and must show the allocation, so that a measurement too coarse to see it fails."""

from __future__ import annotations

import io
import json
import os
import random
import shlex
import shutil
import tracemalloc
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from push_cases import saved_toy
from study_plants import PLANTS, rewritten
from study_rigs import write_study

from mime.build import BuildRefused, MimeError
from mime.cli import main
from mime.results import StudyResult
from mime.results._study_folder import CONTENT_KEY, EPOCH
from mime.saved import pack, pull, unpack
from mime.saved._studies import incomplete_problem, unpack_study

DECLARED = 128 * 2**20
#: What a refused read may allocate: far below the entry's declared size, above what unpacking a toy study holds.
BOUND = 32 * 2**20


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_study(tmp_path_factory.mktemp("written") / "wall_scan")


@pytest.fixture
def study(written: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(written, tmp_path / "work" / written.name))


@pytest.fixture(scope="module")
def container(written: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = Path(shutil.copytree(written, tmp_path_factory.mktemp("packed") / "wall_scan"))
    return pack(folder)


def refusal_of(action: Callable[[], object]) -> BuildRefused | MimeError:
    try:
        action()
    except MimeError as refused:
        return refused  # type: ignore[return-value]
    raise AssertionError("not refused")


def codes_of(action: Callable[[], object]) -> list[str]:
    return [each.code for each in refusal_of(action).refusals]


def peak_of(action: Callable[[], object]) -> tuple[int, list[str]]:
    """What ``action`` refused with, and the most memory it held at once."""
    tracemalloc.start()
    try:
        found = codes_of(action)
        return tracemalloc.get_traced_memory()[1], found
    finally:
        tracemalloc.stop()


# --- a group's NPZ that names more than it holds -----------------------------------------------------------------------


def bombed_npz(original: bytes, *, compressed: bool, victim: str | None = None) -> bytes:
    """``original`` with the entry of member 0's content key replaced by an array of ``DECLARED`` bytes: deflated and
    holding them (a few hundred kilobytes of zeros, ``compressed``), or stored and holding a few bytes of the header's
    claim."""
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "|u1", "fortran_order": False, "shape": (DECLARED,)})
    victim = victim or f"m0/{CONTENT_KEY}.npy"
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as target:
        for info in source.infolist():
            if info.filename != victim:
                target.writestr(info, source.read(info))
                continue
            held = zipfile.ZipInfo(victim, EPOCH)
            held.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
            with target.open(held, "w", force_zip64=True) as entry:
                entry.write(header.getvalue())
                left = DECLARED if compressed else 64
                while left > 0:
                    entry.write(bytes(min(left, 1 << 22)))
                    left -= 1 << 22
    return out.getvalue()


def with_group_file(source: Path, data: bytes, *, record_hash: bool) -> Path:
    """The container ``source`` rewritten with group 0's NPZ as ``data``, every record agreeing with it, and (not
    ``record_hash``) the study's own record of its sha256 left as it was."""
    import hashlib

    def change(members: dict[str, bytes], manifest: dict[str, object]) -> None:
        members["study/groups/0/outputs.npz"] = data
        held = json.loads(members["study/study.json"])
        if record_hash:
            held["groups"][0]["sha256"] = hashlib.sha256(data).hexdigest()
        members["study/study.json"] = (json.dumps(held, indent=1, sort_keys=True) + "\n").encode()

    return rewritten(source, change)


@pytest.fixture(scope="module")
def original_npz(container: Path) -> bytes:
    with zipfile.ZipFile(container) as archive:
        return archive.read("study/groups/0/outputs.npz")


def test_the_measurement_sees_the_allocation_it_bounds(original_npz: bytes):
    """The control: the same entry read with no bound (NumPy's own reader, which the study reader used before) allocates
    what its header names, and the measurement shows it."""
    held = bombed_npz(original_npz, compressed=True)
    tracemalloc.start()
    try:
        with zipfile.ZipFile(io.BytesIO(held)) as archive, archive.open(f"m0/{CONTENT_KEY}.npy") as entry:
            np.lib.format.read_array(entry, allow_pickle=False)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak >= DECLARED > BOUND


@pytest.mark.parametrize("compressed", [True, False], ids=["deflated", "stored-lying-header"])
def test_an_npz_entry_that_names_more_than_it_holds_is_refused_before_it_is_allocated(
        container: Path, original_npz: bytes, tmp_path: Path, compressed: bool):
    """The container's every record agrees with the forged NPZ (a person who rewrites a container makes them agree), so
    only the NPZ's own entry can show it: refused as a group file that is not what a run wrote, at unpack."""
    forged = with_group_file(container, bombed_npz(original_npz, compressed=compressed), record_hash=True)
    peak, found = peak_of(lambda: unpack_study(forged, tmp_path / "out"))
    assert found == ["study.group-file"]
    assert peak < BOUND
    assert not (tmp_path / "out").exists()


def test_an_entry_no_check_reads_is_bounded_too(container: Path, original_npz: bytes, tmp_path: Path):
    """The entry is a member's array that unpacking does not read (a later read of the member would): every entry is held
    to what it holds when the group file is accepted, so that read allocates no more than the file is large."""
    with zipfile.ZipFile(io.BytesIO(original_npz)) as archive:
        victim = next(name for name in archive.namelist() if name.startswith("m2/") and "final" in name)
    forged = with_group_file(container, bombed_npz(original_npz, compressed=True, victim=victim), record_hash=True)
    peak, found = peak_of(lambda: unpack_study(forged, tmp_path / "out"))
    assert found == ["study.group-file"] and peak < BOUND


def test_a_group_file_is_hashed_before_it_is_opened_as_an_npz(
        container: Path, original_npz: bytes, tmp_path: Path):
    """A group file that is not the one the study records is refused on its hash, whatever it holds: the entry that
    would be allocated is never read."""
    forged = with_group_file(container, bombed_npz(original_npz, compressed=True), record_hash=False)
    peak, found = peak_of(lambda: unpack_study(forged, tmp_path / "out"))
    assert found == ["study.group-file"] and peak < BOUND
    assert "is not the file study.json records for group 0" in str(refusal_of(lambda: unpack_study(forged, tmp_path / "o2")))


@pytest.mark.parametrize("record_hash", [True, False], ids=["recorded", "not-recorded"])
def test_a_study_folder_read_in_place_holds_the_same_bound(study: Path, original_npz: bytes, record_hash: bool):
    """``StudyResult.read`` of a folder (one a person is handed, not one unpacked): the same reader, the same bound. A
    study folder is a thing a user is handed, as a folder of files, so the read is not the unpack's alone."""
    forged = bombed_npz(original_npz, compressed=True)
    (study / "groups" / "0" / "outputs.npz").write_bytes(forged)
    if record_hash:
        import hashlib

        held = json.loads((study / "study.json").read_text(encoding="utf-8"))
        held["groups"][0]["sha256"] = hashlib.sha256(forged).hexdigest()
        (study / "study.json").write_text(json.dumps(held, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    peak, found = peak_of(lambda: StudyResult.read(study))
    assert found == ["study.group-file"] and peak < BOUND


def test_a_clean_study_reads_under_the_same_measurement(container: Path, tmp_path: Path):
    """The control on the other side: an honest study unpacks, with what it allocates well under the bound."""
    tracemalloc.start()
    try:
        unpack_study(container, tmp_path / "out")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < BOUND
    assert len(StudyResult.read(tmp_path / "out")) == 4


# --- a manifest or a recipe of the wrong shapes ------------------------------------------------------------------------------

JUNK: list[Any] = [None, 5, -1, 10**30, "", "x", "../../x", [], {}, True, 1.5, "a" * 1000]


def leaves(document: object, path: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """Every path to a leaf (the first three items of each list), in order."""
    if isinstance(document, dict):
        return [found for key, value in document.items() for found in leaves(value, (*path, key))]
    if isinstance(document, list):
        return [found for index, value in enumerate(document[:3]) for found in leaves(value, (*path, index))]
    return [path]


def set_at(document: Any, path: tuple[object, ...], value: object) -> None:
    for step in path[:-1]:
        document = document[step]
    document[path[-1]] = value


def classified(action: Callable[[], object]) -> str:
    """``refused``, ``accepted``, or ``uncaught <ExceptionType>``: the three things a hostile document can make a read do."""
    try:
        action()
    except MimeError:
        return "refused"
    except Exception as error:  # noqa: BLE001 - this is the harness that names what a read raised that it must not
        return f"uncaught {type(error).__name__}"
    return "accepted"


def test_the_harness_names_an_uncaught_exception():
    """The control: a read that raises a ``TypeError`` is not read as a refusal."""

    def raises() -> None:
        raise TypeError("an uncoded failure")

    assert classified(raises) == "uncaught TypeError"


STUDY_KEYS = ["format", "version", "id", "experiment", "axes", "points", "run", "drop_refused", "dropped", "seed",
              "groups", "members", "lineage", "reading"]
RECIPE_KEYS = ["format", "version", "name", "target", "spec", "options", "reads", "description", "reading"]


@pytest.mark.parametrize("name, key", [("study/study.json", key) for key in STUDY_KEYS]
                         + [("study/figures/trace.json", key) for key in RECIPE_KEYS])
def test_a_hostile_manifest_or_recipe_is_refused_never_raised(container: Path, tmp_path: Path, name: str, key: str):
    """Every leaf of ``study.json`` and of a figure's recipe set to each of a dozen hostile values (a null, a number
    where a string is, a list where a number is, a path out of the folder, a number too large to count), the container's
    records made to agree: unpacking refuses it with a code, or accepts it, and never raises anything else. Not drawn at
    random and never reduced: every leaf, and each key whole, every value."""
    with zipfile.ZipFile(container) as archive:
        original = json.loads(archive.read(name))
    uncaught: list[str] = []
    seen = 0
    for path in [(key,), *[each for each in leaves(original) if each and each[0] == key and each != (key,)]]:
        for junk in JUNK:
            held = json.loads(json.dumps(original))
            try:
                set_at(held, path, junk)
            except (KeyError, IndexError, TypeError):
                continue
            if held == original:
                continue
            forged = rewritten(container, lambda members, manifest, held=held: members.update(
                {name: (json.dumps(held, indent=1, sort_keys=True) + "\n").encode()}))
            out = tmp_path / "out"
            found = classified(lambda: unpack_study(forged, out))
            shutil.rmtree(out, ignore_errors=True)
            seen += 1
            if found.startswith("uncaught"):
                uncaught.append(f"{path} = {str(junk)[:20]!r}: {found}")
    assert seen > 0
    assert uncaught == []


def test_every_key_of_the_manifest_and_the_recipe_is_in_the_lists_above(container: Path):
    with zipfile.ZipFile(container) as archive:
        assert sorted(json.loads(archive.read("study/study.json"))) == sorted(STUDY_KEYS)
        assert sorted(json.loads(archive.read("study/figures/trace.json"))) == sorted(RECIPE_KEYS)


@pytest.mark.parametrize("change, said", [
    (lambda held: held["groups"][0].update(compiles="x"), "compiles"),
    (lambda held: held["members"][0].update(point=None), "point"),
    (lambda held: held["groups"][0].update(index=[]), "index"),
    (lambda held: held["points"][0]["values"][0].update(unit="5"), "unit"),
])
def test_the_shapes_the_review_found_are_refused_naming_the_manifest(
        container: Path, tmp_path: Path, change: Callable[[dict[str, Any]], object], said: str):
    def edit(members: dict[str, bytes], manifest: dict[str, object]) -> None:
        held = json.loads(members["study/study.json"])
        change(held)
        members["study/study.json"] = (json.dumps(held, indent=1, sort_keys=True) + "\n").encode()

    refused = refusal_of(lambda: unpack_study(rewritten(container, edit), tmp_path / "out"))
    assert {each.code for each in refused.refusals} <= {"study.manifest", "study.group-file", "bundle.manifest"}
    assert said in str(refused)


def test_a_recipe_that_reads_a_number_is_refused(container: Path, tmp_path: Path):
    def edit(members: dict[str, bytes], manifest: dict[str, object]) -> None:
        held = json.loads(members["study/figures/trace.json"])
        held["reads"] = [5]
        members["study/figures/trace.json"] = (json.dumps(held, indent=1, sort_keys=True) + "\n").encode()

    assert "plot.recipe" in codes_of(lambda: unpack_study(rewritten(container, edit), tmp_path / "out"))


# --- a name that is a shell command --------------------------------------------------------------------------------------------

HOSTILE = "stopped; touch pwned $(id) 'q'"


def command_in(text: str, start: str) -> list[str]:
    """The words of the first backticked command in ``text`` that starts with ``start``."""
    for part in text.split("`"):
        if part.startswith(start):
            return shlex.split(part)
    raise AssertionError(f"no command starting {start!r} in {text!r}")


def test_a_stopped_studys_command_holds_its_name_as_one_word(study: Path, tmp_path: Path):
    hostile = study.with_name(HOSTILE)
    study.rename(hostile)
    path = hostile / "study.json"
    held = json.loads(path.read_text(encoding="utf-8"))
    held["members"][1]["status"] = "pending"
    path.write_text(json.dumps(held, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    refused = refusal_of(lambda: pack(hostile))
    found = [each for each in refused.refusals if each.code == "study.incomplete"]
    assert found
    words = command_in(found[0].action, "mime study run")
    assert words == ["mime", "study", "run", f"{HOSTILE}/study.mime.yaml", "--out", HOSTILE]


def test_a_study_whose_base_is_not_its_own_gives_its_command_the_name_as_one_word(study: Path):
    hostile = study.with_name(HOSTILE)
    study.rename(hostile)
    PLANTS["input-hash"].folder(hostile, random.Random(3))  # type: ignore[misc]
    refused = refusal_of(lambda: pack(hostile))
    found = [each for each in refused.refusals if each.code == "study.input-hash"]
    assert found
    assert command_in(found[0].action, "mime study run")[3] == f"{HOSTILE}/study.mime.yaml"


def test_the_incomplete_refusal_quotes_a_name_given_to_it_directly():
    document = {"members": [{"point": 0, "status": "pending"}]}
    refusal = incomplete_problem(document, HOSTILE)
    assert refusal is not None
    assert command_in(refusal.action, "mime study run")[3] == f"{HOSTILE}/study.mime.yaml"


def test_a_card_of_a_study_with_a_hostile_name_holds_each_command_to_one_word(
        study: Path, capsys: pytest.CaptureFixture[str]):
    hostile = study.with_name("two words; and $(id)")
    study.rename(hostile)
    assert main(["card", str(hostile)]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith(("mime verify", "mime study run"))]
    assert len(lines) == 2
    for line in lines:
        words = shlex.split(line)
        assert words[:2] in (["mime", "verify"], ["mime", "study"])
        assert not any(";" in word and " " not in word for word in words), words
    assert shlex.split(lines[0])[2] == "./two words; and $(id)/"


def test_a_container_named_as_another_kind_gives_its_pull_command_the_name_as_one_word(
        container: Path, tmp_path: Path):
    named = container.with_name("x y;z.mimebundle")
    shutil.copy(container, named)
    refusal = [each for each in refusal_of(lambda: pull(named, into=tmp_path / "out")).refusals
               if each.code == "container.kind"][0]
    assert command_in(refusal.action, "mime pull") == ["mime", "pull", "x y;z.mimestudy"]


# --- a destination that is a link ----------------------------------------------------------------------------------------------


def a_link_to_an_empty_folder(tmp_path: Path) -> Path:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("this platform does not let a test make a symbolic link")
    return link


def test_a_link_is_not_a_destination_for_a_study_and_nothing_is_staged(container: Path, tmp_path: Path):
    link = a_link_to_an_empty_folder(tmp_path)
    before = sorted(os.listdir(tmp_path))
    assert codes_of(lambda: unpack_study(container, link)) == ["pull.destination"]
    assert codes_of(lambda: pull(container, into=link, fetch_assets=False)) == ["pull.destination"]
    assert sorted(os.listdir(tmp_path)) == before and os.listdir(tmp_path / "target") == []


def test_a_link_is_not_a_destination_for_a_bundle_either(tmp_path: Path):
    bundle = pack(saved_toy(tmp_path / "saved"))
    link = a_link_to_an_empty_folder(tmp_path)
    before = sorted(os.listdir(tmp_path))
    assert codes_of(lambda: unpack(bundle, link)) == ["pull.destination"]
    assert codes_of(lambda: pull(bundle, into=link, fetch_assets=False)) == ["pull.destination"]
    assert sorted(os.listdir(tmp_path)) == before and os.listdir(tmp_path / "target") == []


def test_an_empty_folder_is_still_a_destination(container: Path, tmp_path: Path):
    """The control: the refusal is of the link, not of an existing empty folder."""
    (tmp_path / "empty").mkdir()
    assert unpack_study(container, tmp_path / "empty").folder == tmp_path / "empty"
