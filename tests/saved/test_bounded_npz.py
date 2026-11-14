"""An NPZ handed to MIME that names more than it holds (a security fix for hostile folders): the reference outputs of a saved
folder (``reference.npz``) and a run's arrays (``outputs.npz``, or the NPZ ``to_npz`` writes) are read through
``mime._npz``, which refuses an entry that is compressed, larger than the file, or whose header names other bytes than it
holds before NumPy allocates what the header names. A deflated entry of a few kilobytes named gigabytes: a nested zip
bomb, taken whole by ``np.load``.

The memory a read allocates is measured with ``tracemalloc`` (NumPy reports its arrays to it), against a control that reads
the same entry with no bound and must show the allocation, so that a measurement too coarse to see it fails. A hash cannot
stand in for the bound where the record is of the arrays (a saved folder's lock records the arrays' hash, not the file's):
the bound is the defence, and the checks are on every entry whatever the hash says.

Beside them: every NPZ MIME writes is stored, never compressed (nothing the bound refuses is something MIME writes), and
every key of a run's manifest, set to each of a dozen hostile values, is refused or read, never raised."""

from __future__ import annotations

import io
import json
import shutil
import tracemalloc
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mime._npz import StoredNpz
from mime.build import MimeError
from mime.cli import main
from mime.experiment import RUN_OUTPUTS
from mime.results import RUN_MANIFEST, RunResult
from mime.saved import read_folder
from mime.testing.toys import full_toy
from mime.units import s

DECLARED = 128 * 2**20
#: What a refused read may allocate: far below the entry's declared size, above what loading a toy folder holds.
BOUND = 32 * 2**20
EPOCH = (1980, 1, 1, 0, 0, 0)


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("saved") / "full_toy"
    full_toy().save(folder)
    return folder


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("run") / "run"
    full_toy().build().run_to(folder, s(0.04))
    return folder


def bombed(original: bytes, victim: str, *, compressed: bool) -> bytes:
    """``original``'s NPZ with the entry ``victim`` replaced by an array of ``DECLARED`` bytes: deflated and holding them
    (a few hundred kilobytes of zeros, ``compressed``), or stored and holding a few bytes of the header's claim."""
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "|u1", "fortran_order": False, "shape": (DECLARED,)})
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as target:
        assert victim + ".npy" in source.namelist()
        for info in source.infolist():
            if info.filename != victim + ".npy":
                target.writestr(info, source.read(info))
                continue
            held = zipfile.ZipInfo(info.filename, EPOCH)
            held.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
            with target.open(held, "w", force_zip64=True) as entry:
                entry.write(header.getvalue())
                left = DECLARED if compressed else 64
                while left > 0:
                    entry.write(bytes(min(left, 1 << 22)))
                    left -= 1 << 22
    return out.getvalue()


def refusal_codes(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except MimeError as refused:
        return [each.code for each in refused.refusals]
    raise AssertionError("not refused")


def peak_of(action: Callable[[], object]) -> tuple[int, list[str]]:
    """What ``action`` refused with, and the most memory it held at once."""
    tracemalloc.start()
    try:
        found = refusal_codes(action)
        return tracemalloc.get_traced_memory()[1], found
    finally:
        tracemalloc.stop()


def first_array(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()[0].removesuffix(".npy")


def test_the_measurement_sees_the_allocation_it_bounds(run: Path):
    """The control: the same entry read with no bound (NumPy's own reader, which every read here used before) allocates what
    its header names, and the measurement shows it."""
    victim = first_array(run / RUN_OUTPUTS)
    held = bombed((run / RUN_OUTPUTS).read_bytes(), victim, compressed=True)
    tracemalloc.start()
    try:
        with zipfile.ZipFile(io.BytesIO(held)) as archive, archive.open(victim + ".npy") as entry:
            np.lib.format.read_array(entry, allow_pickle=False)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak >= DECLARED > BOUND


# --- reference.npz of a saved folder -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("compressed", [True, False], ids=["deflated", "stored-lying-header"])
def test_a_reference_npz_that_names_more_than_it_holds_is_refused_before_it_is_allocated(
        saved: Path, tmp_path: Path, compressed: bool):
    folder = Path(shutil.copytree(saved, tmp_path / "full_toy"))
    path = folder / "reference.npz"
    path.write_bytes(bombed(path.read_bytes(), first_array(path), compressed=compressed))
    peak, found = peak_of(lambda: read_folder(folder))
    assert found == ["load.reference"] and peak < BOUND


def test_mime_verify_of_a_folder_with_such_a_reference_refuses_it_and_allocates_no_more(
        saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = Path(shutil.copytree(saved, tmp_path / "full_toy"))
    path = folder / "reference.npz"
    path.write_bytes(bombed(path.read_bytes(), first_array(path), compressed=True))
    tracemalloc.start()
    try:
        code = main(["verify", str(folder)])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert code != 0 and "load.reference" in capsys.readouterr().err and peak < BOUND


def test_a_clean_saved_folder_still_loads_under_the_same_measurement(saved: Path):
    tracemalloc.start()
    try:
        form = read_folder(saved)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < BOUND and form.lock["reference"]["file"] == "reference.npz"


# --- outputs.npz of a run, and the NPZ to_npz writes ------------------------------------------------------------------------


@pytest.mark.parametrize("compressed", [True, False], ids=["deflated", "stored-lying-header"])
def test_a_run_folder_whose_outputs_name_more_than_they_hold_is_refused_before_it_is_allocated(
        run: Path, tmp_path: Path, compressed: bool):
    folder = Path(shutil.copytree(run, tmp_path / "run"))
    path = folder / RUN_OUTPUTS
    path.write_bytes(bombed(path.read_bytes(), first_array(path), compressed=compressed))
    peak, found = peak_of(lambda: RunResult.read(folder))
    assert found == ["result.not-a-run-folder"] and peak < BOUND


def test_a_result_npz_whose_arrays_name_more_than_they_hold_is_refused_too(run: Path, tmp_path: Path):
    npz = RunResult.read(run).to_npz(tmp_path / "result.npz")
    npz.write_bytes(bombed(npz.read_bytes(), first_array(npz), compressed=True))
    peak, found = peak_of(lambda: RunResult.read(npz))
    assert found == ["result.not-a-run-folder"] and peak < BOUND


def test_a_clean_run_reads_under_the_same_measurement(run: Path):
    tracemalloc.start()
    try:
        result = RunResult.read(run)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < BOUND and result.steps > 0


# --- nothing MIME writes is refused ------------------------------------------------------------------------------------------


def test_every_npz_mime_writes_is_stored_and_every_one_reads_through_the_bound(saved: Path, run: Path, tmp_path: Path):
    written = [saved / "reference.npz", run / RUN_OUTPUTS, RunResult.read(run).to_npz(tmp_path / "result.npz")]
    for path in written:
        with zipfile.ZipFile(path) as archive:
            assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}, path
        with StoredNpz(path) as held, np.load(path, allow_pickle=False) as plain:
            assert held.files == plain.files
            for key in held.files:
                assert np.array_equal(held[key], plain[key]) and held[key].dtype == plain[key].dtype


def test_no_source_file_of_mime_writes_a_compressed_npz():
    """The writers are ``np.savez`` (entries stored); a ``savez_compressed`` anywhere in the package would write a file the
    bound refuses."""
    root = Path(__file__).resolve().parents[2] / "src" / "mime"
    assert [path.relative_to(root).as_posix() for path in root.rglob("*.py")
            if "savez_compressed" in path.read_text(encoding="utf-8")] == []


# --- a run's manifest of the wrong shapes ------------------------------------------------------------------------------------

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


#: What a public call turns a raw exception into: a "MIME failed" with no code of its own, which is the failure and no refusal.
UNCODED = frozenset({"error.uncoded", "error.unrecognised"})


def classified(action: Callable[[], object]) -> str:
    """``refused``, ``accepted``, or ``uncaught <ExceptionType>``: the three things a hostile document can make a read do."""
    try:
        action()
    except MimeError as refused:
        uncoded = [each.code for each in refused.refusals if each.code in UNCODED]
        return f"uncaught {uncoded[0]}" if uncoded else "refused"
    except Exception as error:  # noqa: BLE001 - the harness that names what a read raised that it must not
        return f"uncaught {type(error).__name__}"
    return "accepted"


def test_the_harness_names_an_uncaught_exception():
    """The control: a read that raises a ``TypeError`` is not read as a refusal."""

    def raises() -> None:
        raise TypeError("an uncoded failure")

    assert classified(raises) == "uncaught TypeError"


def test_a_manifest_whose_steps_is_a_string_is_a_refusal_and_not_an_internal_failure(run: Path, tmp_path: Path):
    """A manifest a read raises on is turned by ``@public`` into ``error.uncoded`` (MIME failed, report it) when nothing
    refuses it first; the harness names that as uncaught, and the read refuses it with a code."""
    forged = tmp_path / "forged"
    shutil.copytree(run, forged)
    held = json.loads((forged / RUN_MANIFEST).read_text(encoding="utf-8"))
    held["steps"] = "x"
    (forged / RUN_MANIFEST).write_text(json.dumps(held), encoding="utf-8")
    assert classified(lambda: RunResult.read(forged)) == "refused"


#: The keys of a run's manifest, and two a stopped or overridden run adds.
MANIFEST_KEYS = ["clock", "containment", "dt", "end", "every", "experiment", "flags", "format", "history", "observations",
                 "overrides", "reading", "recorded", "run_flags", "stamp", "start", "start_step", "steps", "stopped",
                 "version", "watches"]


def test_the_keys_below_are_the_ones_the_manifest_has(run: Path):
    held = json.loads((run / RUN_MANIFEST).read_text(encoding="utf-8"))
    assert set(held) <= set(MANIFEST_KEYS) and {"format", "version", "experiment", "observations", "flags"} <= set(held)


@pytest.mark.parametrize("key", MANIFEST_KEYS)
def test_a_hostile_run_manifest_is_refused_never_raised(run: Path, tmp_path: Path, key: str):
    """Every leaf of ``outputs.json`` under ``key``, and the key whole, set to each of a dozen hostile values (a null, a number
    where a string is, a list where a number is, a path out of the folder, a number too large to count): the run folder is
    refused with a code or read, and nothing else is raised. Not drawn at random and never reduced: every leaf, every value."""
    original = json.loads((run / RUN_MANIFEST).read_text(encoding="utf-8"))
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
            folder = tmp_path / "forged"
            shutil.rmtree(folder, ignore_errors=True)
            shutil.copytree(run, folder)
            (folder / RUN_MANIFEST).write_text(json.dumps(held), encoding="utf-8")
            found = classified(lambda: RunResult.read(folder))
            seen += 1
            if found.startswith("uncaught"):
                uncaught.append(f"{path} = {str(junk)[:20]!r}: {found}")
    assert seen > 0 or key not in original
    assert uncaught == []


def test_an_outputs_json_that_is_not_json_or_not_an_object_is_refused(run: Path, tmp_path: Path):
    for text in ("{", "[]", "5", "\x00", "[" * 100_000):
        folder = tmp_path / "forged"
        shutil.rmtree(folder, ignore_errors=True)
        shutil.copytree(run, folder)
        (folder / RUN_MANIFEST).write_text(text, encoding="utf-8")
        assert refusal_codes(lambda: RunResult.read(folder)) == ["result.not-a-run-folder"], text
