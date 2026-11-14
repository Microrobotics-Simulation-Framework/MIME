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
import struct
import tracemalloc
import warnings
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mime import _npz
from mime._npz import StoredNpz, check_entries
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


def refusal_of(action: Callable[[], object]) -> MimeError:
    try:
        action()
    except MimeError as refused:
        return refused
    raise AssertionError("not refused")


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


# --- the other ways a file is not an NPZ MIME wrote ---------------------------------------------------------------------------


def npy(array: np.ndarray, *, pickled: bool = False) -> bytes:
    held = io.BytesIO()
    np.lib.format.write_array(held, array, allow_pickle=pickled)
    return held.getvalue()


def stored_zip(entries: list[tuple[str, bytes]]) -> bytes:
    """A zip of stored entries, a name written twice as the caller lists it (the standard library warns of one)."""
    out = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
            for name, data in entries:
                archive.writestr(zipfile.ZipInfo(name, EPOCH), data)
    return out.getvalue()


def declared_only(count: int) -> bytes:
    """The array header of ``count`` bytes, and a few bytes after it: an entry that names what it does not hold."""
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "|u1", "fortran_order": False, "shape": (count,)})
    return header.getvalue() + bytes(64)


def overlapping_npz() -> bytes:
    """Four entries whose local headers each sit inside the previous entry's data, every entry's data running to the end of
    the file: each is a well-formed array of the bytes its header names, with the CRC its record gives it, and together they
    overlap."""
    segment, names = 4096, [f"e{i}.npy" for i in range(4)]
    total = segment * len(names)
    body = bytearray(total)

    def array_header(count: int) -> bytes:
        held = io.BytesIO()
        np.lib.format.write_array_header_1_0(held, {"descr": "|u1", "fortran_order": False, "shape": (count,)})
        return held.getvalue()

    def local(name: str, size: int, crc: int) -> bytes:
        text = name.encode()
        return struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0x21, crc, size, size, len(text), 0) + text

    layout = []
    for index, name in enumerate(names):
        offset = index * segment
        start = offset + len(local(name, 0, 0))
        size = total - start
        count = size
        for _ in range(5):
            count = size - len(array_header(count))
        assert len(array_header(count)) + count == size
        body[start:start + len(array_header(count))] = array_header(count)
        layout.append((name, offset, start, size))
    crcs: dict[str, int] = {}
    for name, offset, start, size in reversed(layout):  # an entry's data holds the later entries' headers, which hold their CRCs
        crcs[name] = zlib.crc32(bytes(body[start:start + size]))
        body[offset:start] = local(name, size, crcs[name])
    directory = b"".join(
        struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 0, 0, 0x21, crcs[name], size, size, len(name.encode()), 0,
                    0, 0, 0, 0, offset) + name.encode() for name, offset, _, size in layout)
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(layout), len(layout), len(directory), len(body), 0)
    return bytes(body) + directory + end


HOSTILE_FILES: dict[str, Callable[[], bytes]] = {
    "pickled": lambda: stored_zip([("a.npy", npy(np.array([{"x": 1}, object()], dtype=object), pickled=True))]),
    "a name twice, the bad entry first": lambda: stored_zip([("a.npy", declared_only(DECLARED)),
                                                             ("a.npy", npy(np.arange(4.0)))]),
    "a name twice, the bad entry last": lambda: stored_zip([("a.npy", npy(np.arange(4.0))),
                                                            ("a.npy", declared_only(DECLARED))]),
    "a name twice, both entries good": lambda: stored_zip([("a.npy", npy(np.arange(4.0))),
                                                           ("a.npy", npy(np.arange(5.0)))]),
    "overlapping entries": overlapping_npz,
}


@pytest.mark.parametrize("kind", sorted(HOSTILE_FILES))
def test_a_reference_npz_that_is_not_one_mime_wrote_is_refused_with_its_code(saved: Path, tmp_path: Path, kind: str):
    folder = Path(shutil.copytree(saved, tmp_path / "full_toy"))
    (folder / "reference.npz").write_bytes(HOSTILE_FILES[kind]())
    peak, found = peak_of(lambda: read_folder(folder))
    assert found == ["load.reference"] and peak < BOUND


@pytest.mark.parametrize("kind", sorted(HOSTILE_FILES))
def test_a_run_whose_outputs_are_not_an_npz_mime_wrote_is_refused_with_its_code(run: Path, tmp_path: Path, kind: str):
    folder = Path(shutil.copytree(run, tmp_path / "run"))
    (folder / RUN_OUTPUTS).write_bytes(HOSTILE_FILES[kind]())
    peak, found = peak_of(lambda: RunResult.read(folder))
    assert found == ["result.not-a-run-folder"] and peak < BOUND


def blind_to_the_standard_librarys_overlap_check(archive: zipfile.ZipFile) -> None:
    """What Python 3.12.0 and 3.12.1 are: no refusal of overlapped entries (the later versions' check, CVE-2024-0450, lives in
    each entry's ``_end_offset``, which a version without it has no slot for)."""
    for info in archive.infolist():
        if hasattr(info, "_end_offset"):
            info._end_offset = None  # type: ignore[attr-defined]


def test_overlapping_entries_are_refused_by_the_readers_own_check_whatever_the_standard_library_does(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "overlap.npz"
    path.write_bytes(overlapping_npz())
    with zipfile.ZipFile(path) as archive:
        blind_to_the_standard_librarys_overlap_check(archive)
        with pytest.raises(ValueError, match="overlap"):
            check_entries(archive, path.stat().st_size)
    # The control: each entry is, alone, a well-formed array of the bytes its header names, so without the reader's own layout
    # check nothing but the standard library's version-dependent one would refuse the file.
    monkeypatch.setattr(_npz, "_layout", lambda archive, size: None)
    with zipfile.ZipFile(path) as archive:
        blind_to_the_standard_librarys_overlap_check(archive)
        check_entries(archive, path.stat().st_size)


def test_an_entry_that_runs_past_its_central_directory_is_refused(tmp_path: Path):
    data = bytearray(stored_zip([("a.npy", npy(np.arange(4.0)))]))
    (start,) = [i for i in range(len(data)) if data[i:i + 4] == b"PK\x01\x02"]
    struct.pack_into("<I", data, start + 20, 10**6)  # the central directory's compressed size, past the file
    path = tmp_path / "past.npz"
    path.write_bytes(bytes(data))
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(ValueError, match="runs past"):
            check_entries(archive, path.stat().st_size)


def test_a_clean_npz_passes_the_layout_check(saved: Path, run: Path):
    for path in (saved / "reference.npz", run / RUN_OUTPUTS):
        with zipfile.ZipFile(path) as archive:
            check_entries(archive, path.stat().st_size)


# --- the file the lock names for the reference outputs ------------------------------------------------------------------------


def named_in_the_lock(folder: Path, name: str) -> None:
    path = folder / "mime.lock"
    lock = json.loads(path.read_text(encoding="utf-8"))
    lock["reference"]["file"] = name
    path.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def test_a_reference_file_outside_the_folder_is_refused_by_name_before_it_is_read(saved: Path, tmp_path: Path):
    """A name that climbs out of the folder, or is absolute, is refused by the lock's own shape check (``load.lock-shape``,
    which leaves the lock unread, so no file it names is opened); what is outside holds a bomb, and nothing is allocated."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    for number, name in enumerate(("../elsewhere/reference.npz", str(outside / "reference.npz"))):
        folder = Path(shutil.copytree(saved, tmp_path / "work" / f"full_toy{number}"))
        (outside / "reference.npz").write_bytes(bombed((folder / "reference.npz").read_bytes(),
                                                       first_array(folder / "reference.npz"), compressed=True))
        named_in_the_lock(folder, name)
        tracemalloc.start()
        try:
            found = refusal_of(lambda: read_folder(folder)).refusals
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        shape = [each for each in found if each.code == "load.lock-shape"]
        assert len(shape) == 1 and name in shape[0].message and peak < BOUND
        assert not [each for each in found if each.code in ("load.reference", "load.reference-changed")]


def test_a_reference_file_that_is_a_link_leaving_the_folder_is_refused_by_name(saved: Path, tmp_path: Path):
    """The file is the right one (its bytes are the folder's own): only that it is reached through a link out of the folder
    refuses it."""
    folder = Path(shutil.copytree(saved, tmp_path / "full_toy"))
    elsewhere = tmp_path / "elsewhere.npz"
    shutil.move(str(folder / "reference.npz"), elsewhere)
    try:
        (folder / "reference.npz").symlink_to(elsewhere)
    except OSError:
        pytest.skip("this platform does not let a test make a symbolic link")
    found = [each for each in refusal_of(lambda: read_folder(folder)).refusals if each.code == "load.reference"]
    assert len(found) == 1 and "outside the saved folder" in found[0].message


def test_a_reference_file_that_is_a_link_inside_the_folder_still_loads(saved: Path, tmp_path: Path):
    """The control on the refusal above: a link that stays in the folder is not refused."""
    folder = Path(shutil.copytree(saved, tmp_path / "full_toy"))
    shutil.move(str(folder / "reference.npz"), folder / "reference_held.npz")
    try:
        (folder / "reference.npz").symlink_to("reference_held.npz")
    except OSError:
        pytest.skip("this platform does not let a test make a symbolic link")
    assert read_folder(folder).lock["reference"]["file"] == "reference.npz"


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
