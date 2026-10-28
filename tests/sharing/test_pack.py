"""``mime pack``: a saved folder becomes one bundle file, deterministic, that refuses to carry a path of this machine or
a token MIME holds, and refuses a folder that does not load.

The controls plant the error each refusal guards against (the scan for a path skipped, the scan for a token skipped,
the load check skipped) and the refusals must stop being refusals, every one of twenty runs.
"""

import json
import os
import secrets
import shutil
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import script_a as script_a_module
from bundles import contents, manifest_of
from script_a import script_a

from mime import auth
from mime.build import BuildRefused
from mime.cli import main
from mime.experiment import Experiment
from mime.saved import BUNDLE_SUFFIX, LoadRefused, _bundle, pack
from mime.testing.toys import full_toy


@pytest.fixture(scope="module")
def saved_toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("toy") / "full_toy"
    full_toy().save(folder)
    return folder


@pytest.fixture
def folder(saved_toy: Path, tmp_path: Path) -> Path:
    """A copy of the saved toy to edit."""
    return Path(shutil.copytree(saved_toy, tmp_path / "work" / "full_toy"))


#: The plants are built from parts, so that this file holds no path the repository's private-path scan reads as a home
#: directory.
HOME_ROOT = "/" + "home"
USERS_ROOT = "/" + "Users"
ALICE = f"{HOME_ROOT}/alice/lab/run3"


def refused_codes(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def a_token() -> str:
    return "tok" + secrets.token_urlsafe(30)


# --- what a bundle is ------------------------------------------------------------------------------------------------


def test_a_bundle_holds_the_folder_and_a_manifest_that_describes_it(folder: Path):
    bundle = pack(folder)
    assert bundle == folder.with_name("full_toy" + BUNDLE_SUFFIX)
    with zipfile.ZipFile(bundle) as archive:
        names = [info.filename for info in archive.infolist()]
        assert names[0] == "manifest.json" and names[1:] == sorted(names[1:])
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        manifest = json.loads(archive.read("manifest.json"))
    held = contents(bundle)
    on_disk = {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    assert held == on_disk and set(on_disk) >= {"experiment.mime.yaml", "mime.lock", "reference.npz", "scene/scene.usda"}
    assert manifest == manifest_of(on_disk, "full_toy")
    assert manifest["format"] == "mime.bundle/1" and manifest["assets"] == []


def test_the_same_folder_makes_the_same_bytes(folder: Path, tmp_path: Path):
    first = pack(folder, out=tmp_path / "one.mimebundle")
    second = pack(folder, out=tmp_path / "elsewhere" / "two.mimebundle")
    assert first.read_bytes() == second.read_bytes()
    copy = Path(shutil.copytree(folder, tmp_path / "copy" / "full_toy"))
    for path in sorted(copy.rglob("*")):
        os.utime(path, (1.9e9, 1.9e9))
    os.chmod(copy / "mime.lock", 0o600)
    third = pack(copy, out=tmp_path / "three.mimebundle")
    assert third.read_bytes() == first.read_bytes(), "a modification time or a mode reached the bundle"


def test_a_changed_file_makes_other_bytes_and_another_bundle_hash(folder: Path, tmp_path: Path):
    first = pack(folder, out=tmp_path / "one.mimebundle")
    (folder / "notes.txt").write_text("a note", encoding="utf-8")
    second = pack(folder, out=tmp_path / "two.mimebundle")
    hashes = [json.loads(zipfile.ZipFile(each).read("manifest.json"))["bundle_hash"] for each in (first, second)]
    assert first.read_bytes() != second.read_bytes() and hashes[0] != hashes[1]


def test_an_empty_folder_and_a_link_and_a_pipe_are_not_carried(folder: Path):
    (folder / "empty").mkdir()
    (folder / "link").symlink_to(folder / "mime.lock")
    os.mkfifo(folder / "pipe")
    assert refused_codes(lambda: pack(folder)) == ["bundle.special-file", "bundle.special-file"]
    (folder / "link").unlink()
    (folder / "pipe").unlink()
    assert "empty" not in "".join(contents(pack(folder)))


# --- a folder that does not load ---------------------------------------------------------------------------------------


def test_a_folder_that_does_not_load_is_refused_and_no_bundle_is_written(folder: Path):
    (folder / "experiment.mime.yaml").write_text("schema: mime.document/1\n", encoding="utf-8")
    with pytest.raises(LoadRefused):
        pack(folder)
    assert not list(folder.parent.glob("*" + BUNDLE_SUFFIX)) and not list(folder.parent.glob(".*"))


def an_edited_lock_is_refused(root: Path, folder: Path) -> None:
    lock = json.loads((folder / "mime.lock").read_text(encoding="utf-8"))
    lock["hashes"]["input"] = "0" * 64
    (folder / "mime.lock").write_text(json.dumps(lock), encoding="utf-8")
    try:
        pack(folder)
    except LoadRefused:
        assert not list(folder.parent.glob("*" + BUNDLE_SUFFIX))
        return
    raise AssertionError("a folder whose lock does not match its document was packed")


def test_an_edited_lock_is_refused(tmp_path: Path, folder: Path):
    an_edited_lock_is_refused(tmp_path, folder)


def test_the_bundle_cannot_be_written_inside_the_folder(folder: Path):
    assert refused_codes(lambda: pack(folder, out=folder / "in.mimebundle")) == ["bundle.output-inside"]
    assert refused_codes(lambda: pack(folder, out=folder)) == ["bundle.output-inside"]


# --- a path of the machine -----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pathed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Script A saved with a home path in the reason of a value it declares, which reaches the document and the lock."""
    folder = tmp_path_factory.mktemp("pathed") / "script_a"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(script_a_module, "REASON", f"measured by alice at {ALICE}")
        script_a().save(folder)
    return folder


def a_home_path_in_a_document_is_refused(root: Path, folder: Path) -> None:
    assert ALICE in (folder / "experiment.mime.yaml").read_text(encoding="utf-8")
    Experiment.load(folder)
    try:
        pack(folder)
    except BuildRefused as refused:
        assert set(refused.codes()) == {"bundle.machine-path"}  # the document, and the lock's record of its reason
        assert "experiment.mime.yaml" in str(refused) and "alice" not in str(refused), "the path is quoted"
        return
    raise AssertionError("a document holding /home/alice was packed")


def test_a_home_path_in_a_document_is_refused(tmp_path: Path, pathed: Path):
    a_home_path_in_a_document_is_refused(tmp_path, pathed)


@pytest.mark.parametrize("text", [f"see {HOME_ROOT}/alice/x".encode(), f"in {USERS_ROOT}/bob/work".encode(),
                                  b"/root/.cache/y", b"at /tmp/run/z", b"C:\\Users\\carol\\data",
                                  f"D:{USERS_ROOT}/dave/data".encode(), b"/var/lib/q", b"/mnt/disk/r"])
def test_each_form_of_machine_path_is_refused(folder: Path, text: bytes):
    (folder / "notes.txt").write_bytes(b"before " + text + b" after")
    assert refused_codes(lambda: pack(folder)) == ["bundle.machine-path"]


@pytest.mark.parametrize("text", [b"relative/home/alice", b"a/tmp/x", b"home/alice", b"the /homes of the free",
                                  b"https://example.org/home/x", b"/usr/lib/python"])
def test_a_text_that_is_not_a_path_of_the_machine_is_carried(folder: Path, text: bytes):
    (folder / "notes.txt").write_bytes(text)
    assert "notes.txt" in contents(pack(folder))


def test_the_users_home_directory_is_refused_wherever_it_is(folder: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/data/people/alice")
    (folder / "notes.txt").write_bytes(b"run by /data/people/alice in march")
    assert refused_codes(lambda: pack(folder)) == ["bundle.machine-path"]


def test_the_root_as_home_is_not_a_reason_to_refuse(folder: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/")
    (folder / "notes.txt").write_bytes(b"a / b")
    assert "notes.txt" in contents(pack(folder))


def test_the_pattern_is_the_documentation_generators(folder: Path):
    from mime.docs._build import _ABSOLUTE_PATH

    assert _ABSOLUTE_PATH.pattern.encode("utf-8") == _bundle._MACHINE_PATH.pattern


# --- a token MIME holds -----------------------------------------------------------------------------------------------------


def a_stored_token_in_a_record_file_is_refused(root: Path, folder: Path) -> None:
    secret = a_token()
    auth.login("zenodo", secret)
    (folder / "records").mkdir(exist_ok=True)
    (folder / "records" / "run.log").write_bytes(b"token=" + secret.encode() + b"\n")
    try:
        pack(folder)
    except BuildRefused as refused:
        assert refused.codes() == ["bundle.credential"]
        assert "records/run.log" in str(refused) and secret not in str(refused) and secret[:12] not in str(refused)
        return
    raise AssertionError("a file holding a stored token was packed")


def test_a_stored_token_in_a_record_file_is_refused(tmp_path: Path, folder: Path):
    a_stored_token_in_a_record_file_is_refused(tmp_path, folder)


def test_a_token_from_the_environment_is_refused_too_in_any_file(folder: Path, monkeypatch: pytest.MonkeyPatch):
    secret = a_token()
    monkeypatch.setenv("ZENODO_SANDBOX_TOKEN", secret)
    (folder / "scene" / "layers").mkdir(parents=True, exist_ok=True)
    (folder / "scene" / "layers" / "a.usda").write_bytes(b"#usda\n# " + secret.encode())
    assert refused_codes(lambda: pack(folder)) == ["bundle.credential"]


def test_a_token_in_a_binary_file_is_found(folder: Path):
    secret = a_token()
    auth.login("zenodo-sandbox", secret)
    (folder / "blob.bin").write_bytes(os.urandom(500) + secret.encode() + os.urandom(500))
    assert refused_codes(lambda: pack(folder)) == ["bundle.credential"]


def test_a_value_too_short_to_be_a_token_does_not_block_a_pack(folder: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZENODO_TOKEN", "abc")
    assert pack(folder).is_file()


def test_every_reason_is_given_at_once(folder: Path):
    secret = a_token()
    auth.login("zenodo", secret)
    (folder / "a.txt").write_text(f"{HOME_ROOT}/alice/x " + secret, encoding="utf-8")
    (folder / "b.txt").write_text("/tmp/y", encoding="utf-8")
    refusal = refused_codes(lambda: pack(folder))
    assert sorted(refusal) == ["bundle.credential", "bundle.machine-path", "bundle.machine-path"]


# --- a large file -------------------------------------------------------------------------------------------------------------


LIMIT = 20_000


def lock_with_asset(folder: Path, name: str, payload: bytes, url: str) -> dict[str, object]:
    from mime.assets import AssetRef

    record = AssetRef(name, _bundle.sha256_of(payload), len(payload), "CC0-1.0", (url,)).to_record()
    lock = json.loads((folder / "mime.lock").read_text(encoding="utf-8"))
    lock["assets"] = [record]
    (folder / "mime.lock").write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return dict(record)


def test_a_file_over_the_limit_is_refused_unless_it_is_an_asset_the_lock_references(folder: Path,
                                                                                     monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_bundle, "INLINE_LIMIT_BYTES", LIMIT)
    big = os.urandom(LIMIT + 1000)
    (folder / "table.bin").write_bytes(big)
    refusal = refused_codes(lambda: pack(folder))
    assert refusal == ["bundle.file-too-large"]
    record = lock_with_asset(folder, "table", big, "https://example.org/table.bin")
    bundle = pack(folder)
    assert "table.bin" not in contents(bundle), "a referenced large file was copied into the bundle"
    manifest = json.loads(zipfile.ZipFile(bundle).read("manifest.json"))
    assert manifest["assets"] == [record]
    assert big not in bundle.read_bytes()


def test_the_limit_is_a_named_constant_a_bundle_stays_under():
    assert _bundle.INLINE_LIMIT_BYTES == 50_000_000


# --- the command ----------------------------------------------------------------------------------------------------------------


def test_the_command_prints_the_file_its_size_and_its_hash(folder: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["pack", str(folder), "--out", str(folder.parent / "x.mimebundle")]) == 0
    out = capsys.readouterr().out
    import hashlib

    assert "x.mimebundle" in out and hashlib.sha256((folder.parent / "x.mimebundle").read_bytes()).hexdigest() in out


def test_the_command_exits_two_on_a_refusal(folder: Path, capsys: pytest.CaptureFixture[str]):
    (folder / "notes.txt").write_text(f"{HOME_ROOT}/alice/x", encoding="utf-8")
    assert main(["pack", str(folder)]) == 2
    assert "bundle.machine-path" in capsys.readouterr().err


# --- the controls ----------------------------------------------------------------------------------------------------------------


@contextmanager
def path_scan_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "_MACHINE_PATH", _never())
        patch.setattr(_bundle, "_WINDOWS_HOME", _never())
        patch.setattr(os.path, "expanduser", lambda path: "/")
        yield


def _never() -> "_bundle.re.Pattern[bytes]":
    return _bundle.re.compile(rb"(?!)")


@contextmanager
def credential_scan_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "held_values", lambda: [])
        yield


@contextmanager
def load_check_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Experiment, "load", classmethod(lambda cls, folder: None))
        yield


RUNS = 20


def planted_runs(tmp_path: Path, scenario: Callable[..., None], plant: Callable[[], object], *,
                 saved: Path | None = None) -> int:
    detected = 0
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("HOME", str(root / "home"))
            patch.setenv("XDG_CONFIG_HOME", str(root / "home" / ".config"))
            try:
                with plant():  # type: ignore[attr-defined]
                    if saved is None:
                        scenario(root)
                    else:
                        scenario(root, Path(shutil.copytree(saved, root / "work" / saved.name)))
            except AssertionError:
                detected += 1
    return detected


def test_control_a_skipped_path_scan_is_detected_every_run(tmp_path: Path, pathed: Path):
    assert planted_runs(tmp_path, a_home_path_in_a_document_is_refused, path_scan_skipped, saved=pathed) == RUNS


def test_control_a_skipped_token_scan_is_detected_every_run(tmp_path: Path, saved_toy: Path):
    assert planted_runs(tmp_path, a_stored_token_in_a_record_file_is_refused, credential_scan_skipped,
                        saved=saved_toy) == RUNS


def test_control_a_skipped_load_check_is_detected_every_run(tmp_path: Path, saved_toy: Path):
    assert planted_runs(tmp_path, an_edited_lock_is_refused, load_check_skipped, saved=saved_toy) == RUNS


def test_the_scenarios_pass_unplanted(tmp_path: Path, saved_toy: Path, pathed: Path):
    a_home_path_in_a_document_is_refused(tmp_path, pathed)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
        a_stored_token_in_a_record_file_is_refused(tmp_path, Path(shutil.copytree(saved_toy, tmp_path / "w" / "full_toy")))
