"""``mime pack``: a saved folder becomes one bundle file, deterministic, that refuses to carry a path of this machine or
a token MIME holds, and refuses a folder that does not load.

The controls plant the error each refusal guards against (the scan for a path skipped, the scan for a token skipped,
the load check skipped) and the refusals must stop being refusals, every one of twenty runs.
"""

import base64
import gzip
import io
import json
import os
import secrets
import shutil
import urllib.parse
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import script_a as script_a_module
from bundles import contents, manifest_of
from script_a import script_a

from mime import credentials
from mime.build import BuildRefused
from mime.cli import main
from mime.experiment import Experiment
from mime.saved import BUNDLE_SUFFIX, EXPANSION_LIMIT_BYTES, LoadRefused, _bundle, pack
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


def put(service: str, value: str) -> Path:
    return credentials.store(service, credentials.Token(service, value))


@contextmanager
def carried(folder: Path, files: dict[str, bytes]) -> Iterator[None]:
    """``files`` written into the folder and, for the length of the block, named by the saved form: the scans are of what
    a bundle would carry, and these tests plant what they are to find in a file that would be carried. (A file the saved
    form does not name is refused before any scan, and the tests of that refusal do not use this.)"""
    for name, data in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    real = _bundle.named_files
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "named_files", lambda form: real(form) | set(files))
        yield


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
    with carried(folder, {"descriptions/notes.txt": b"a note"}):
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
    with carried(folder, {"descriptions/notes.txt": b"before " + text + b" after"}):
        assert refused_codes(lambda: pack(folder)) == ["bundle.machine-path"]


@pytest.mark.parametrize("text", [b"relative/home/alice", b"a/tmp/x", b"home/alice", b"the /homes of the free",
                                  b"https://example.org/home/x", b"/usr/lib/python"])
def test_a_text_that_is_not_a_path_of_the_machine_is_carried(folder: Path, text: bytes):
    with carried(folder, {"descriptions/notes.txt": text}):
        assert "descriptions/notes.txt" in contents(pack(folder))


def test_the_users_home_directory_is_refused_wherever_it_is(folder: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/data/people/alice")
    with carried(folder, {"descriptions/notes.txt": b"run by /data/people/alice in march"}):
        assert refused_codes(lambda: pack(folder)) == ["bundle.machine-path"]


def test_the_root_as_home_is_not_a_reason_to_refuse(folder: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/")
    with carried(folder, {"descriptions/notes.txt": b"a / b"}):
        assert "descriptions/notes.txt" in contents(pack(folder))


def test_the_pattern_is_the_documentation_generators(folder: Path):
    from mime.docs._build import _ABSOLUTE_PATH

    assert _ABSOLUTE_PATH.pattern.encode("utf-8") == _bundle._MACHINE_PATH.pattern


# --- a token MIME holds -----------------------------------------------------------------------------------------------------


def a_stored_token_in_a_record_file_is_refused(root: Path, folder: Path) -> None:
    secret = a_token()
    put("zenodo", secret)
    with carried(folder, {"records/run.log": b"token=" + secret.encode() + b"\n"}):
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
    with carried(folder, {"scene/layers/a.usda": b"#usda\n# " + secret.encode()}):
        assert refused_codes(lambda: pack(folder)) == ["bundle.credential"]


def test_a_token_in_a_binary_file_is_found(folder: Path):
    secret = a_token()
    put("zenodo-sandbox", secret)
    with carried(folder, {"descriptions/blob.bin": os.urandom(500) + secret.encode() + os.urandom(500)}):
        assert refused_codes(lambda: pack(folder)) == ["bundle.credential"]


def test_a_value_too_short_to_be_a_token_does_not_block_a_pack(folder: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZENODO_TOKEN", "abc")
    assert pack(folder).is_file()


def test_every_reason_is_given_at_once(folder: Path):
    secret = a_token()
    put("zenodo", secret)
    with carried(folder, {"descriptions/a.txt": f"{HOME_ROOT}/alice/x {secret}".encode(),
                          "descriptions/b.txt": b"/tmp/y"}):
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
    with carried(folder, {"descriptions/table.bin": big}):
        refusal = refused_codes(lambda: pack(folder))
        assert refusal == ["bundle.file-too-large"]
        record = lock_with_asset(folder, "table", big, "https://example.org/table.bin")
        bundle = pack(folder)
    assert "descriptions/table.bin" not in contents(bundle), "a referenced large file was copied into the bundle"
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
    with carried(folder, {"descriptions/notes.txt": f"{HOME_ROOT}/alice/x".encode()}):
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
        patch.setattr(_bundle, "held_tokens", lambda: [])
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


# --- what the scans find, and what they do not ----------------------------------------------------------------------------
#
# The scans are functions of a file's bytes, so these plant each form straight against ``_scan``; the pack-level tests
# above show the same refusal reaching a person.

def scanned(data: bytes, *, tokens: list[credentials.Token] | None = None) -> list[str]:
    return [each.code for each in _bundle._scan("file", data, tokens or [])]


@pytest.mark.parametrize("text", [
    b"see file:///home/alice/x", b"file:///tmp/x", b"file:////home/alice/x",
    b"/Volumes/Backup/run", b"in /media/alice/disk", b"/run/user/1000/x", b"D:\\data\\x", b"D:\\\\data\\\\x",
    b"d:/lab/x", b"\\\\server\\share", b"\\\\\\\\server\\\\share", b"E:\\Users\\bob\\data"])
def test_each_further_form_of_machine_path_is_refused(text: bytes):
    assert scanned(b"before " + text + b" after") == ["bundle.machine-path"]


@pytest.mark.parametrize("text", [b"/data/x", b"/scratch/run", b"/work/dir", b"~/x", b"$HOME/x", b"https://example.org/a",
                                  b"a:b", b"http://x", b"12:30:45", b"T=3:1", b"\\alpha \\beta", b"C", b"//host/x"])
def test_what_is_not_flagged_on_purpose_is_carried(text: bytes):
    assert scanned(b"before " + text + b" after") == []


def test_the_pattern_set_is_the_generators_rule_and_the_forms_added_to_it():
    from mime.docs._build import _ABSOLUTE_PATH

    assert _ABSOLUTE_PATH.search(b"/home/x".decode()) and _bundle._MACHINE_PATH.search(b"/home/x")
    assert not _ABSOLUTE_PATH.search("file:///home/x") and _bundle._MORE_PATHS[0].search(b"file:///home/x")


def token_forms(value: str) -> dict[str, bytes]:
    raw = value.encode()
    return {
        "as written": raw, "percent-encoded": urllib.parse.quote(value, safe="").encode(),
        "base64": base64.b64encode(raw), "base64url": base64.urlsafe_b64encode(raw),
        "hex": raw.hex().encode(), "upper hex": raw.hex().upper().encode(),
        "json": json.dumps(value).encode(), "utf-16": value.encode("utf-16"), "utf-16 big-endian": value.encode("utf-16-be"),
        "base64 inside a blob, one byte in": base64.b64encode(b"x" + raw + b"yz"),
        "base64 inside a blob, two bytes in": base64.b64encode(b"xy" + raw + b"z"),
        "base64url inside a blob": base64.urlsafe_b64encode(b"abc" + raw + b"\xff\xfe"),
        "gzip": gzip.compress(raw), "gzip of base64": gzip.compress(base64.b64encode(raw)),
        "two gzip members": gzip.compress(b"nothing here") + gzip.compress(raw),
    }


def a_stored_token_in_every_form_is_found(root: Path) -> None:
    secret = "tok+/=? " + secrets.token_urlsafe(30)  # characters an encoder changes (the Token itself checks one word)
    held = credentials.Token("zenodo", secret)
    missed = [name for name, data in token_forms(secret).items()
              if scanned(data if "gzip" in name else os.urandom(40) + data + os.urandom(40), tokens=[held])
              != ["bundle.credential"]]
    assert not missed, f"not found: {missed}"


def test_a_stored_token_is_found_in_every_form_the_search_lists(tmp_path: Path):
    a_stored_token_in_every_form_is_found(tmp_path)


def test_a_token_whose_json_form_differs_from_its_text_is_found():
    for secret in ('tok"quoted\\and-back', "tok-\u00e9-accent-chars-0123456789"):
        data = b'{"k": ' + json.dumps(secret).encode() + b"}"
        assert scanned(data, tokens=[credentials.Token("zenodo", secret)]) == ["bundle.credential"], secret
        data = b'{"k": ' + json.dumps(secret, ensure_ascii=False).encode() + b"}"
        assert scanned(data, tokens=[credentials.Token("zenodo", secret)]) == ["bundle.credential"], secret


def test_a_token_inside_a_compressed_npz_member_and_a_nested_archive_is_found():
    secret = "tok" + secrets.token_urlsafe(30)
    held = [credentials.Token("zenodo", secret)]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a.npy", b"padding " * 50 + secret.encode() + b" more")
    assert secret.encode() not in buffer.getvalue(), "the plant is compressed"
    assert scanned(buffer.getvalue(), tokens=held) == ["bundle.credential"]
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("inner.gz", gzip.compress(b"x" * 100 + secret.encode()))
    assert scanned(nested.getvalue(), tokens=held) == ["bundle.credential"]
    path = io.BytesIO()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a.txt", b"/home/alice/lab")
    assert scanned(path.getvalue()) == ["bundle.machine-path"]


def test_a_compressed_file_that_cannot_be_searched_is_refused_and_not_carried(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_bundle, "EXPANSION_LIMIT_BYTES", 1000)
    assert scanned(gzip.compress(b"\0" * 100_000)) == ["bundle.unscannable"]
    assert scanned(gzip.compress(b"\0" * 500)) == []
    layers = b"deep"
    for _ in range(6):
        layers = gzip.compress(layers)
    assert scanned(layers) == ["bundle.unscannable"]
    assert EXPANSION_LIMIT_BYTES == 200_000_000


def test_the_forms_the_search_does_not_list_are_not_found():
    """The limit written down: a token reversed, upper-cased or split across a newline passes the scan, which is why a
    pack carries only the files a saved folder names."""
    secret = "tok" + secrets.token_urlsafe(30)
    held = [credentials.Token("zenodo", secret)]
    for data in (secret[::-1].encode(), secret.upper().encode(), (secret[:20] + "\n" + secret[20:]).encode(),
                 codecs_rot13(secret)):
        assert scanned(data, tokens=held) == [], data


def codecs_rot13(text: str) -> bytes:
    import codecs

    return codecs.encode(text, "rot13").encode()


@contextmanager
def further_paths_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "_MORE_PATHS", ())
        yield


@contextmanager
def only_the_written_form_searched() -> Iterator[None]:
    from mime.credentials import _token

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_token, "_forms", lambda value: (value.encode(),))
        yield


@contextmanager
def compressed_content_not_searched() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "_inside", lambda data, budget, depth=0: iter(()))
        yield


def a_file_url_path_is_refused(root: Path) -> None:
    assert scanned(b"see file:///home/alice/lab") == ["bundle.machine-path"]
    assert scanned(b"in /Volumes/Backup/x") == ["bundle.machine-path"]
    assert scanned(b"D:\\data\\x") == ["bundle.machine-path"]
    assert scanned(b"\\\\server\\share") == ["bundle.machine-path"]


def a_gzipped_token_is_found(root: Path) -> None:
    secret = "tok" + secrets.token_urlsafe(30)
    assert scanned(gzip.compress(b"x" + secret.encode()), tokens=[credentials.Token("zenodo", secret)]) == [
        "bundle.credential"]


def test_control_further_paths_skipped_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_file_url_path_is_refused, further_paths_skipped) == RUNS


def test_control_only_the_written_form_searched_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_stored_token_in_every_form_is_found, only_the_written_form_searched) == RUNS


def test_control_compressed_content_not_searched_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_gzipped_token_is_found, compressed_content_not_searched) == RUNS


def test_the_new_scenarios_pass_unplanted(tmp_path: Path):
    for run in range(RUNS):
        a_file_url_path_is_refused(tmp_path)
        a_gzipped_token_is_found(tmp_path)
        a_stored_token_in_every_form_is_found(tmp_path)
