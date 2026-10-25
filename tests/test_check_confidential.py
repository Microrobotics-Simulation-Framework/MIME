"""The confidentiality scan, ``scripts/check_confidential.py`` (OpenMaterica's, copied): it finds invented terms however
they are written, and the private paths, in tracked files, commit messages and a branch's name, never echoes a term,
and refuses a key its digests were not made with. These are OpenMaterica's tests of the parts copied.

And N5, MIME's change: without its key the scan fails, so it cannot pass having checked nothing, on a plain run, a
push and a pull request from MIME's own repository alike, and on a payload that cannot be read; only a pull request
from a fork, told apart by the event's payload, warns, and still fails on a planted private path. With a test key and
the digest of a planted word, a file holding that word fails. And what MIME's copy reads beyond the tree's contents:
each tracked path, and with `--diffs` what each commit of a range adds, so a word added and removed within a pull
request fails, as does one in a path; `CLAUDE.md`'s allowed path holds for that file alone. The real key is CI's; these
use test keys only.

The paths are built by concatenation, so that this file does not hold them for the scan to find.
"""

import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("mime_check_confidential", ROOT / "scripts" / "check_confidential.py")
assert _SPEC is not None and _SPEC.loader is not None
check_confidential = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_confidential)
ALLOWED_PATHS, ScanError, Scanner = check_confidential.ALLOWED_PATHS, check_confidential.ScanError, \
    check_confidential.Scanner

KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))
TERMS = ("Zorblax", "quux-widget")
HOME = "/" + "home/someone/"
SCRATCHPAD = "/tmp/x/" + "scratch" + "pad/notes.txt"
TEMPORARY = "/tmp/" + "claude-1000/"
PLANNING = "plan" + "s/"
MIME = "Microrobotics-Simulation-Framework/MIME"
GITHUB = ("GITHUB_EVENT_NAME", "GITHUB_EVENT_PATH", "GITHUB_REPOSITORY")


def scanner(key: bytes | None = KEY) -> Scanner:
    digests = check_confidential.read_digests(check_confidential.digest_file(KEY, TERMS), KEY)
    return Scanner(key, digests)


@pytest.mark.parametrize(
    "text",
    [
        "Zorblax",
        "the ZORBLAX team",
        "mail@zorblax.org",
        "quux widget",
        "Quux-Widget",
        "quux_widget",
        "a quux\nwidget",
        "the quux-widget-core repository",
    ],
)
def test_a_term_is_found_however_it_is_written(text: str) -> None:
    assert scanner().findings("notes.md", text)


@pytest.mark.parametrize("text", ["zorblaxes", "quuxwidget", "widget quux", "quux, then a widget", ""])
def test_other_words_are_not_found(text: str) -> None:
    assert scanner().findings("notes.md", text) == []


def test_a_finding_gives_its_place_and_never_the_term() -> None:
    assert scanner().findings("notes.md", "First line.\nBy zorblax.\nAnd quux\nwidget.") == [
        "notes.md:2: a confidential term (z..., 7 characters)",
        "notes.md:4: a confidential term (q..., 11 characters)",
    ]


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (f"see {HOME}notes", "an absolute home directory path"),
        (f"see {SCRATCHPAD}", "a session scratchpad path"),
        (f"see {TEMPORARY}x", "a path into Claude Code's temporary directory"),
        (f"see {PLANNING}draft.md", "a path into the private planning directory"),
        (f"see ../{PLANNING}draft.md", "a path into the private planning directory"),
    ],
)
def test_the_private_paths_are_found_without_a_key(text: str, kind: str) -> None:
    assert scanner(key=None).findings("notes.md", f"Line one.\n{text}") == [f"notes.md:2: {kind}"]


def test_only_claude_md_names_the_planning_directory_and_only_one_level_up() -> None:
    allowed = ALLOWED_PATHS["CLAUDE.md"]
    assert scanner().findings("CLAUDE.md", f"Working files are in `../{PLANNING}`.", allowed) == []
    assert scanner().findings("CLAUDE.md", f"Working files are in `{PLANNING}`.", allowed) == [
        "CLAUDE.md:1: a path into the private planning directory"
    ]


def test_the_digests_are_refused_under_another_key() -> None:
    text = check_confidential.digest_file(KEY, TERMS)
    assert check_confidential.read_digests(text, KEY) == {
        check_confidential.digest(KEY, check_confidential.normalised(term)) for term in TERMS
    }
    with pytest.raises(ScanError, match="not the one the digests were made with"):
        check_confidential.read_digests(text, OTHER_KEY)
    with pytest.raises(ScanError, match="not the one the digests were made with"):
        check_confidential.read_digests(text.replace("check ", "# check "), KEY)
    with pytest.raises(ScanError, match="not a digest"):
        check_confidential.read_digests(text + "zorblax\n", KEY)


def test_the_committed_digest_file_holds_a_check_and_digests_only() -> None:
    lines = [
        line
        for line in check_confidential.DIGESTS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert [line for line in lines if line.startswith("check ")] == lines[:1]
    assert len(lines) > 1
    assert all(len(line.removeprefix("check ")) == 64 for line in lines)


@pytest.mark.parametrize("text", ["", "00" * 31, "zz" * 32, "00" * 33])
def test_a_key_is_64_hexadecimal_digits(text: str) -> None:
    with pytest.raises(ScanError, match="64 hexadecimal digits"):
        check_confidential.parse_key(text)


def test_a_key_file_comes_before_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(check_confidential.KEY_VARIABLE, OTHER_KEY.hex())
    assert check_confidential.load_key(None) == OTHER_KEY
    (tmp_path / "key").write_text(KEY.hex() + "\n", encoding="utf-8")
    assert check_confidential.load_key(tmp_path / "key") == KEY
    monkeypatch.setenv(check_confidential.KEY_VARIABLE, "")
    assert check_confidential.load_key(None) is None
    with pytest.raises(ScanError, match="--key-file"):
        check_confidential.load_key(tmp_path / "absent")


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repository as the working directory, cut off from any repository around the run (every `GIT_`
    variable removed, no global or system configuration, no automatic maintenance), with the digest file of `TERMS`
    under `KEY`, and no key or GitHub event in the environment."""
    digests = tmp_path / "digests.txt"
    digests.write_text(check_confidential.digest_file(KEY, TERMS), encoding="utf-8")
    monkeypatch.setattr(check_confidential, "DIGESTS", digests)
    for name in [name for name in os.environ if name.startswith("GIT_")] + [check_confidential.KEY_VARIABLE, *GITHUB]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.chdir(root)
    subprocess.run(["git", "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "config", "maintenance.auto", "false"], check=True)
    return root


@pytest.fixture
def key_file(tmp_path: Path) -> Path:
    written = tmp_path / "key"
    written.write_text(KEY.hex(), encoding="utf-8")
    return written


def commit(message: str, files: dict[str, bytes | None]) -> None:
    """A commit of `files`, each written, or removed where its content is None."""
    for name, data in files.items():
        if data is None:
            Path(name).unlink()
            continue
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        Path(name).write_bytes(data)
    subprocess.run(["git", "add", "-A", "--", *files], check=True)
    identity = ["-c", "user.name=Author", "-c", "user.email=author@example.org"]
    subprocess.run(["git", *identity, "commit", "-q", "-m", message], check=True)


def test_the_command_scans_tracked_files_and_the_messages_in_a_range(
    repository: Path, key_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    keyed = ["--key-file", str(key_file)]
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    commit("docs: by Zorblax", {"README.md": b"Still nothing.\n"})
    (repository / "untracked.md").write_text("zorblax\n", encoding="utf-8")
    assert check_confidential.main(keyed) == 0
    assert check_confidential.main([*keyed, "--messages", "HEAD~1..HEAD"]) == 1
    assert "message:1: a confidential term (z..., 7 characters)" in capsys.readouterr().err
    commit("docs: notes", {"notes.md": f"By quux widget.\nIn {HOME}x.\n".encode()})
    assert check_confidential.main(keyed) == 1
    assert capsys.readouterr().err.splitlines() == [
        "notes.md:1: a confidential term (q..., 11 characters)",
        "notes.md:2: an absolute home directory path",
    ]
    monkeypatch.setenv(check_confidential.KEY_VARIABLE, OTHER_KEY.hex())
    assert check_confidential.main([]) == 2
    assert check_confidential.main([*keyed, "--messages", "no-such-revision"]) == 2


def test_the_command_scans_a_ref_name_it_is_given(
    repository: Path, key_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CI gives a pull request's branch name, public from its push, which no tracked file holds. The control: a plain
    name passes."""
    keyed = ["--key-file", str(key_file)]
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    assert check_confidential.main([*keyed, "--ref-name", "docs-fix"]) == 0
    assert check_confidential.main([*keyed, "--ref-name", "fix/quux-widget-core"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        "ref fix/quux-widget-core:1: a confidential term (q..., 11 characters)"
    ]


def test_the_digest_command_writes_the_digest_file(
    repository: Path, key_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(TERMS) + "\n\n"))
    assert check_confidential.main(["--key-file", str(key_file), "--digest"]) == 0
    assert capsys.readouterr().out == check_confidential.digest_file(KEY, TERMS)
    assert check_confidential.main(["--digest"]) == 2


# Binary contents and text in UTF-16 and UTF-32, read by their printable runs.
def utf(encoding: str, text: str) -> bytes:
    return text.encode(encoding)


@pytest.mark.parametrize(
    "data",
    [
        b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\x01tEXtAuthor\0By Zorblax, in " + HOME.encode() + b"x\0\xff",
        utf("utf-16", f"By Zorblax, in {HOME}x.\n"),
        utf("utf-16-be", f"By Zorblax, in {HOME}x.\n"),
        utf("utf-32-le", f"By Zorblax, in {HOME}x.\n"),
        utf("utf-32-be", f"By Zorblax, in {HOME}x.\n"),
        b"\x00\x01\x02" + utf("utf-16-le", f"By Zorblax, in {HOME}x.") + b"\x03",
        utf("utf-16-le", "Café 中 Zorblax, ") + utf("utf-16-le", f"{HOME}x"),
    ],
    ids=["a-png-text-chunk", "utf-16", "utf-16-be", "utf-32-le", "utf-32-be", "utf-16-a-byte-off", "utf-16-mixed"],
)
def test_a_binary_content_is_read_by_its_printable_runs_once(data: bytes) -> None:
    """Each term and path found, once: UTF-16 read a byte off, UTF-16 of the other byte order, is left out."""
    found = check_confidential.content_findings(scanner(), "data", data, None)
    kinds = [line.split(": ", 1)[1] for line in found]
    assert sorted(kinds) == ["a confidential term (z..., 7 characters)", "an absolute home directory path"], found
    assert all(line.startswith("data (binary):") for line in found)


@pytest.mark.parametrize(
    "data",
    [
        bytes(range(256)) * 64,
        b"\0Zor\0blax\0",
        utf("utf-16-le", "Zor") + b"\xff\xff" + utf("utf-16-le", "blax"),
        b"\0" + utf("utf-16-le", "nothing private here") + utf("utf-32-be", "nor here"),
    ],
    ids=["every-byte", "runs-under-four", "utf-16-runs-under-four", "clean-text"],
)
def test_a_binary_content_without_a_finding_has_none(data: bytes) -> None:
    """The controls: noise, a term split into runs shorter than four, and clean text in UTF-16 and UTF-32."""
    assert check_confidential.content_findings(scanner(), "data", data, None) == []


def test_the_command_reads_a_tracked_binary_file(
    repository: Path, key_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    commit("docs: the start", {"README.md": b"Nothing here.\n", "notes.txt": utf("utf-16", "By Zorblax.\n")})
    assert check_confidential.main(["--key-file", str(key_file)]) == 1
    assert capsys.readouterr().err.splitlines() == ["notes.txt (binary):1: a confidential term (z..., 7 characters)"]


def test_a_name_that_is_not_utf_8_is_scanned_and_shown_escaped(
    repository: Path, key_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """git's paths and messages are bytes. A path that is not UTF-8 is read and shown with its bytes escaped, rather
    than stop the scan with a traceback; a message that is not UTF-8 is read too."""
    commit("docs: the start", {os.fsdecode(b"caf\xe9.md"): f"In {HOME}x.\n".encode()})
    assert check_confidential.main(["--key-file", str(key_file)]) == 1
    assert capsys.readouterr().err.splitlines() == ["caf\\xe9.md:1: an absolute home directory path"]
    Path("message").write_bytes(b"docs: by Zorblax \xff\n")
    subprocess.run(["git", "rm", "-q", os.fsdecode(b"caf\xe9.md")], check=True)
    Path("clean.md").write_text("Nothing.\n", encoding="utf-8")
    subprocess.run(["git", "add", "clean.md"], check=True)
    identity = ["-c", "user.name=Author", "-c", "user.email=author@example.org"]
    subprocess.run(["git", *identity, "commit", "-q", "-F", "message"], check=True)
    assert check_confidential.main(["--key-file", str(key_file), "--messages", "HEAD~1..HEAD"]) == 1
    assert capsys.readouterr().err.splitlines()[0].endswith("message:1: a confidential term (z..., 7 characters)")


# N5: the key, and the one run that may go without it.
def event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, payload: object) -> None:
    """GitHub Actions' environment for an event `name` whose payload is `payload`, on MIME's repository."""
    path = tmp_path / "event.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_NAME", name)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(path))
    monkeypatch.setenv("GITHUB_REPOSITORY", MIME)


def pull_request(head: str | None) -> dict[str, object]:
    """A pull request's payload, its head repository `head` (None: the fork was deleted)."""
    return {"pull_request": {"head": {"repo": None if head is None else {"full_name": head}},
                             "base": {"repo": {"full_name": MIME}}}}


def run(*arguments: str) -> int:
    return check_confidential.main(list(arguments))


def test_n5_without_its_key_the_scan_fails_and_says_why(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plain run, as on a contributor's machine: a clean tree fails, since the names were not checked; a private
    path is still found and printed."""
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    assert run() == 2
    assert capsys.readouterr().err.splitlines() == [
        "no key (CONFIDENTIAL_SCAN_KEY or --key-file), so the confidential names were not checked, and a run without "
        "the key fails: only a pull request from a fork, which has no secrets, runs without it"]
    commit("docs: notes", {"notes.md": f"In {HOME}x.\n".encode()})
    assert run() == 2
    assert capsys.readouterr().err.splitlines()[0] == "notes.md:1: an absolute home directory path"


@pytest.mark.parametrize(("name", "payload"), [
    ("push", {"ref": "refs/heads/v0.3"}),
    ("pull_request", pull_request(MIME)),
    ("pull_request", pull_request(MIME.lower())),
    ("pull_request", {"pull_request": {}}),
    ("pull_request", {"action": "opened"}),
    ("workflow_dispatch", pull_request("someone/MIME")),
], ids=["a-push", "a-pull-request-from-mimes-own-branch", "the-same-in-another-case", "a-pull-request-with-no-head",
        "no-pull-request-in-the-payload", "a-fork-payload-under-another-event"])
def test_n5_any_keyless_run_but_a_fork_pull_request_fails(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, payload: object,
    capsys: pytest.CaptureFixture[str]
) -> None:
    """The key's absence alone never makes a run a fork's: a push, and a pull request from MIME's own repository, fail
    without it, and so does a payload that does not say its head is a fork."""
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    event(monkeypatch, tmp_path, name, payload)
    assert run() == 2
    assert "a run without the key fails" in capsys.readouterr().err


@pytest.mark.parametrize("head", ["someone/MIME", None], ids=["a-fork", "a-fork-deleted-since"])
def test_n5_a_pull_request_from_a_fork_warns_and_still_fails_a_private_path(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: str | None,
    capsys: pytest.CaptureFixture[str]
) -> None:
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    event(monkeypatch, tmp_path, "pull_request", pull_request(head))
    assert run() == 0
    (warning,) = capsys.readouterr().err.splitlines()
    assert warning.startswith("warning: a pull request from a fork has no secrets, so the confidential names were not "
                              "checked; the private paths were")
    commit("docs: notes", {"notes.md": f"In {SCRATCHPAD}\nBy Zorblax.\n".encode()})
    assert run() == 1
    assert capsys.readouterr().err.splitlines()[0] == "notes.md:1: a session scratchpad path"


def test_n5_with_a_test_key_a_file_holding_a_planted_word_fails(
    repository: Path, tmp_path: Path, key_file: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
) -> None:
    """The planted word's digest is in the digest file made under the test key; a file holding it fails, giving its
    place, the word's first letter and its length, and the control, the same tree without it, passes. A fork's run
    that has a key checks the names too."""
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    monkeypatch.setenv(check_confidential.KEY_VARIABLE, KEY.hex())
    assert run() == 0
    commit("docs: notes", {"notes.md": b"Written for the zorblax team.\n"})
    assert run() == 1
    assert capsys.readouterr().err.splitlines() == ["notes.md:1: a confidential term (z..., 7 characters)"]
    event(monkeypatch, tmp_path, "pull_request", pull_request("someone/MIME"))
    assert run() == 1
    assert capsys.readouterr().err.splitlines() == ["notes.md:1: a confidential term (z..., 7 characters)"]


def test_n5_an_event_payload_that_cannot_be_read_is_no_forks(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A payload that is not JSON, or missing, or not an object, fails closed: the run is not a fork's."""
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_REPOSITORY", MIME)
    for name, payload in (("not-json", "{ not json"), ("missing", None), ("a-list", "[]")):
        path = tmp_path / f"{name}.json"
        if payload is not None:
            path.write_text(payload, encoding="utf-8")
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(path))
        assert run() == 2, name
        assert "a run without the key fails" in capsys.readouterr().err, name


def test_claude_mds_allowed_path_holds_for_claude_md_alone(
    repository: Path, key_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keyed = ["--key-file", str(key_file)]
    commit("docs: the start", {"CLAUDE.md": f"Working files are in `../{PLANNING}`.\n".encode()})
    assert run(*keyed) == 0
    commit("docs: notes", {"notes.md": f"Working files are in `../{PLANNING}`.\n".encode()})
    assert run(*keyed) == 1
    assert capsys.readouterr().err.splitlines() == ["notes.md:1: a path into the private planning directory"]


def test_a_word_in_a_tracked_path_fails(repository: Path, key_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A path is public as its content is: a word in a directory's name or a file's, with clean contents, fails, and
    so does a private path's directory; the clean tree is the control."""
    keyed = ["--key-file", str(key_file)]
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    assert run(*keyed) == 0
    commit("docs: notes", {"zorblax/notes.md": b"Clean.\n", "docs/quux_widget.md": b"Clean.\n"})
    assert run(*keyed) == 1
    assert capsys.readouterr().err.splitlines() == [
        "path docs/quux_widget.md:1: a confidential term (q..., 11 characters)",
        "path zorblax/notes.md:1: a confidential term (z..., 7 characters)"]
    commit("docs: tidy", {"zorblax/notes.md": None, "docs/quux_widget.md": None, f"{PLANNING}notes.md": b"Clean.\n"})
    assert run(*keyed) == 1
    assert capsys.readouterr().err.splitlines() == [f"path {PLANNING}notes.md:1: a path into the private planning "
                                                    "directory"]


def short(revision: str) -> str:
    return subprocess.run(["git", "rev-parse", revision], capture_output=True, text=True, check=True).stdout[:12]


def test_a_word_added_and_removed_within_the_range_fails_by_its_diff(
    repository: Path, key_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tree and the messages are clean at the end of the range, so without `--diffs` the scan passes: that is the
    gap `--diffs` closes. Each commit is read for the lines and paths it adds, a deleted file's among them."""
    keyed = ["--key-file", str(key_file)]
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    commit("docs: notes", {"README.md": b"Nothing here.\nBy Zorblax.\n", "quux_widget.md": b"Clean.\n"})
    commit("docs: tidy", {"README.md": b"Nothing here.\n", "quux_widget.md": None})
    assert run(*keyed, "--messages", "HEAD~2..HEAD") == 0
    assert run(*keyed, "--messages", "HEAD~2..HEAD", "--diffs", "HEAD~2..HEAD") == 1
    added = short("HEAD~1")
    assert capsys.readouterr().err.splitlines() == [
        f"commit {added} README.md:2: a confidential term (z..., 7 characters)",
        f"commit {added} path quux_widget.md:1: a confidential term (q..., 11 characters)"]


def test_the_diffs_read_a_binary_content_whole_and_claude_mds_allowance_holds_there(
    repository: Path, key_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keyed = ["--key-file", str(key_file)]
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    commit("docs: notes", {"notes.txt": utf("utf-16", "By Zorblax.\n"),
                           "CLAUDE.md": f"Working files are in `../{PLANNING}`.\n".encode()})
    commit("docs: tidy", {"notes.txt": None})
    assert run(*keyed, "--diffs", "HEAD~2..HEAD") == 1
    assert capsys.readouterr().err.splitlines() == [
        f"commit {short('HEAD~1')} notes.txt (binary):1: a confidential term (z..., 7 characters)"]


def test_a_public_branchs_commits_are_left_out_of_the_ranges(
    repository: Path, key_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The control is the same range without `--public`, which fails on the public branch's commit."""
    keyed = ["--key-file", str(key_file)]
    commit("docs: the start", {"README.md": b"Nothing here.\n"})
    commit("docs: by Zorblax", {"README.md": b"Nothing here.\nBy Zorblax.\n"})
    subprocess.run(["git", "branch", "public"], check=True)
    commit("docs: tidy", {"README.md": b"Nothing here.\n"})
    assert run(*keyed, "--messages", "HEAD~2..HEAD", "--diffs", "HEAD~2..HEAD") == 1
    assert len(capsys.readouterr().err.splitlines()) == 2
    assert run(*keyed, "--messages", "HEAD~2..HEAD", "--diffs", "HEAD~2..HEAD", "--public", "public") == 0
