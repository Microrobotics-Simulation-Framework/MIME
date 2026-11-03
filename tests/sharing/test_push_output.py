"""What ``mime push`` and ``mime pull`` print when output is not going to a terminal is byte for byte what #100 printed:
a script, CI's log and Modal read it. The oracle is a transcript of a session of ten commands (two pushes of a toy,
a site edit kept, a push of a toy with three layers, its status, a publish refused for want of a terminal, a publish
refused for a stale ``--confirm``, a publish by the confirmation the last push printed, a pull and ``--init``), written
from the commit before the display existed (``push_output_piped.txt``) and extended since by what the pushes and pulls
added that no display draws (the NOTICE as ``NOTICE.txt`` and the record's preview, the ``--confirm`` every push prints
and the stale confirmation's refusal; then, from the review of #112, the sharing defaults shown, ``--setup`` with no
terminal, ``--init`` from the defaults, and the record restricted and pulled with no token, signed in and by its share
link read from the environment, seven commands more); the session is run again here against the offline fake, and
compared as it stands.

Controls, each caught on every run:

- **a colour code in a piped line**: planted in what ``mime push`` says, the same comparison finds it;
- **the display drawn on a pipe**: with the terminal's display forced on a captured stream, what is printed is not the
  oracle's, so the equality is a test that a pipe is not drawn for;
- **a forced colour**: ``FORCE_COLOR`` and the settings that ask a program to colour a pipe change nothing, and a
  ``styled`` that honoured one is caught.

And what a line's kind, a progress report and a quoted command add without changing a byte of the words: a ``Said`` is
its text; a file's upload is reported to a terminal's progress and the bytes sent are the file's; a command printed for
pasting holds a folder's name as one word."""

from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from _platform_support import needs_pty
from _terminal_support import screen_of, stripped, terminal
from push_cases import copy_of, with_token
from push_output_scenario import LINK_KEY, Run, build, session
from zenodo_records import FakeRecords, served

import mime.cli
from mime import push
from mime._said import PLAN_HEAD, Said, kind_of
from mime.cli import main
from mime.push import _client

ORACLE = Path(__file__).with_name("push_output_piped.txt")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The session's two folders, made once."""
    root = tmp_path_factory.mktemp("session")
    build(root)
    return root


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeRecords]:
    token = with_token(monkeypatch)
    import pooch

    monkeypatch.setattr(pooch.get_logger(), "disabled", True)  # its handler writes to the stream it first saw, long closed
    monkeypatch.delenv("MIME_TRACEBACK", raising=False)  # a refusal is printed short, with its traceback in a file
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path / "tracebacks"))
    monkeypatch.setattr(_client, "PAUSE_SECONDS", 0.0)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    with served(token) as held:
        monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", held.service())  # type: ignore[index]
        yield held


def piped(capsys: pytest.CaptureFixture[str]):  # noqa: ANN201
    def run(words: list[str]) -> tuple[int, str, str]:
        code = main(words)
        held = capsys.readouterr()
        return code, held.out, held.err

    return run


def transcript(built: Path, fake: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    work = Path(shutil.copytree(built, tmp_path / "work"))
    return session(work, fake, piped(capsys))


def test_the_session_piped_is_exactly_what_it_was_before_the_display(built: Path, fake: FakeRecords, tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str]):
    assert transcript(built, fake, tmp_path, capsys) == ORACLE.read_text(encoding="utf-8")


def test_the_oracle_holds_what_it_is_the_oracle_of():
    """The session's words are in it: the plan, the reference, the permission, the publish statement, the refusal."""
    text = ORACLE.read_text(encoding="utf-8")
    for words in ("Pushing 'full_toy' to the fake of zenodo-sandbox", "  by reference, never uploaded: scene/layers/arm_visual.usda",
                  "! 1 permission declared", "Publishing 'full_toy' on zenodo-sandbox as 10.5072/zenodo.629525",
                  "push.permission-unconfirmed", "kept what was edited on the site", "Bundle 'full_toy' unpacked",
                  "[exit 0]", "[exit 2]", "<fake>/uploads/629525", "your sharing defaults", "push.setup",
                  "filled from your sharing defaults", "pull.restricted", "--signed-in", "$ mime pull --link env"):
        assert words in text, words
    assert LINK_KEY not in text, "a share link's key is in no line"
    assert "\x1b" not in text


def test_the_control_a_colour_code_planted_in_a_piped_line_is_found(built: Path, fake: FakeRecords, tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str],
                                                                     monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mime.cli, "_say", lambda line: print(f"\x1b[1m{line}\x1b[0m" if line.startswith("made the draft")
                                                          else line))
    planted = transcript(built, fake, tmp_path, capsys)
    assert "\x1b[1mmade the draft" in planted and planted != ORACLE.read_text(encoding="utf-8")


def test_the_control_the_display_drawn_on_a_pipe_is_not_the_oracle(built: Path, fake: FakeRecords, tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture[str],
                                                                    monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mime.cli, "styled", lambda stream: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    drawn = transcript(built, fake, tmp_path, capsys)
    assert "╰" in drawn or "╯" in drawn, "the plan's frame was drawn"
    assert drawn != ORACLE.read_text(encoding="utf-8")


FORCING = [{"FORCE_COLOR": "1"}, {"FORCE_COLOR": "3"}, {"CLICOLOR_FORCE": "1"}, {"TTY_COMPATIBLE": "1"},
           {"TTY_INTERACTIVE": "1"}, {"CI": "true"}]


@pytest.mark.parametrize("environment", FORCING, ids=lambda each: "+".join(f"{k}={v}" for k, v in each.items()))
def test_a_pipe_told_to_force_colour_is_still_the_oracle_and_holds_no_escape(environment, built: Path,
                                                                             fake: FakeRecords, tmp_path: Path,
                                                                             capsys: pytest.CaptureFixture[str],
                                                                             monkeypatch: pytest.MonkeyPatch):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    held = transcript(built, fake, tmp_path, capsys)
    assert held == ORACLE.read_text(encoding="utf-8") and "\x1b" not in held


def test_the_control_a_styled_that_honours_force_colour_is_caught(built: Path, fake: FakeRecords, tmp_path: Path,
                                                                   capsys: pytest.CaptureFixture[str],
                                                                   monkeypatch: pytest.MonkeyPatch):
    """The arms above fail for the regression they guard: a ``styled`` that read ``FORCE_COLOR`` would draw on the pipe."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    real = mime.cli.styled
    monkeypatch.setattr(mime.cli, "styled", lambda stream: bool(os.environ.get("FORCE_COLOR")) or real(stream))
    assert transcript(built, fake, tmp_path, capsys) != ORACLE.read_text(encoding="utf-8")


# --- a line's kind is not its text ----------------------------------------------------------------------------------


def test_a_said_line_is_its_text_in_every_way_a_pipe_or_a_test_can_read_it():
    line = Said("  up: full_toy.mimebundle 31 kB", PLAN_HEAD)
    assert line == "  up: full_toy.mimebundle 31 kB" and kind_of(line) == PLAN_HEAD
    assert str(line) == f"{line}" == "".join([line]) == "  up: full_toy.mimebundle 31 kB"
    assert hash(line) == hash("  up: full_toy.mimebundle 31 kB") and line.encode() == b"  up: full_toy.mimebundle 31 kB"
    assert kind_of("  up: full_toy.mimebundle 31 kB") is None
    printed = io.StringIO()
    print(line, file=printed)
    assert printed.getvalue() == "  up: full_toy.mimebundle 31 kB\n"


def test_what_push_says_is_told_apart_by_kind_and_each_kind_is_its_text(built: Path, fake: FakeRecords, tmp_path: Path):
    said: list[str] = []
    folder = copy_of(built / "mime_sandbox_terms", tmp_path)
    push.push(folder, push.SERVICES["zenodo-sandbox"], say=said.append)
    assert [kind_of(line) for line in said] == ["plan-head", "plan-up", "plan-reference", "plan-field", "plan-field",
                                                "plan-warning", "done", "done", None, "link"]
    assert [bool(getattr(line, "last", False)) for line in said] == [False] * 5 + [True] + [False] * 4
    assert said[5].startswith("! 1 permission declared") and said[-1] == f"{fake.site}/uploads/629525"


# --- the upload is reported ----------------------------------------------------------------------------------------


class Told:
    """A :class:`~mime._transfers.Transfers` that keeps what it is told, in order."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def files(self, files) -> None:  # noqa: ANN001
        self.events.append(("files", tuple(files)))

    def sending(self, key: str, done: int, total: int | None) -> None:
        self.events.append(("sending", key, done, total))

    def sent(self, key: str) -> None:
        self.events.append(("sent", key))

    def doing(self, words: str) -> None:
        self.events.append(("doing", words))

    def of(self, kind: str) -> list[tuple]:
        return [event[1:] for event in self.events if event[0] == kind]


def test_each_file_goes_up_reported_to_the_end_and_the_bytes_sent_are_the_files(built: Path, fake: FakeRecords,
                                                                                  tmp_path: Path,
                                                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_client, "_REPORT_BYTES", 4096)
    folder = copy_of(built / "mime_sandbox_toy", tmp_path)
    told = Told()
    push.push(folder, push.SERVICES["zenodo-sandbox"], progress=told, transport=_client.HttpTransport())
    held = fake.records["629525"].files
    staged = push.state_directory(push.SERVICES["zenodo-sandbox"], "full_toy") / "staged"
    for key, entry in held.items():
        reports = [(done, total) for name, done, total in told.of("sending") if name == key]
        assert entry.data == (staged / key).read_bytes() and len(entry.data) == reports[-1][1]
        assert reports[0][0] == 0 and reports[-1][0] == reports[-1][1], "from nothing to the whole file"
        assert [done for done, _ in reports] == sorted(done for done, _ in reports), "never backwards"
    bundle = [done for name, done, _ in told.of("sending") if name == "full_toy.mimebundle"]
    assert len(bundle) > 4, "a file of many blocks is reported on the way, not only at its ends"


def test_the_files_to_come_are_announced_first_each_is_whole_after_its_bytes_and_the_gaps_are_named(
        built: Path, fake: FakeRecords, tmp_path: Path):
    """What the display counts to and says between the bars: the files and their sizes before the first byte, a file whole
    only after the service has committed it and its md5 is checked, and the waits (``pause`` spaces the requests) and
    commits named."""
    folder = copy_of(built / "mime_sandbox_toy", tmp_path)
    told = Told()
    push.push(folder, push.SERVICES["zenodo-sandbox"], progress=told, transport=_client.HttpTransport(), pause=0.02)
    held = fake.records["629525"].files
    assert told.events[0][0] in ("doing", "files")
    announced = told.of("files")
    assert len(announced) == 1 and dict(announced[0][0]) == {key: len(entry.data) for key, entry in held.items()}
    first_bytes = next(i for i, event in enumerate(told.events) if event[0] == "sending")
    assert told.events.index(("files", announced[0][0])) < first_bytes
    for key in held:
        last_bytes = max(i for i, event in enumerate(told.events) if event[:2] == ("sending", key))
        commit = next(i for i, event in enumerate(told.events) if event[0] == "doing" and "committing " + key in event[1])
        whole = told.events.index(("sent", key))
        assert last_bytes < commit < whole, "whole only once committed and checked"
    words = [event[1] for event in told.events if event[0] == "doing"]
    assert any(text.startswith("waiting between requests, then ") for text in words), "the pacing is said, and what follows it"
    assert any("reading the draft again" in text for text in words), "the check at the end is said"


def test_a_transport_that_is_not_the_real_one_is_told_the_files_but_never_their_bytes(built: Path, fake: FakeRecords,
                                                                                         tmp_path: Path):
    from zenodo_records import FakeTransport

    told = Told()
    folder = copy_of(built / "mime_sandbox_toy", tmp_path)
    drafted = push.push(folder, push.SERVICES["zenodo-sandbox"], progress=told, transport=FakeTransport(fake), pause=0)
    assert told.of("sending") == [] and drafted.uploaded, "files went up, and a test's transport was told no bytes"
    assert len(told.of("sent")) == len(drafted.uploaded) and told.of("files")


# --- a command printed for pasting holds a name as one word --------------------------------------------------------


def test_a_folder_named_with_a_space_a_semicolon_and_a_substitution_is_one_word_in_every_command_push_prints(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    odd = tmp_path / "my toy; touch pwned $(id) 'x'"
    shutil.copytree(built / "mime_sandbox_toy", odd)
    run = piped(capsys)
    code, out, _ = run(["push", str(odd), "--to", "zenodo-sandbox", "--init"])
    command = out.splitlines()[-1].split("then ", 1)[1].strip("`")
    assert code == 0 and shlex.split(command) == ["mime", "push", str(odd), "--to", "zenodo-sandbox"]
    code, _, err = run(["push", str(odd), "--to", "zenodo-sandbox", "--confirm", "10.5072/zenodo.1"])
    command = next(line for line in err.splitlines() if "--publish --confirm" in line).split("`")[1]
    assert code == 2 and shlex.split(command)[:3] == ["mime", "push", str(odd)]
    plain = tmp_path / "plain"
    shutil.copytree(odd, plain)
    (plain / "sharing.yaml").unlink()
    code, _, err = run(["push", str(plain), "--to", "zenodo-sandbox"])
    assert code == 2 and f"`mime push {plain} --to <service> --init`" in err, "a name that needs no quoting is not changed"
    (odd / "sharing.yaml").unlink()
    code, _, err = run(["push", str(odd), "--to", "zenodo-sandbox"])
    command = next(line for line in err.splitlines() if "--init" in line).split("`")[1]
    assert code == 2 and shlex.split(command)[:3] == ["mime", "push", str(odd)]


def test_the_control_a_name_that_is_not_quoted_is_not_one_word():
    name = "my toy; touch x"
    assert shlex.split(f"mime push {name} --to zenodo")[:3] != ["mime", "push", name]
    assert shlex.split(f"mime push {shlex.quote(name)} --to zenodo")[:3] == ["mime", "push", name]


# --- on a terminal ---------------------------------------------------------------------------------------------------

FRAME = re.compile(r"[\u2500-\u257f]")


def words(text: str) -> str:
    return " ".join(FRAME.sub(" ", text).split())


def blocks(text: str) -> list[str]:
    """A transcript as its commands: each block starts at a ``$ mime`` line."""
    return ["$ mime" + each for each in text.split("$ mime")[1:]]


@contextlib.contextmanager
def on_terminals(monkeypatch: pytest.MonkeyPatch, columns: int = 160) -> Iterator[Run]:
    """A ``run`` whose standard output and standard error are two pseudo-terminals, each command's share read as text."""
    held = (sys.stdout, sys.stderr)
    with terminal(columns=columns) as tty_out, terminal(columns=columns) as tty_err:
        monkeypatch.setattr(sys, "stdout", tty_out.stream)
        monkeypatch.setattr(sys, "stderr", tty_err.stream)
        seen = {"out": 0, "err": 0}

        def run(command: list[str]) -> tuple[int, str, str]:
            code = main(command)
            shown = []
            for name, tty in (("out", tty_out), ("err", tty_err)):
                text = tty.transcript().replace("\r\n", "\n")
                shown.append(text[seen[name]:])
                seen[name] = len(text)
            return code, shown[0], shown[1]

        run.on_screen = lambda: stripped(tty_out.transcript())  # type: ignore[attr-defined]  # standard output so far
        try:
            yield run
        finally:
            monkeypatch.setattr(sys, "stdout", held[0])
            monkeypatch.setattr(sys, "stderr", held[1])


@needs_pty
def test_every_word_the_pipe_gets_is_on_the_terminal_too_in_the_whole_session(built: Path, fake: FakeRecords,
                                                                              tmp_path: Path, capsys,
                                                                              monkeypatch: pytest.MonkeyPatch):
    """Frames, colours and a progress line are added; no word of standard output is lost, and standard error (a refusal
    drawn as a block in its own words) is the other test's."""
    work = Path(shutil.copytree(built, tmp_path / "work"))
    with on_terminals(monkeypatch) as run:
        shown = session(work, fake, run)
    expected = ORACLE.read_text(encoding="utf-8")
    assert len(blocks(shown)) == len(blocks(expected)) == 17
    for drawn, piped in zip(blocks(shown), blocks(expected), strict=True):
        text = words(stripped(drawn))
        for line in piped.split("[stderr]")[0].splitlines():
            assert words(line) in text or not words(line), f"missing from the terminal: {line!r}"
    assert "\u256d" in shown, "the plan was drawn as a block"


@needs_pty
@pytest.mark.parametrize("asked", ["--plain", "NO_COLOR", "TERM=dumb"])
def test_a_terminal_asked_to_be_plain_gets_exactly_what_the_pipe_gets(asked, built: Path, fake: FakeRecords,
                                                                      tmp_path: Path, capsys,
                                                                      monkeypatch: pytest.MonkeyPatch):
    if asked == "NO_COLOR":
        monkeypatch.setenv("NO_COLOR", "1")
    elif asked == "TERM=dumb":
        monkeypatch.setenv("TERM", "dumb")
    work = Path(shutil.copytree(built, tmp_path / "work"))
    with on_terminals(monkeypatch) as run:
        shown = session(work, fake, lambda command: run(["--plain", *command] if asked == "--plain" else command))
    assert shown == ORACLE.read_text(encoding="utf-8")


@needs_pty
def test_a_publish_in_a_terminal_frames_the_step_and_each_permission_sets_each_question_apart_and_publishes_only_the_number_typed(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    work = Path(shutil.copytree(built, tmp_path / "work"))
    terms = str(work / "mime_sandbox_terms")
    assert main(["push", terms, "--to", "zenodo-sandbox"]) == 0
    capsys.readouterr()
    prompts: list[str] = []
    answers = iter(["y", "629525"])
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(answers))
    monkeypatch.setattr("mime.cli._a_terminal", lambda: True)
    with on_terminals(monkeypatch) as run:
        code, out, _ = run(["push", terms, "--to", "zenodo-sandbox", "--publish"])
    plain = stripped(out)
    lines = plain.splitlines()
    assert code == 0 and fake.records["629525"].published and prompts == ["", ""]
    heavy = next(i for i, line in enumerate(lines) if "\u250f" in line)
    permission = next(i for i, line in enumerate(lines) if "\u2554" in line)
    first = next(i for i, line in enumerate(lines) if "? Declare permission 1 publicly, under your name?" in line)
    number = next(i for i, line in enumerate(lines) if "? Type the record number to publish (629525):" in line)
    assert heavy < permission < first < number
    assert "This is the sandbox, a test service" in words(plain) and "published 'mime_sandbox_terms' as" in plain
    assert lines[-1].strip() == f"{fake.site}/records/629525"


@needs_pty
def test_a_number_other_than_the_drafts_publishes_nothing_on_a_terminal_too(built: Path, fake: FakeRecords,
                                                                            tmp_path: Path, capsys,
                                                                            monkeypatch: pytest.MonkeyPatch):
    work = Path(shutil.copytree(built, tmp_path / "work"))
    toy = str(work / "mime_sandbox_toy")
    assert main(["push", toy, "--to", "zenodo-sandbox"]) == 0
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", lambda prompt="": "629524")
    monkeypatch.setattr("mime.cli._a_terminal", lambda: True)
    with on_terminals(monkeypatch) as run:
        code, out, _ = run(["push", toy, "--to", "zenodo-sandbox", "--publish"])
    assert code == 0 and not fake.records["629525"].published
    assert "Not published: '629524' is not the record number 629525." in stripped(out)


@needs_pty
def test_a_site_edit_on_a_terminal_is_two_values_side_by_side_and_the_answers_settle_it(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    from push_output_scenario import edit_on_site

    work = Path(shutil.copytree(built, tmp_path / "work"))
    toy = str(work / "mime_sandbox_toy")
    assert main(["push", toy, "--to", "zenodo-sandbox"]) == 0
    capsys.readouterr()
    edit_on_site(fake, "629525")
    answers = iter(["n", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("mime.cli._a_terminal", lambda: True)
    with on_terminals(monkeypatch) as run:
        code, out, _ = run(["push", toy, "--to", "zenodo-sandbox"])
    plain = stripped(out)
    row = next(line for line in plain.splitlines() if "A title corrected on the site" in line)
    assert code == 0 and "title: full_toy" in row
    assert "! metadata.creators was edited on the site" in plain and "! wrote over what was edited on the site: " in plain
    assert fake.records["629525"].body["metadata"]["title"] == "full_toy"  # type: ignore[index]


@needs_pty
def test_an_upload_and_a_download_leave_their_count_and_last_bar_on_standard_error_and_nothing_among_the_lines(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    work = Path(shutil.copytree(built, tmp_path / "work"))
    toy = str(work / "mime_sandbox_toy")
    with on_terminals(monkeypatch) as run:
        pushed = run(["push", toy, "--to", "zenodo-sandbox"])
        assert run(["push", toy, "--to", "zenodo-sandbox", "--publish", "--confirm", "10.5072/zenodo.629525"])[0] == 0
        pulled = run(["pull", "--from", "zenodo-sandbox", "629525", "--into", str(work / "pulled")])
    assert pushed[0] == 0 and pulled[0] == 0 and (work / "pulled" / "mime.lock").is_file()
    up, down = [line for line in screen_of(pushed[2]) if line], [line for line in screen_of(pulled[2]) if line]
    assert up[-2].startswith("2/2 files uploaded") and "uploaded NOTICE" in up[-1], up
    assert down[-2].startswith("1/1 file downloaded") and "downloaded bundle" in down[-1], down
    assert "uploaded" not in stripped(pushed[1]).split("made the draft")[0], "nothing of the progress among the lines"
    assert "files uploaded" not in stripped(pushed[1]) and "downloaded bundle" not in stripped(pulled[1])


# --- the publish step is on the screen, whole, before the request that publishes ----------------------------------------


def published_on_a_terminal(built: Path, fake: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """``--publish --confirm`` of a toy on a terminal (the typed number's question is not there to draw the statement):
    what standard output showed at the moment the publish request was made."""
    work = Path(shutil.copytree(built, tmp_path / "work"))
    toy = str(work / "mime_sandbox_toy")
    assert main(["push", toy, "--to", "zenodo-sandbox"]) == 0
    seen: list[str] = []
    publish = _client.Client.publish

    def watching(self: _client.Client, record: str):  # noqa: ANN202
        seen.append(run.on_screen())  # type: ignore[attr-defined]
        return publish(self, record)

    monkeypatch.setattr(_client.Client, "publish", watching)
    with on_terminals(monkeypatch) as run:
        code, _, _ = run(["push", toy, "--to", "zenodo-sandbox", "--publish", "--confirm", "10.5072/zenodo.629525"])
    assert code == 0 and fake.records["629525"].published and len(seen) == 1
    return seen[0]


@needs_pty
def test_with_confirm_on_a_terminal_the_whole_statement_is_on_the_screen_before_the_publish_request(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    """The publish is irreversible, and ``--confirm`` asks nothing that would draw the statement first: it was drawn with
    the answer, seconds after the request (the first review of this display, F1)."""
    from mime.push._publish import SANDBOX_FACTS

    screen = published_on_a_terminal(built, fake, tmp_path, monkeypatch)
    before = words(screen)
    for line in ("Publishing 'full_toy' on zenodo-sandbox as 10.5072/zenodo.629525", SANDBOX_FACTS,
                 "You are publishing, under your name,", "You state that everyone named as a creator or contributor "
                                                          "agreed to be named."):
        assert words(line) in before, f"not yet on the screen when the request was made: {line!r}"
    assert "\u250f" in screen and "\u2517" in screen, "the heavy frame, closed"
    assert "published 'full_toy'" not in before


@needs_pty
def test_the_control_a_statement_not_marked_as_ending_is_not_on_the_screen_at_the_request(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    """The same run with the statement's last line not marked (as it was): the screen holds nothing of it at the request,
    so the test above is a test that can fail."""
    monkeypatch.setattr("mime.push._publish.ending", lambda lines: lines)
    screen = published_on_a_terminal(built, fake, tmp_path, monkeypatch)
    assert "agreed to be named" not in screen and "\u250f" not in screen


# --- a terminal that reads 0 columns, end to end -----------------------------------------------------------------------


@needs_pty
def test_on_a_terminal_that_reads_0_columns_a_push_and_a_publish_show_their_plan_statement_and_questions(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    work = Path(shutil.copytree(built, tmp_path / "work"))
    terms = str(work / "mime_sandbox_terms")
    monkeypatch.delenv("COLUMNS", raising=False)
    answers = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("mime.cli._a_terminal", lambda: True)
    with on_terminals(monkeypatch, columns=0) as run:
        code, out, _ = run(["push", terms, "--to", "zenodo-sandbox"])
        assert code == 0 and "Pushing 'mime_sandbox_terms'" in stripped(out), "the push drew nothing"
        code, out, _ = run(["push", terms, "--to", "zenodo-sandbox", "--publish"])
    plain = stripped(out)
    assert "Publishing 'mime_sandbox_terms' on zenodo-sandbox" in plain and "? Declare permission 1 publicly, under your name?" in plain
    assert plain.rstrip().endswith("[y/N]") or "Not published: permission 1 was not declared." in plain
    assert not fake.records["629525"].published


@needs_pty
def test_the_control_at_0_columns_with_the_terminals_own_reading_nothing_is_drawn(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    from mime.cli import _display

    monkeypatch.setattr(_display, "_columns_of", lambda stream: os.get_terminal_size(stream.fileno()).columns)
    work = Path(shutil.copytree(built, tmp_path / "work"))
    with on_terminals(monkeypatch, columns=0) as run:
        code, out, _ = run(["push", str(work / "mime_sandbox_terms"), "--to", "zenodo-sandbox"])
    assert code == 0 and "Pushing" not in stripped(out), "the fault the test above guards: a push that draws nothing"


# --- a fault of the display's own never changes what push does about the service ------------------------------------


def test_a_progress_that_raises_is_read_by_the_library_as_a_dropped_connection_which_is_why_a_display_catches_its_own(
        built: Path, fake: FakeRecords, tmp_path: Path):
    """The control: a ``Transfers`` that raises in ``sending`` (the library does not catch it) refuses the push as a lost
    connection, though the network is fine."""
    from mime.build import BuildRefused

    class Raises(Told):
        def sending(self, key: str, done: int, total: int | None) -> None:
            raise RuntimeError("the terminal went away")

    folder = copy_of(built / "mime_sandbox_toy", tmp_path)
    with pytest.raises(BuildRefused) as refused:
        push.push(folder, push.SERVICES["zenodo-sandbox"], progress=Raises(), transport=_client.HttpTransport(), pause=0)
    assert refused.value.codes() == ["push.service"] and "connection dropped" in str(refused.value)


@needs_pty
def test_a_display_that_raises_does_not_refuse_the_push_it_is_drawing(built: Path, fake: FakeRecords, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch):
    from mime.cli._display import TransferDisplay

    folder = copy_of(built / "mime_sandbox_toy", tmp_path)
    with terminal() as tty:
        display = TransferDisplay(tty.stream, "uploading", "uploaded")

        def broken(*_: object, **__: object) -> None:
            raise RuntimeError("the terminal went away")

        monkeypatch.setattr(display._live, "start", broken)  # noqa: SLF001
        drafted = push.push(folder, push.SERVICES["zenodo-sandbox"], progress=display,
                            transport=_client.HttpTransport(), pause=0)
        display.close()
    assert display.failed and len(drafted.uploaded) == 2
    assert {key: entry.data for key, entry in fake.records["629525"].files.items()}.keys() == {
        "full_toy.mimebundle", "NOTICE.txt"}


# --- a stranger's name on a terminal --------------------------------------------------------------------------------


@needs_pty
def test_a_creators_name_in_cjk_with_a_combining_mark_and_a_reordering_escape_is_text_on_the_screen(
        built: Path, fake: FakeRecords, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    """A name from outside is shown as the characters it is, the control and format characters named rather than obeyed
    (here U+202E, which would reverse what follows), and a double-width name does not break the plan's frame."""
    work = Path(shutil.copytree(built, tmp_path / "work"))
    toy = work / "mime_sandbox_toy"
    sharing = toy / "sharing.yaml"
    sharing.write_text(sharing.read_text(encoding="utf-8").replace(
        "{family: Carberry, given: Josiah", '{family: "\u5c71\u7530\u202edcba", given: "\u592a\u90ce e\u0301"'), encoding="utf-8")
    with on_terminals(monkeypatch) as run:
        code, out, _ = run(["push", str(toy), "--to", "zenodo-sandbox", "--full-plan"])
    plain = stripped(out)
    assert code == 0, plain
    assert "\u5c71\u7530\\u202edcba, \u592a\u90ce e\u0301" in plain, "the characters it is, and the reordering escape named"
    assert "\u202e" not in out, "and never sent to the terminal"
    from rich.cells import cell_len

    assert len({cell_len(line) for line in plain.splitlines() if line[:1] in "\u256d\u2502\u2570"}) == 1, \
        "a name of double-width characters leaves the frame's right side where it was"
