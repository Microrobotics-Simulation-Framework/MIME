"""The licence gate and the disk gate as the command line meets them: a real pseudo-terminal for the panel and its question
(POSIX; ``_platform_support.needs_pty`` says what does not run on Windows), and pipes for the other half of the rule: with
no terminal to ask at, ``mime visuals fetch`` refuses, naming the licence and ``--accept-licence <id>``, and neither
``--yes`` nor ``--force`` accepts a licence. The meshes are served by a server on this machine; nothing is downloaded from
the network.

The panel is read from the terminal's transcript: its words (a heading in words, the conditions as lines, the licence's
title and link, the notice under it, and ``Do you accept these conditions? [y/N]``), and that its border is amber (the
colour is a styling of words that are all there without it: ``--plain`` asks in the same words with no escape at all)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assets"))

from _platform_support import needs_pty  # noqa: E402
from _terminal_support import run_on_terminal, run_with_streams, stripped  # noqa: E402
from assets_rig import Rig  # noqa: E402

CLOSED = "LicenseRef-Maker-NonCommercial"
LINES = ["non-commercial use only", "no redistribution in any form"]
NOTICE = "The maker's notice, in the maker's words."
AMBER = "38;5;214"


def stl(triangles: int = 40) -> bytes:
    body = b"".join(struct.pack("<12fH", 0.0, 0.0, 1.0, *(0.01 * ((i + k) % 5) for k in range(9)), 0)
                    for i in range(triangles))
    return b"made".ljust(80, b"\0") + struct.pack("<I", triangles) + body


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    held = Rig(tmp_path, monkeypatch)
    yield held
    held.server.close()


@pytest.fixture
def manifest(tmp_path: Path, rig: Rig) -> Path:
    """A manifest of one mesh under a licence with conditions, served by the rig."""
    data = stl()
    rig.server.files["/made.stl"] = data
    held = {"format": "mime.asset_manifest/2", "name": "made", "title": "a made mesh", "credit": "a maker",
            "notice": NOTICE,
            "licences": {CLOSED: {"title": "Maker Licence 1.1", "url": "https://example.org/licence",
                                  "redistribution": "not_permitted", "conditions": LINES}},
            "archives": [],
            "assets": [{"name": "made.stl", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                        "licence": CLOSED, "locations": [rig.server.url("/made.stl")], "role": "presentation"}],
            "files": {"meshes/made.stl": {"asset": "made.stl"}}}
    path = tmp_path / "made.visuals.json"
    path.write_text(json.dumps(held), encoding="utf-8")
    return path


def environment(tmp_path: Path, rig: Rig, **extra: str) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    held = {key: value for key, value in os.environ.items() if key not in ("MIME_TRACEBACK", "NO_COLOR", "FORCE_COLOR",
                                                                            "TERM", "MIME_OFFLINE")}
    return {**held, "HOME": str(home), "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"), "APPDATA": str(home / "AppData"), "TERM": "xterm-256color",
            "JAX_PLATFORMS": "cpu", "MIME_ASSETS_DIR": str(rig.cache), "MIME_ASSET_LOCATIONS": str(tmp_path / "none.toml"),
            "MIME_LICENCE_ACCEPTANCES": str(tmp_path / "accepted.json"), "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost", **extra}


def flat(text: str) -> str:
    """A transcript with the frames dropped and every run of white space one space: a refusal's block wraps its words at
    the width of the terminal, which a runner sets as it likes."""
    return " ".join(re.sub(r"[│╭╮╰╯─┃┏┓┗┛━]", " ", text).split())


def argv(*words: str) -> list[str]:
    return [sys.executable, "-m", "mime.cli", *words]


def on_a_terminal(tmp_path: Path, rig: Rig, *words: str, typed: bytes = b"", **extra: str) -> tuple[str, int]:
    """``mime <words>`` with a terminal for all three streams, a person typing ``typed`` once the child waits for it."""
    (tmp_path / "none.toml").write_text("", encoding="utf-8")
    transcript, code = run_with_streams(argv(*words), env=environment(tmp_path, rig, **extra), cwd=tmp_path,
                                        terminals=("out", "err"), columns=120, typed=typed)
    return transcript["tty"], code


def piped(tmp_path: Path, rig: Rig, *words: str, **extra: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "none.toml").write_text("", encoding="utf-8")
    return subprocess.run(argv(*words), capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL,
                          env=environment(tmp_path, rig, **extra), cwd=tmp_path)


def accepted(tmp_path: Path) -> list[str]:
    path = tmp_path / "accepted.json"
    return [row["id"] for row in json.loads(path.read_text(encoding="utf-8"))["accepted"]] if path.is_file() else []


# --- without a terminal ------------------------------------------------------------------------------------------------


def test_without_a_terminal_the_fetch_refuses_naming_the_licence_and_the_flag_and_requests_nothing(
        tmp_path: Path, rig: Rig, manifest: Path):
    done = piped(tmp_path, rig, "visuals", "fetch", str(manifest))
    assert done.returncode == 2 and done.stdout == ""
    assert "asset.licence-not-accepted" in done.stderr and f"--accept-licence {CLOSED}" in done.stderr
    assert "Maker Licence 1.1" in done.stderr and LINES[0] in done.stderr
    assert rig.requests == 0 and accepted(tmp_path) == []


@pytest.mark.parametrize("switch", [["--yes"], ["-y"], ["--force"], ["-f"], ["--yes", "--force"]])
def test_neither_yes_nor_force_accepts_a_licence_at_the_command_line(tmp_path: Path, rig: Rig, manifest: Path,
                                                                     switch: list[str]):
    done = piped(tmp_path, rig, "visuals", "fetch", str(manifest), *switch)
    assert done.returncode == 2 and "asset.licence-not-accepted" in done.stderr
    assert rig.requests == 0


def test_the_flag_accepts_the_licence_it_names_and_only_that_one(tmp_path: Path, rig: Rig, manifest: Path):
    wrong = piped(tmp_path, rig, "visuals", "fetch", str(manifest), "--accept-licence", "LicenseRef-Someone-Else")
    assert wrong.returncode == 2 and f"--accept-licence {CLOSED}" in wrong.stderr and rig.requests == 0
    done = piped(tmp_path, rig, "visuals", "fetch", str(manifest), "--accept-licence", "LicenseRef-Someone-Else",
                 "--accept-licence", CLOSED)
    assert done.returncode == 0, done.stderr
    assert "a made mesh\ncredit: a maker" in done.stdout and NOTICE in done.stdout, "the notice is said as it was"
    assert accepted(tmp_path) == [], "a flag accepts for the command and is not remembered"
    assert rig.requests == 1


def test_the_flag_is_taken_by_assets_fetch_and_pull_as_well(tmp_path: Path, rig: Rig):
    for words in (("assets", "fetch"), ("pull",)):
        done = piped(tmp_path, rig, *words, "--help")
        assert "--accept-licence ID" in done.stdout and "-y, --yes" in done.stdout and "-f, --force" in done.stdout


# --- on a terminal ---------------------------------------------------------------------------------------------------------


@needs_pty
def test_the_panel_is_amber_titled_in_words_and_asks_in_bold_and_a_no_downloads_nothing(tmp_path: Path, rig: Rig,
                                                                                         manifest: Path):
    raw, code = on_a_terminal(tmp_path, rig, "visuals", "fetch", str(manifest), typed=b"n\n")
    text = stripped(raw)
    assert code == 2 and rig.requests == 0 and accepted(tmp_path) == []
    assert "Licence conditions: your acceptance is needed" in text
    for line in LINES:
        assert line in flat(text)
    assert "Maker Licence 1.1 (LicenseRef-Maker-NonCommercial)" in flat(text) and "https://example.org/licence" in text
    assert NOTICE in flat(text)
    assert text.index("Licence conditions") < text.index(LINES[0]) < text.index("https://example.org/licence") \
        < text.index(NOTICE) < text.index("Do you accept these conditions? [y/N]")
    assert "asset.licence-declined" in flat(text) and "nothing was downloaded" in flat(text)
    assert AMBER in raw, "the border is amber (xterm colour 214)"
    assert "\x1b[1m" in raw and raw.index("\x1b[1m") < raw.index("Do you accept"), "the question is bold"


@needs_pty
def test_a_yes_downloads_remembers_and_the_next_fetch_asks_nothing(tmp_path: Path, rig: Rig, manifest: Path):
    raw, code = on_a_terminal(tmp_path, rig, "visuals", "fetch", str(manifest), typed=b"y\n")
    text = stripped(raw)
    assert code == 0, text
    assert "Do you accept these conditions? [y/N]" in text and "baked into the asset cache" in text
    assert "Download " in text and "free of" in text, "the line that says what the download takes and the drive has"
    assert rig.requests == 1 and accepted(tmp_path) == [CLOSED]
    again, code = on_a_terminal(tmp_path, rig, "visuals", "fetch", str(manifest))
    assert code == 0 and "Do you accept" not in stripped(again) and "Licence conditions" not in stripped(again)
    assert rig.requests == 1
    listed, code = on_a_terminal(tmp_path, rig, "assets", "licences")
    assert code == 0 and CLOSED in stripped(listed) and "accepted" in stripped(listed)
    revoked = piped(tmp_path, rig, "assets", "licences", "--revoke", CLOSED)
    assert revoked.returncode == 0 and CLOSED in revoked.stdout and accepted(tmp_path) == []
    missing = piped(tmp_path, rig, "assets", "licences", "--revoke", CLOSED)
    assert missing.returncode == 1 and "no such acceptance" in missing.stderr
    cached, code = on_a_terminal(tmp_path, rig, "visuals", "fetch", str(manifest), typed=b"n\n")
    assert code == 0 and "Do you accept" not in stripped(cached), "a file already in the cache is no download to ask about"
    shutil.rmtree(rig.cache)
    asked_again, code = on_a_terminal(tmp_path, rig, "visuals", "fetch", str(manifest), typed=b"n\n")
    assert code == 2 and "Do you accept these conditions? [y/N]" in stripped(asked_again)


@needs_pty
@pytest.mark.parametrize("answer", [b"\n", b"no\n", b"yes please\n", b"Y\n"])
def test_only_y_or_yes_accepts_and_the_default_is_no(tmp_path: Path, rig: Rig, manifest: Path, answer: bytes):
    raw, code = on_a_terminal(tmp_path, rig, "visuals", "fetch", str(manifest), typed=answer)
    assert (code == 0) is (answer == b"Y\n"), stripped(raw)
    assert rig.requests == (1 if answer == b"Y\n" else 0)


@needs_pty
def test_plain_asks_the_same_question_in_the_same_words_with_no_escape_sequence(tmp_path: Path, rig: Rig, manifest: Path):
    raw, code = on_a_terminal(tmp_path, rig, "visuals", "fetch", str(manifest), "--plain", typed=b"n\n")
    assert code == 2 and rig.requests == 0 and "\x1b" not in raw
    for words in ("Licence conditions: your acceptance is needed", *LINES, "Maker Licence 1.1", NOTICE,
                  "Do you accept these conditions? [y/N]"):
        assert words in raw
    assert "free of" not in raw, "the line about the drive is a terminal's, not a plain command line's"


@needs_pty
def test_with_standard_input_a_terminal_and_output_piped_the_question_goes_to_the_terminal_and_the_pipe_is_unchanged(
        tmp_path: Path, rig: Rig, manifest: Path):
    (tmp_path / "none.toml").write_text("", encoding="utf-8")
    seen, code = run_with_streams(argv("visuals", "fetch", str(manifest)), env=environment(tmp_path, rig), cwd=tmp_path,
                                  terminals=("err",), columns=120, typed=b"y\n")
    assert code == 0 and "Do you accept these conditions? [y/N]" in stripped(seen["tty"])
    assert "Do you accept" not in seen["out"] and "free of" not in seen["out"]
    assert "a made mesh\ncredit: a maker" in seen["out"] and "baked into the asset cache" in seen["out"]


# --- the disk gate at the command line ---------------------------------------------------------------------------------


def test_the_stop_refuses_at_the_command_line_and_force_overrides_only_it(tmp_path: Path, rig: Rig, manifest: Path):
    """A drive that is nearly full (the child is told the cache is on one by a sitecustomize-free patch: the test sets
    ``MIME_ASSETS_DIR`` to a folder and the child's ``shutil.disk_usage`` is wrapped by ``-c``)."""
    code = ("import shutil, collections, sys; U = collections.namedtuple('U', 'total used free'); "
            "shutil.disk_usage = lambda p: U(500*10**9, 500*10**9-2_000_000_000, 2_000_000_000); "
            "from mime.cli import main; sys.exit(main(sys.argv[1:]))")
    (tmp_path / "none.toml").write_text("", encoding="utf-8")

    def run(*words: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-c", code, *words], capture_output=True, text=True, check=False,
                              stdin=subprocess.DEVNULL, env=environment(tmp_path, rig), cwd=tmp_path)

    stopped = run("visuals", "fetch", str(manifest), "--accept-licence", CLOSED)
    assert stopped.returncode == 2 and "asset.disk-low" in stopped.stderr and "--force" in stopped.stderr
    assert rig.requests == 0
    still = run("visuals", "fetch", str(manifest), "--accept-licence", CLOSED, "--yes")
    assert still.returncode == 2 and "asset.disk-low" in still.stderr, "yes does not override the stop"
    refused = run("visuals", "fetch", str(manifest), "--force")
    assert refused.returncode == 2 and "asset.licence-not-accepted" in refused.stderr, "force accepts no licence"
    assert rig.requests == 0
    forced = run("visuals", "fetch", str(manifest), "--accept-licence", CLOSED, "--force")
    assert forced.returncode == 0, forced.stderr
    assert rig.requests == 1


@needs_pty
def test_the_stop_refuses_on_a_terminal_as_well(tmp_path: Path, rig: Rig, manifest: Path):
    code = ("import shutil, collections, sys; U = collections.namedtuple('U', 'total used free'); "
            "shutil.disk_usage = lambda p: U(500*10**9, 500*10**9-2_000_000_000, 2_000_000_000); "
            "from mime.cli import main; sys.exit(main(sys.argv[1:]))")
    (tmp_path / "none.toml").write_text("", encoding="utf-8")
    raw, status = run_on_terminal([sys.executable, "-c", code, "visuals", "fetch", str(manifest), "--accept-licence",
                                   CLOSED], env=environment(tmp_path, rig), cwd=tmp_path, columns=120)
    assert status == 2 and "asset.disk-low" in flat(stripped(raw)) and rig.requests == 0
