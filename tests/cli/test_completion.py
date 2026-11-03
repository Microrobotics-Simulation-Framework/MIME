"""``mime completion bash|zsh|fish``: a script written from ``build_parser()``, checked by the shell that will read it.

Each script is syntax-checked by its shell, and then asked what it completes at every command of the parser: the
commands under it, its flags, a flag's choices, and files where an argument is a path. The answers are compared with what
the parser says (read here by a walk of its own, not the generator's), so a command or flag the script leaves out, or one
it lists that the parser does not hold, is a difference. The controls plant exactly those two errors and require the
comparison to find them, every run.

bash and fish are asked for real (`COMP_WORDS` and the function; `complete -C`). zsh is asked with its completion
builtins (`compadd`, `_describe`, `_files`, `compset`) replaced by functions that print what they were given: that checks
the script's own logic, and not compsys, which wants a terminal (the script was also tried by hand under compinit on a
pseudo-terminal). A shell that is not installed skips its tests on a machine, and fails them under CI, which installs
every one (a skipped test fails a CI lane)."""

import argparse
import functools
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from mime.cli import _completion, build_parser
from mime.cli import main as mime_main

SHELLS = ("bash", "zsh", "fish")
FILES_HERE = {"afile", "adir"}
TIMEOUT = 60


@functools.cache
def shell_path(shell: str) -> str:
    found = shutil.which(shell)
    if found is None:
        if os.environ.get("CI"):
            pytest.fail(f"{shell} is not installed; every CI lane installs bash, zsh and fish for the completion tests")
        pytest.skip(f"{shell} is not installed")
    return found


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A folder holding one file and one folder, where the shell is asked what a path completes to."""
    (tmp_path / "afile").write_text("")
    (tmp_path / "adir").mkdir()
    return tmp_path


# -- what the parser says, read here --------------------------------------------------------------------------------

def _commands_of(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _flag_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [action for action in parser._actions if action.option_strings and action.help != argparse.SUPPRESS]


def _positionals(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [action for action in parser._actions
            if not action.option_strings and not isinstance(action, argparse._SubParsersAction)]


#: Positional arguments that are file names though the parser does not type them as paths.
FILE_NAMED = {(("pull",), "source"), (("assets", "fetch"), "names")}


def _walk(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()):
    yield path, parser
    for name, child in _commands_of(parser).items():
        yield from _walk(child, (*path, name))


def _asked(parser: argparse.ArgumentParser) -> list[tuple[tuple[str, ...], tuple[str, ...], set[str]]]:
    """Every question worth asking a script, as (command's words, the words typed after them, what should complete): at
    each command, an empty word and a lone ``-``, and an empty word after each flag that takes a value."""
    top = _flag_actions(parser)
    asked: list[tuple[tuple[str, ...], tuple[str, ...], set[str]]] = []
    for path, here in _walk(parser):
        flags = [*_flag_actions(here), *(action for action in top if action not in _flag_actions(here))]
        names = {string for action in flags for string in action.option_strings}
        words = set(_commands_of(here))
        positional = False
        for action in _positionals(here):
            words |= {str(choice) for choice in action.choices or ()}
            positional = positional or action.type is Path or (path, action.dest) in FILE_NAMED
        if positional:
            words |= FILES_HERE
        asked += [(path, ("",), words), (path, ("-",), names)]
        for action in flags:
            if action.nargs == 0:
                continue
            value = {str(choice) for choice in action.choices or ()} or (FILES_HERE if action.type is Path else set())
            asked += [(path, (action.option_strings[0], ""), value)]
    return asked


# -- asking a script -----------------------------------------------------------------------------------------------------

def _quoted(words: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(word) for word in words)


def _bash_replies(script: str, cases: list[tuple[str, ...]], workdir: Path) -> list[set[str]]:
    driver = [script, '_ask() { echo "@@"; COMP_WORDS=("$@"); COMP_CWORD=$(($# - 1)); _mime; '
                      'printf "%s\\n" "${COMPREPLY[@]}"; }']
    driver += [f"_ask mime {_quoted(case)}" for case in cases]
    return _split(_run([shell_path("bash"), "-c", "\n".join(driver)], workdir))


def _zsh_replies(script: str, cases: list[tuple[str, ...]], workdir: Path) -> list[set[str]]:
    stubs = """
compdef() { :; }
compadd() { [[ $1 == -- ]] && shift; local item; for item; do [[ $item == $PREFIX* ]] && print -r -- $item; done; return 0; }
_describe() { local -a held; held=("${(@P)4}"); local item; for item in "${(@)held%%:*}"; do [[ $item == $PREFIX* ]] && print -r -- $item; done; }
_files() { print -rl -- *(N); }
compset() { PREFIX=${PREFIX#*=}; }
_ask() { print @@; words=("$@"); CURRENT=$#words; PREFIX=${words[CURRENT]}; _mime; return 0; }
"""
    path = workdir.parent / f"{workdir.name}.zsh"
    path.write_text(stubs + script + "\n" + "\n".join(f"_ask mime {_quoted(case)}" for case in cases) + "\n")
    return _split(_run([shell_path("zsh"), "-f", str(path)], workdir))


def _fish_replies(script: str, cases: list[tuple[str, ...]], workdir: Path) -> list[set[str]]:
    path = workdir.parent / f"{workdir.name}.fish"
    lines = [script, "function _ask", "    echo @@", '    complete -C "$argv"', "end"]
    lines += [f"_ask {shlex.quote('mime ' + ' '.join(case))}" for case in cases]
    path.write_text("\n".join(lines) + "\n")
    done = _run([shell_path("fish"), "--no-config", str(path)], workdir)
    return [{line.split("\t")[0].rstrip("/") for line in block} for block in _blocks(done)]


def _run(command: list[str], workdir: Path) -> str:
    done = subprocess.run(command, capture_output=True, text=True, cwd=workdir, check=False, timeout=TIMEOUT)
    assert done.returncode == 0 and done.stderr == "", (done.returncode, done.stderr)
    return done.stdout


def _blocks(output: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for line in output.splitlines():
        if line == "@@":
            blocks.append([])
        elif line:
            blocks[-1].append(line)
    return blocks


def _split(output: str) -> list[set[str]]:
    return [set(block) for block in _blocks(output)]


REPLIES = {"bash": _bash_replies, "zsh": _zsh_replies, "fish": _fish_replies}


def differences(shell: str, script: str, parser: argparse.ArgumentParser, workdir: Path) -> list[str]:
    """What ``script`` completes differently from what ``parser`` holds, one line each (empty when they agree)."""
    asked = _asked(parser)
    found = REPLIES[shell](script, [(*path, *typed) for path, typed, _ in asked], workdir)
    assert len(found) == len(asked)
    problems: list[str] = []
    for (path, typed, expected), reply in zip(asked, found, strict=True):
        if reply != expected:
            problems.append(f"mime {' '.join((*path, *typed))!r}: missing {sorted(expected - reply)}, "
                            f"extra {sorted(reply - expected)}")
    return problems


# -- the tests -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shell", SHELLS)
def test_the_script_is_valid_for_its_shell(shell, tmp_path):
    text = _completion.script(shell, build_parser())
    path = tmp_path / f"mime.{shell}"
    path.write_text(text)
    check = {"bash": ["-n"], "zsh": ["-n"], "fish": ["--no-execute"]}[shell]
    done = subprocess.run([shell_path(shell), *check, str(path)], capture_output=True, text=True, check=False, timeout=TIMEOUT)
    assert done.returncode == 0 and done.stderr == "", done.stderr


@pytest.mark.parametrize("shell", SHELLS)
def test_every_command_flag_and_choice_the_parser_holds_completes_and_nothing_else_does(shell, workdir):
    parser = build_parser()
    assert differences(shell, _completion.script(shell, parser), parser, workdir) == []


@pytest.mark.parametrize("shell", SHELLS)
def test_a_typed_prefix_completes_to_the_commands_it_starts(shell, workdir):
    """The example in the design: ``mime pu`` completes to ``pull`` and ``push``."""
    found = REPLIES[shell](_completion.script(shell, build_parser()), [("pu",), ("pus",), ("pul",)], workdir)
    assert found == [{"pull", "push"}, {"push"}, {"pull"}]


@pytest.mark.parametrize("shell", SHELLS)
def test_a_flags_choices_complete_after_an_equals_sign_too(shell, workdir):
    """bash hands the function ``--to``, ``=`` and the rest as three words (COMP_WORDBREAKS); zsh and fish as one."""
    typed = {
        "bash": [("push", "--to", "="), ("push", "--to", "=", "zen"), ("push", "--to", "=", "zenodo-s")],
        "zsh": [("push", "--to="), ("push", "--to=zen"), ("push", "--to=zenodo-s")],
        "fish": [("push", "--to="), ("push", "--to=zen"), ("push", "--to=zenodo-s")],
    }[shell]
    found = REPLIES[shell](_completion.script(shell, build_parser()), typed, workdir)
    both = {"--to=zenodo", "--to=zenodo-sandbox"} if shell == "fish" else {"zenodo", "zenodo-sandbox"}
    expected = [both, both, {"--to=zenodo-sandbox"} if shell == "fish" else {"zenodo-sandbox"}]
    assert found == expected


@pytest.mark.parametrize("shell", SHELLS)
def test_control_a_flag_added_to_the_parser_after_the_script_was_written_is_found(shell, workdir):
    """The comparison can fail: it names a flag the script predates, every run."""
    parser = build_parser()
    script = _completion.script(shell, parser)
    _commands_of(parser)["run"].add_argument("--planted-after", action="store_true")
    problems = differences(shell, script, parser, workdir)
    assert any("--planted-after" in line and "mime 'run -'" in line for line in problems), problems


@pytest.mark.parametrize("shell", SHELLS)
def test_control_a_command_left_out_of_the_generator_is_found(shell, workdir):
    """The same, for a command the generator never saw, and for its own flags."""
    parser = build_parser()
    held = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    left_out = held.choices.pop("pull")
    script = _completion.script(shell, parser)
    held.choices["pull"] = left_out
    problems = differences(shell, script, parser, workdir)
    assert any("pull" in line and "missing" in line for line in problems), problems
    assert any("'pull -'" in line and "--signed-in" in line for line in problems), problems


def test_each_listed_file_argument_is_an_argument_the_parser_has():
    parser = build_parser()
    walked = {path: here for path, here in _walk(parser)}
    for path, dest in _completion.ALSO_FILES:
        assert any(action.dest == dest for action in _positionals(walked[path])), (path, dest)
    assert _completion.ALSO_FILES == FILE_NAMED


@pytest.mark.parametrize("shell", SHELLS)
def test_a_tab_runs_no_python_and_starts_no_program(shell, workdir, monkeypatch):
    """The script is static, so a Tab does not import JAX: it asks its shell's own builtins alone, and answers the
    same with no program on PATH to start (a call to Python would find none)."""
    parser = build_parser()
    text = _completion.script(shell, parser)
    shell_path(shell)
    monkeypatch.setenv("PATH", "")
    assert differences(shell, text, parser, workdir) == []


@pytest.mark.parametrize("shell", SHELLS)
def test_the_command_prints_the_script_and_only_that_in_plain_ascii(shell, capsys):
    assert mime_main(["completion", shell]) == 0
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == _completion.script(shell, build_parser())
    assert captured.out.isascii() and "\x1b" not in captured.out and captured.out.endswith("\n")


def test_the_help_says_where_each_script_goes_a_line_each_and_shows_an_example(capsys):
    with pytest.raises(SystemExit) as stopped:
        mime_main(["completion", "--help"])
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    for line in ("  bash: mime completion bash > ~/.local/share/bash-completion/completions/mime",
                 "  zsh: mime completion zsh > ~/.zfunc/_mime",
                 "  fish: mime completion fish > ~/.config/fish/completions/mime.fish"):
        assert line in out.splitlines() or any(each.startswith(line) for each in out.splitlines()), line
    assert "example:\n  mime completion bash > " in out
    assert str(Path.home()) not in out


def test_a_shell_it_does_not_write_for_is_refused_and_a_missing_one_is_asked_for(capsys):
    for words in (["completion", "powershell"], ["completion"]):
        with pytest.raises(SystemExit) as stopped:
            mime_main(words)
        assert stopped.value.code == 2
    assert "powershell" in capsys.readouterr().err


def test_the_installed_command_pipes_the_same_bytes_and_nothing_on_the_error_stream(tmp_path):
    """The command as a user runs it, once (each start imports MIME): through a pipe, in a process of its own."""
    command = shutil.which("mime")
    if command is None:
        pytest.skip("the mime command is not on PATH")
    done = subprocess.run([command, "completion", "bash"], capture_output=True, text=True, cwd=tmp_path, check=False,
                          timeout=TIMEOUT * 3)
    assert done.returncode == 0 and done.stderr == ""
    assert done.stdout == _completion.script("bash", build_parser())
